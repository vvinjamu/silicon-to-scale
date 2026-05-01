# Chapter 7 — GPU Kernels and CUDA Optimization
## AI/ML Infrastructure from Silicon to Scale
### Production Source Pack — Markdown

**Author:** Venkat Vinjam  
**Book:** AI/ML Infrastructure from Silicon to Scale  
**Chapter slug:** `ch07_gpu_kernels_cuda_optimization`  
**Edition note:** Current as of 2026 edition  
**Production status:** Reader-facing production source with separated production notes

---

## Confidence Label Key

| Label | Meaning |
|---|---|
| [SHIPPED] | Verified behavior from shipping hardware/software or official documentation. |
| [ANNOUNCED] | Vendor-announced or published but not yet broadly production-proven. |
| [DERIVED FROM SHIPPED] | Formula or conclusion derived from shipped specs or stable performance models. |
| [ESTIMATED] | Engineering estimate; methodology should be shown. |
| [REPRESENTATIVE] | Illustrative example, not a universal benchmark result. |
| [ENV-SPECIFIC] | Depends strongly on model, shape, framework, kernel version, runtime configuration, or cluster environment. |

---

## Chapter Overview

Chapter 7 moves from workload-level reasoning into the smallest unit of GPU execution that a performance engineer can directly reason about: the **kernel**. A model layer, PyTorch operation, or inference runtime step eventually becomes one or more GPU kernels. Those kernels consume registers, shared memory, cache bandwidth, HBM bandwidth, scheduler slots, Tensor Core issue bandwidth, and launch overhead.

This chapter is not a full CUDA programming textbook. Its goal is to build the principal-level mental model required to answer these questions:

1. Is this kernel worth optimizing?
2. Is the bottleneck memory movement, math throughput, launch overhead, dependency stalls, occupancy, or communication overlap?
3. Should the fix be a library call, layout change, fused kernel, compiler path, Triton prototype, CUDA kernel, or no action?
4. How do you prove the optimization improved the **end-to-end workload**, not only a microbenchmark?

A principal AI/ML performance architect does not optimize kernels because kernels are interesting. They optimize kernels only when the kernel is on the critical path of a meaningful metric: training step time, tokens/sec/GPU, TTFT, TPOT, P95/P99 latency, cost per token, or fleet utilization.

---

## 7.0 Chapter Manifesto — Kernel Optimization Is a Systems Decision

GPU kernel optimization sits at a dangerous point in the stack. It is technically deep enough to attract engineering attention, but local enough to mislead teams into optimizing the wrong thing. A 2× speedup to a kernel that consumes 1% of wall time improves total runtime by at most about 0.5%. That is not a strategy; it is a distraction. [DERIVED FROM SHIPPED]

The right kernel question is not:

> “Can I make this kernel faster?”

The right principal-level question is:

> “Will improving this kernel move the workload-level bottleneck enough to justify the engineering cost and maintenance risk?”

That question requires three layers of evidence:

1. **Timeline evidence:** The kernel is on the critical path, not hidden behind another bottleneck.
2. **Counter evidence:** The kernel has a specific bottleneck classification.
3. **System evidence:** Fixing it improves an end-to-end workload metric under realistic conditions.

This chapter teaches that evidence chain.

### Table 7.1 — Kernel Bottleneck Taxonomy

| Bottleneck | Typical symptom | Profiler signal | Likely cause | First fix | Confidence |
|---|---|---|---|---|---|
| Memory bandwidth | High bytes moved, low math utilization | High HBM traffic relative to useful FLOPs | Poor reuse, unfused ops, strided loads | Coalesce, fuse, tile, reduce bytes | [ENV-SPECIFIC] |
| Compute / Tensor Core | Math pipes are busy but wall time is still large | High Tensor Core or FP pipe utilization | True math-heavy GEMM/attention work | Use vendor libraries, verify precision/layout/alignment | [ENV-SPECIFIC] |
| Launch overhead | Many small kernels separated by host/runtime gaps | Nsight Systems shows CPU-side gaps | Many tiny ops, Python/runtime dispatch | Fuse, compiler capture, CUDA Graphs where applicable | [ENV-SPECIFIC] |
| Latency / dependency | Low bandwidth and low compute despite stalls | Warp stall metrics dominate | Dependent loads, insufficient independent work | Improve tiling, prefetch, increase independent work | [ENV-SPECIFIC] |
| Occupancy-limited | Too few resident warps or blocks | Occupancy constrained by registers or shared memory | Register pressure, large shared-memory allocation | Tune block size, reduce register pressure, change tile shape | [SHIPPED] |
| Communication overlap | GPU compute finishes, then collectives run serially | Timeline shows NCCL/RCCL after compute | Poor stream scheduling or bucket timing | Overlap communication, tune buckets, adjust topology placement | [ENV-SPECIFIC] |

**Key idea:** classify before changing code. Random tuning creates fragile wins; bottleneck-driven tuning creates reproducible wins.

---

## 7.1 The GPU Kernel Execution Model

A GPU kernel is a program launched across many parallel threads. In CUDA terminology, a kernel launch creates a **grid**. The grid contains **thread blocks**, also called CTAs. Each block contains threads. Threads are scheduled in groups called **warps** on NVIDIA GPUs, with 32 threads per warp. [SHIPPED]

The hierarchy matters because different levels have different synchronization and memory rules:

- A **thread** owns its registers.
- A **warp** is the scheduling granularity for many execution behaviors.
- A **block/CTA** can use shared memory and block-level synchronization.
- A **grid** spans the full kernel launch but does not provide cheap global synchronization inside a normal kernel.
- An **SM** schedules resident blocks and warps subject to resource limits.

> **Figure 7.1 — GPU Kernel Execution Hierarchy**  
> _Placeholder: Grid → CTA/block → warp → thread hierarchy, with blocks assigned to SMs and warps shown as 32-thread scheduling groups._  
> **Caption:** A CUDA kernel launches a grid of blocks. Each block runs on an SM as one or more warps. The hierarchy determines scheduling, memory access, synchronization, and occupancy.  
> **Placement:** §7.1 The GPU Kernel Execution Model.  
> **Key takeaway:** Kernel performance starts with execution mapping: enough independent work must exist to fill SMs and hide latency.  
> **Production note:** Create or adapt from `diagrams_batch3.html#d27`; keep NVIDIA-specific warp wording separate from AMD wavefront terminology.

### Blocks, warps, and SM residency

When a kernel launches, blocks are distributed to SMs. An SM can host multiple resident blocks if it has enough registers, shared memory, scheduler resources, and block slots. The GPU then schedules warps from those resident blocks. If one warp stalls on memory, another warp can issue work. This is the central latency-hiding model of GPU execution. [SHIPPED]

The performance implication is important: GPUs tolerate long memory latency not by making each thread fast, but by running enough independent warps that useful work is available while other warps wait.

### CUDA vs portable mental model

This chapter uses CUDA terms because most public documentation, profiling tools, and interview questions use CUDA vocabulary. The mental model transfers to other accelerators, but exact details differ:

- NVIDIA uses warps of 32 threads. [SHIPPED]
- AMD uses wavefront terminology; wave size and compiler/runtime behavior may differ by architecture and mode. [ENV-SPECIFIC]
- Vendor profilers use different metric names and counter availability. [ENV-SPECIFIC]

The portable idea is not “warp size equals 32 everywhere.” The portable idea is: **hardware schedules groups of lanes, and your memory/control-flow pattern determines whether those lanes do useful work together.**

---

## 7.2 Memory Coalescing — The First Rule of GPU Performance

Many slow kernels do not perform too many FLOPs. They move data in a pattern that prevents the memory system from serving it efficiently.

A coalesced global-memory access occurs when neighboring lanes in a warp access neighboring addresses. This allows the memory system to serve the warp with fewer, wider memory transactions. A scattered or strided pattern can require more transactions for the same useful data, wasting bandwidth. [SHIPPED]

> **Figure 7.2 — Memory Coalescing: Consecutive Threads, Consecutive Addresses**  
> _Placeholder: Good case shows lanes 0–31 reading contiguous addresses. Bad case shows lanes reading strided/scattered addresses across cache lines._  
> **Caption:** Coalesced memory access lets neighboring threads in a warp load neighboring addresses, minimizing wasted memory transactions. Strided or scattered access wastes bandwidth even when arithmetic work is unchanged.  
> **Placement:** §7.2 Memory Coalescing.  
> **Key takeaway:** Memory bandwidth is not just a spec-sheet number; access pattern determines how much bandwidth a kernel can actually use.  
> **Production note:** Use architecture-agnostic transaction wording; avoid claiming one fixed transaction size for all GPUs.

### Whiteboard rule

For the first pass, ask:

```text
lane i should usually access address base + i
```

That rule is not universal, but it catches many performance bugs. If lane 0 reads element 0, lane 1 reads element 1024, lane 2 reads element 2048, and so on, the memory system sees scattered demand. If each lane reads a nearby element, the memory system can combine work more efficiently. [SHIPPED]

### Coalescing in transformer workloads

Coalescing shows up in transformer systems in several ways:

- Tensor layouts that force transposed or strided reads before GEMM.
- KV cache layouts that make attention reads less contiguous.
- Unfused elementwise chains that repeatedly read and write full tensors.
- Custom kernels that treat logical layout as if it were physical layout.

The principal lesson is simple: before inventing a new algorithm, verify that the existing kernel reads and writes memory in a pattern the hardware can serve efficiently.

---

## 7.3 Shared Memory, Registers, and Bank Conflicts

The GPU memory hierarchy was introduced earlier in the book. Chapter 7 focuses on what that hierarchy means for kernel implementation.

### Table 7.2 — CUDA Memory Spaces and What They Are Good For

| Memory space | Scope | Typical use | Risk | Confidence |
|---|---|---|---|---|
| Registers | Per thread | Fragments, accumulators, loop variables | Register pressure can reduce occupancy | [SHIPPED] |
| Shared memory | Per block/CTA | Reused tiles, staging, reductions | Bank conflicts, synchronization overhead, limited capacity | [SHIPPED] |
| L1 / texture cache | Per SM behavior, architecture-dependent | Cached global/local accesses | Hit rate depends on access pattern | [ENV-SPECIFIC] |
| L2 cache | GPU-wide cache | Cross-SM reuse, global-memory locality | Capacity pressure and eviction | [ENV-SPECIFIC] |
| HBM / global memory | GPU-wide device memory | Tensors, model weights, activations, KV cache | Bandwidth and capacity bottleneck | [SHIPPED] |
| Host memory | CPU/system memory | Data staging, offload, input pipeline | Transfer latency and bandwidth limit | [ENV-SPECIFIC] |

The hierarchy follows a tradeoff: faster memory is smaller and more explicit; larger memory is slower and easier to overuse.

### Registers

Registers are the fastest storage visible to a thread. Accumulators in GEMM fragments, loop variables, and temporary values live here. But registers are not free. If each thread uses too many registers, fewer warps can be resident on the SM, reducing latency hiding. [SHIPPED]

### Shared memory

Shared memory is explicitly managed scratchpad memory shared by threads in a block. It is useful when data is reused by multiple threads, as in tiled GEMM, reductions, scans, and staging patterns. It is not automatically faster in every situation because staging has overhead: loads, stores, synchronization, bank layout constraints, and occupancy pressure. [ENV-SPECIFIC]

> **Figure 7.5 — Shared Memory Bank Conflicts: When Fast Memory Serializes**  
> _Placeholder: 32 conceptual shared-memory banks. Good path maps lanes to distinct banks. Bad path maps many lanes to the same bank, causing serialization._  
> **Caption:** Shared memory is fast when a warp accesses distinct banks or broadcast-friendly addresses. Conflicting bank access patterns can serialize requests and reduce effective bandwidth.  
> **Placement:** §7.3 Shared Memory, Registers, and Bank Conflicts.  
> **Key takeaway:** Shared memory is a tool for data reuse, but bad access patterns can turn it into a bottleneck.  
> **Production note:** Label conflict behavior as architecture-dependent and cite CUDA documentation in the final bibliography/source notes.

### Bank conflicts

Shared memory is divided into banks. When multiple lanes in a warp access different addresses that map to the same bank, requests may serialize, reducing effective bandwidth. Broadcast-style cases and exact bank behavior vary by architecture, so use this as a design model rather than a fixed latency formula. [SHIPPED]

Common mitigations include:

- Padding shared-memory arrays.
- Changing tile layout.
- Transposing access order.
- Using library kernels that already solve the layout problem.

The last bullet matters. If cuBLAS, cuDNN, CUTLASS, or a framework kernel already implements the pattern well, the best optimization may be to route work into that path instead of writing a fragile custom kernel.

---

## 7.4 Warp Divergence, Occupancy, and Latency Hiding

A warp is efficient when lanes execute the same instruction path on useful data. If lanes take different branches, execution can serialize across paths. This is warp divergence. [SHIPPED]

Warp divergence matters most when:

- Branches are data-dependent and vary across lanes.
- The divergent region does substantial work.
- The kernel is already latency-sensitive or underutilized.

It matters less when the branch is uniform across the warp, the divergent region is tiny, or the kernel is dominated by a different bottleneck. [ENV-SPECIFIC]

### Occupancy is not utilization

Occupancy is the ratio of active warps resident on an SM to the maximum possible resident warps. It is a measure of scheduling capacity, not proof that useful work is happening. [SHIPPED]

High occupancy can help hide latency. But maximum occupancy is not always optimal. Some high-performance kernels intentionally use more registers or shared memory per block to increase data reuse, even if occupancy drops. [ENV-SPECIFIC]

> **Figure 7.6 — Occupancy Is Limited by Threads, Registers, Shared Memory, and Blocks**  
> _Placeholder: SM resource budget showing register file, shared memory, resident block slots, resident warp slots, and two example kernels: register-limited and shared-memory-limited._  
> **Caption:** Occupancy is constrained by multiple per-SM resources. More occupancy is useful only until enough independent work exists to hide latency.  
> **Placement:** §7.4 Warp Divergence, Occupancy, and Latency Hiding.  
> **Key takeaway:** Treat occupancy as a diagnostic clue, not a performance objective by itself.  
> **Production note:** Exact limits vary by architecture and compilation choices.

### Table 7.6 — Occupancy Limiters and Mitigation Options

| Limiter | Symptom | Mitigation | Confidence |
|---|---|---|---|
| Registers per thread | Low resident warps due to register pressure | Reduce unrolling, split kernel, tune tile size | [SHIPPED] |
| Shared memory per block | Few blocks resident per SM | Smaller tiles, double buffering only when beneficial | [SHIPPED] |
| Threads per block | Too few or too many active warps | Tune block shape and work per thread | [ENV-SPECIFIC] |
| Blocks per SM | Resident CTA limit reached | Adjust block granularity | [ENV-SPECIFIC] |
| Long dependency chains | Warps resident but stalled | Increase independent work, prefetch, restructure computation | [ENV-SPECIFIC] |

### Interview-safe wording

Do not say:

> “I would maximize occupancy.”

Say:

> “I would check whether occupancy is the limiting factor. If low occupancy is caused by register or shared-memory pressure and the kernel is latency-bound, I would tune tile shape or resource use. If lower occupancy buys much better reuse, I may keep it.”

That answer separates senior-level metric awareness from principal-level tradeoff judgment.

---

## 7.5 Tensor Cores, GEMM, and Library-First Optimization

Transformer models are GEMM-heavy. Linear projections, MLP layers, QKV projection, output projection, and many batched operations reduce to matrix multiply patterns. For matrix multiplication:

```text
C[M, N] = A[M, K] × B[K, N]
FLOPs_GEMM ≈ 2 × M × N × K
```

Confidence label: [DERIVED FROM SHIPPED]

The factor of 2 comes from multiply-add counting. This is a convention; be explicit in interviews.

### Tensor Cores conceptually

Tensor Cores accelerate matrix multiply-accumulate operations for supported precision and layout combinations. At the kernel level, the code does not perform one scalar multiply at a time. It loads matrix fragments, performs matrix multiply-accumulate instructions, and accumulates output fragments. [SHIPPED]

This is why GEMM optimization is primarily a data-reuse problem. The kernel wants to move tiles from HBM to on-chip storage, reuse those tiles many times, and keep Tensor Cores fed.

> **Figure 7.3 — GEMM Tiling: HBM → Shared Memory → Registers → Tensor Cores**  
> _Placeholder: Matrices A and B loaded as tiles from HBM into shared memory, fragments moved into registers, Tensor Cores perform MMA, output tile accumulated into C._  
> **Caption:** High-performance GEMM is a data-reuse problem. Tiles are loaded from HBM into shared memory, fragments are staged into registers, Tensor Cores perform matrix multiply-accumulate, and output tiles are written back.  
> **Placement:** §7.5 Tensor Cores, GEMM, and Library-First Optimization.  
> **Key takeaway:** GEMM is fast because it turns memory traffic into repeated compute through tiling and reuse.  
> **Production note:** Label this as conceptual; real kernels vary by architecture, library, precision, and shape.

### Table 7.5 — GEMM Tuning Levers and When They Matter

| Lever | Why it matters | First check | Confidence |
|---|---|---|---|
| Matrix dimensions | Tensor Core kernels prefer hardware-friendly tile shapes | Are dimensions aligned for the intended library path? | [ENV-SPECIFIC] |
| Precision | BF16, FP16, TF32, FP8, INT8 map to different hardware/library paths | Is the intended precision actually active? | [SHIPPED] |
| Layout | Strides/transposes affect memory coalescing and library selection | Are tensors contiguous or in the expected layout? | [ENV-SPECIFIC] |
| Batch size | Small batches may underutilize compute | Can batching increase arithmetic intensity? | [ENV-SPECIFIC] |
| Library path | Vendor kernels are heavily optimized | Is cuBLAS/cuBLASLt/CUTLASS or framework equivalent being used? | [SHIPPED] |
| Fusion | Epilogues can create extra memory traffic | Can bias, activation, residual, or normalization be fused? | [ENV-SPECIFIC] |

### Library-first rule

For GEMM, custom kernel work is rarely the first move. The usual order is:

1. Verify the workload is actually GEMM-bound.
2. Verify dimensions, layout, and precision route into the intended Tensor Core path.
3. Try vendor libraries and framework-supported fused epilogues.
4. Use CUTLASS or Triton for specialized shapes or epilogues.
5. Write custom CUDA only when the shape, fusion, or scheduling requirement justifies it.

A principal engineer earns trust by avoiding expensive custom work until simpler high-leverage options are exhausted.

---

## 7.6 FlashAttention — IO-Aware Kernel Design

Naive attention can materialize an `S × S` score matrix in HBM. For long sequences, that intermediate becomes extremely expensive. FlashAttention changes the memory schedule: it tiles Q, K, and V; keeps intermediate state on-chip; uses an online softmax formulation; and writes the final output without storing the full attention-score matrix in HBM. The mathematical attention result is exact for the supported formulation; the memory traffic is reduced. [SHIPPED]

Do not describe FlashAttention as simply “changing attention from O(S²) to O(S).” That statement is misleading. The compute still has attention-dependent terms. The safer statement is:

> FlashAttention reduces HBM IO by avoiding materialization of the full `S × S` score matrix while preserving exact attention semantics for supported attention patterns. [SHIPPED]

> **Figure 7.4 — FlashAttention: Compute Attention Without Materializing the S×S Score Matrix in HBM**  
> _Placeholder: Left side naive attention path writes QKᵀ scores to HBM. Right side FlashAttention path streams Q tiles over K/V tiles, performs online softmax, and writes final output only._  
> **Caption:** FlashAttention tiles Q, K, and V so attention can be computed through on-chip memory without writing the full `S × S` attention matrix to HBM.  
> **Placement:** §7.6 FlashAttention — IO-Aware Kernel Design.  
> **Key takeaway:** FlashAttention is the canonical example of IO-aware algorithm design: same mathematical output, less HBM traffic.  
> **Production note:** Existing `diagrams_batch1.html#d5` can be adapted for Ch07.

### The memory-traffic argument

The simple model is:

```text
Naive attention path:
Q, K, V read from HBM
S×S attention scores written to HBM
S×S scores read back for softmax / value weighting
Output written to HBM

FlashAttention-style path:
Q/K/V tiles streamed through on-chip memory
Online softmax state maintained per tile
Final output written to HBM
```

Confidence label: [REPRESENTATIVE]

Actual implementation details depend on architecture, sequence length, precision, mask type, head dimension, batch shape, and framework integration. [ENV-SPECIFIC]

### Why this belongs in a kernel chapter

FlashAttention is not just an attention algorithm. It is the best teaching example of how kernel design changes system behavior. It shows that a kernel can produce the same mathematical answer while changing:

- HBM traffic.
- Intermediate allocation size.
- Long-context feasibility.
- Serving latency.
- Training memory pressure.
- Whether attention becomes the bottleneck.

That is why FlashAttention appears here, and KV-cache serving appears later in Chapter 11.

---

## 7.7 FlashAttention-2, FlashAttention-3, and Hardware-Specific Kernels

FlashAttention evolved because GPUs evolved. New GPU generations expose new scheduling, memory movement, and matrix-instruction capabilities. A production chapter should not promise universal speedups; it should teach the reader to ask whether their stack, precision, sequence length, and GPU generation use the intended path.

### Table 7.4 — FlashAttention Version Comparison

| Version | Main idea | Best suited for | Safe performance wording | Confidence |
|---|---|---|---|---|
| FlashAttention | IO-aware exact attention, tiled to reduce HBM traffic | Long sequence attention | Reduces memory traffic versus naive attention implementations that materialize full score matrices | [SHIPPED] |
| FlashAttention-2 | Better parallelism and work partitioning | A100/H100-class deployments and many training/inference stacks | Often faster than v1; exact gain varies by shape and implementation | [ENV-SPECIFIC] |
| FlashAttention-3 | Hopper-specific asynchrony, WGMMA/TMA usage, and FP8 path | H100/Hopper-class GPUs | Paper and release materials report 1.5–2.0× FP16 speedups versus FlashAttention-2, up to about 740 TFLOP/s FP16, and close to 1.2 PFLOP/s FP8 on H100; treat as reported benchmark results | [ENV-SPECIFIC] |

The principal-level framing is:

> “I would first verify which attention kernel is active for my sequence length, head dimension, precision, mask, and GPU. Then I would compare it under the workload’s real prefill/decode or training-step mix.”

This avoids the common mistake of quoting a kernel benchmark as if it directly predicts service-level latency or training MFU.

---

## 7.8 Fused Kernels and Memory-Traffic Reduction

Fusion combines multiple operations into one kernel so intermediate tensors do not need to be written to and read from HBM. This is especially valuable for memory-bandwidth-sensitive operations such as normalization, activation, bias add, residual add, dropout, and simple elementwise chains. [ENV-SPECIFIC]

A simple traffic model:

```text
Unfused two-op chain:
read input
write intermediate
read intermediate
write output

Fused chain:
read input
write output
```

Confidence label: [REPRESENTATIVE]

This model is not a benchmark promise. Cache behavior, compiler fusion, tensor layout, operation shape, and launch overhead all matter. But it explains why fusion is often powerful: it removes memory round trips and reduces launch count.

### Common fused patterns in AI workloads

- Bias + activation.
- Residual add + normalization.
- Dropout + residual + layer norm in training stacks.
- QKV projection fusion at higher framework/library levels.
- Optimizer update fusion.
- Attention kernels that fuse score computation, softmax, and value weighting.

Fusion can improve performance, but it can also reduce flexibility. A fused kernel may be harder to debug, harder to support across shapes, and harder to maintain across hardware generations. [ENV-SPECIFIC]

---

## 7.9 Triton as a Productivity Layer for Custom Kernels

Triton is a language and compiler for writing GPU kernels with a Python-like programming model. It is especially useful for custom or fused tensor programs where standard framework operations or vendor libraries do not already provide the right optimized path. [SHIPPED]

Triton does not remove the need to understand GPU performance. It changes the productivity curve:

- You still reason about blocks/programs, memory loads/stores, masks, tiling, and vectorized operations.
- You still measure HBM traffic, occupancy, and achieved throughput.
- You still compare against library baselines.
- You may get to a high-quality prototype faster than hand-written CUDA.

### When Triton is a good first custom-kernel move

Triton is a strong candidate when:

- The operation is a fused elementwise/reduction pattern not covered by existing libraries.
- The shape is stable enough to tune.
- The kernel is on the critical path.
- The team needs iteration speed.
- Portability across CUDA and ROCm-style backends matters, with validation. [ENV-SPECIFIC]

Triton is usually not the first move for mainstream GEMM, convolution, or already-optimized attention paths. For those, vendor libraries and framework kernels should be the baseline.

---

## 7.10 Kernel Profiling Workflow: nsys → ncu → Fix → Verify

Nsight Systems and Nsight Compute answer different questions.

- **Nsight Systems** answers: Where did time go across the application timeline?
- **Nsight Compute / ncu** answers: Why did this selected CUDA kernel behave this way?

The most common mistake is starting with `ncu` before proving the kernel matters.

> **Figure 7.7 — The Kernel Profiling Workflow**  
> _Placeholder: System symptom → Nsight Systems timeline → identify critical kernel/gap → Nsight Compute counters → classify bottleneck → apply fix → re-run workload benchmark. Include stop branch for “kernel not on critical path.”_  
> **Caption:** Start with Nsight Systems to find where time goes. Use Nsight Compute only after a specific kernel is proven important. Then fix one hypothesis and verify end-to-end impact.  
> **Placement:** §7.10 Kernel Profiling Workflow.  
> **Key takeaway:** Nsight Systems answers “where did time go?” Nsight Compute answers “why was this kernel slow?”  
> **Production note:** Add a reproducibility note because profiler settings can perturb performance.

### Table 7.3 — Nsight Compute Metrics: What to Look At First

| Metric family | What it indicates | When to care | Typical response | Confidence |
|---|---|---|---|---|
| Tensor/FP pipe utilization | Whether compute units are busy | GEMM and attention kernels | Check precision, layout, Tensor Core path, library selection | [ENV-SPECIFIC] |
| DRAM/HBM throughput | Whether memory bandwidth is limiting | Elementwise, normalization, attention, copy-heavy kernels | Coalesce, fuse, tile, reduce bytes | [ENV-SPECIFIC] |
| L2 hit rate / cache throughput | Reuse across accesses | Reused weights, activations, KV blocks | Improve locality or layout | [ENV-SPECIFIC] |
| Warp stall reasons | Why warps cannot issue | Any slow selected kernel | Map dominant stall to a specific hypothesis | [ENV-SPECIFIC] |
| Occupancy / launch statistics | Active warps and resource limits | Suspected latency-hiding problem | Tune registers, shared memory, or block size | [SHIPPED] |
| Memory workload analysis | Load/store pattern quality | Suspected bad coalescing | Reorder layout or access pattern | [SHIPPED] |

Exact metric names change across Nsight Compute versions and GPU architectures. Treat the table as a starting map, not a literal metric-name contract. [ENV-SPECIFIC]

### Profiling checklist

1. Reproduce the workload with fixed input shapes, seed, batch size, precision, and runtime configuration.
2. Capture a system timeline.
3. Identify the top wall-time regions and idle gaps.
4. Select only critical kernels for detailed counter profiling.
5. Build a bottleneck hypothesis before changing code.
6. Apply one fix at a time.
7. Re-measure both microbenchmark and end-to-end workload.
8. Report the metric that matters: step time, tokens/sec/GPU, TTFT, TPOT, P95/P99, or cost per token.

---

## 7.11 Should You Write a Custom Kernel?

Custom kernels are powerful, but expensive. They create maintenance burden, portability risk, testing load, shape coverage issues, numerical risk, and future hardware risk. A principal engineer protects the team from custom work unless the leverage is clear.

> **Figure 7.8 — Should You Write a Custom Kernel?**  
> _Placeholder: Decision tree: Is operation >10% of end-to-end time? Is a library path available? Is layout/precision correct? Can fusion/compiler/CUDA Graphs solve it? Is shape stable? Then consider Triton/CUDA custom work._  
> **Caption:** A custom kernel is justified only after library, layout, batching, fusion, compiler, and configuration options are ruled out or shown insufficient.  
> **Placement:** §7.11 Principal-Level Optimization Prioritization.  
> **Key takeaway:** The highest-leverage kernel optimization is often to avoid writing a kernel.  
> **Production note:** Keep compiler details shallow here; Chapter 9 covers compiler internals.

### Table 7.7 — Should This Kernel Be Optimized?

| Situation | Optimize kernel now? | Better first move | Priority | Confidence |
|---|---|---|---|---|
| Kernel is less than 1% of wall time | Usually no | Ignore or batch with nearby work | P2 | [DERIVED FROM SHIPPED] |
| Kernel is 10–30% and memory-bound | Maybe | Check fusion, coalescing, layout, library path | P1/P0 | [ENV-SPECIFIC] |
| GEMM underperforms | Usually not custom first | Verify cuBLAS/cuBLASLt precision/layout/alignment | P0 | [SHIPPED] |
| Many small kernels with CPU gaps | Do not tune each kernel separately | CUDA Graphs, compiler capture, fusion | P0 | [ENV-SPECIFIC] |
| FlashAttention disabled or unsupported path active | No custom attention yet | Enable supported attention backend first | P0 | [ENV-SPECIFIC] |
| Unique op unsupported by libraries and on critical path | Yes, possibly | Triton prototype, then CUDA/CUTLASS if justified | P1 | [ENV-SPECIFIC] |

### Amdahl framing

If a kernel consumes fraction `p` of total wall time and you speed it up by factor `s`, total speedup is:

```text
Speedup_total = 1 / ((1 - p) + p / s)
```

Confidence label: [DERIVED FROM SHIPPED]

Example: if `p = 0.01` and `s = 2`, then:

```text
Speedup_total = 1 / (0.99 + 0.01 / 2) ≈ 1.005
```

That is about 0.5% improvement. This is why the first optimization skill is not CUDA. It is prioritization.

---

## 7.12 Common Kernel Optimization Anti-Patterns

### Anti-pattern 1 — Starting with Nsight Compute

If you do not know whether the kernel is on the critical path, detailed counters can waste time. Start with timeline-level evidence.

### Anti-pattern 2 — Maximizing occupancy blindly

Maximum occupancy can reduce performance if it forces smaller tiles, less reuse, or worse memory behavior. Occupancy is a means to hide latency, not an objective by itself.

### Anti-pattern 3 — Writing custom GEMM too early

Mainstream GEMM paths are heavily optimized. If a GEMM is slow, first check shape, layout, precision, batching, and library path.

### Anti-pattern 4 — Treating microbenchmark gains as production wins

A kernel benchmark can improve while the service-level metric stays flat because the bottleneck moved to launch overhead, queueing, communication, or memory allocation.

### Anti-pattern 5 — Ignoring numerical and maintenance risk

Kernel changes can introduce precision differences, shape-specific bugs, portability issues, and hidden performance cliffs. Production optimization requires validation, not just speed.

---

## 7.13 Principal-Level Interview Section

Chapter 7 is a common interview differentiator because it tests whether the candidate can bridge low-level performance and system-level impact.

### Interview prompt 1 — Slow inference endpoint

**Prompt:** “A production LLM endpoint has poor P99 TPOT. The GPU utilization dashboard looks moderate, but latency is high. How do you investigate?”

Strong answer:

1. Separate TTFT and TPOT to identify prefill vs decode pressure.
2. Use Nsight Systems or service tracing to locate timeline gaps, kernel regions, CPU dispatch, and scheduling delays.
3. Check whether decode is memory-bandwidth-bound, launch-overhead-bound, or scheduler/KV-cache-bound.
4. If a kernel dominates, use Nsight Compute to classify memory, compute, occupancy, or stall behavior.
5. Prioritize runtime batching, KV layout, attention backend, CUDA Graphs/compiler capture, and fusion before custom CUDA.
6. Verify with P50/P95/P99 TPOT under production-like concurrency.

### Interview prompt 2 — Slow training step

**Prompt:** “Training step time regressed by 12% after a model change. How do you debug?”

Strong answer:

1. Compare step breakdown: data, forward, backward, communication, optimizer, checkpoint.
2. Use timeline to see whether the regression is compute, collective, launch, or data pipeline related.
3. Check shape changes, sequence length, padding, attention kernel path, precision, and activation memory behavior.
4. If a selected kernel regressed, inspect Tensor Core path, HBM traffic, occupancy, and stall reasons.
5. Validate whether MFU changed and whether communication overlap broke.
6. Fix one hypothesis, then re-run the full training step benchmark.

### Interview prompt 3 — Should we write a custom kernel?

Strong principal answer:

> “Only if the operation is on the end-to-end critical path, existing library/compiler/fusion options are insufficient, the shape is stable enough, numerical behavior can be validated, and the maintenance cost is justified by workload-level gain. I would prototype in Triton if appropriate, compare against vendor/library baselines, and only move to lower-level CUDA if the prototype proves the opportunity.”

### Principal rubric

| Level | What the answer sounds like |
|---|---|
| Senior engineer | “I can profile the kernel and tune memory/occupancy.” |
| Staff engineer | “I can classify the bottleneck and choose the right optimization technique.” |
| Principal engineer | “I can decide whether the kernel should be optimized at all, connect the fix to product/SLO/cost impact, and prevent local optimization from harming the system.” |

---

## 7.14 Key Takeaways

1. A kernel is worth optimizing only when it affects an end-to-end metric that matters.
2. GPU execution hierarchy — grid, block, warp, thread, SM — determines scheduling, synchronization, and resource limits.
3. Memory coalescing is the first global-memory performance rule: neighboring lanes should usually access neighboring addresses.
4. Shared memory helps when it creates reuse, but bank conflicts, synchronization, and occupancy pressure can erase the benefit.
5. Occupancy is a diagnostic clue, not a goal. Enough occupancy hides latency; maximum occupancy is not always best.
6. GEMM performance is usually a library-first problem: shape, layout, precision, and Tensor Core path matter before custom kernels.
7. FlashAttention is the canonical IO-aware kernel: it preserves attention semantics while reducing HBM traffic from intermediate materialization.
8. Fusion is powerful when it removes intermediate HBM writes/reads and launch overhead, but gains are workload-specific.
9. Triton is a productivity layer for custom kernels, not a substitute for performance reasoning.
10. The correct profiling loop is `nsys → ncu → fix → verify`, with end-to-end validation.

---

## Review Questions

1. Explain the hierarchy of grid, block/CTA, warp, thread, and SM in a CUDA kernel launch.
2. Why does memory coalescing matter for HBM bandwidth utilization?
3. Give an example where shared memory improves performance and an example where it could hurt performance.
4. What is a bank conflict, and why should the exact behavior be described as architecture-dependent?
5. Why is maximum occupancy not always the best goal?
6. Derive the GEMM FLOP estimate `2 × M × N × K`.
7. Why should GEMM optimization usually start with library path, layout, precision, and shape checks?
8. Explain FlashAttention without saying it simply changes attention from `O(S²)` to `O(S)`.
9. What kind of operations benefit most from fusion?
10. When is Triton a good first custom-kernel implementation path?
11. Why should Nsight Systems usually come before Nsight Compute?
12. What metric would you use to prove a kernel optimization helped an inference service?
13. What metric would you use to prove a kernel optimization helped a training job?
14. Use Amdahl’s law to explain why a 2× speedup to a 1% kernel is usually low priority.
15. In a principal-level design review, what evidence would you require before approving custom CUDA work?

---

## Production Notes — Not Reader-Facing

### Files expected in repository

- Markdown source: `source/chapters/ch07_gpu_kernels_cuda_optimization.md`
- HTML chapter: `chapters/ch07_gpu_kernels_cuda_optimization.html`

### Figure production status

| Figure | Status | Notes |
|---|---|---|
| Fig. 7.1 | Must create/adapt | Use `diagrams_batch3.html#d27` if suitable. |
| Fig. 7.2 | Must create | Coalesced vs strided/scattered memory. |
| Fig. 7.3 | Must create | GEMM tiling pipeline. |
| Fig. 7.4 | Adapt existing | Use `diagrams_batch1.html#d5` conceptually. |
| Fig. 7.5 | Must create | Shared-memory bank conflict diagram. |
| Fig. 7.6 | Must create | Occupancy limiter diagram. |
| Fig. 7.7 | Adapt existing | Use `diagrams_batch3.html#d30` conceptually. |
| Fig. 7.8 | Must create | Custom-kernel decision tree. |

### Validation reminders

- Keep warp size wording NVIDIA-specific.
- Do not overclaim FlashAttention complexity or universal speedups.
- Keep FlashAttention-3 numbers explicitly labeled as paper/release-reported and environment-specific.
- Do not claim Triton is universally faster than PyTorch or vendor libraries.
- Do not claim occupancy should be maximized.
- Treat performance gains as workload-specific unless measured in this book’s benchmark harness.

