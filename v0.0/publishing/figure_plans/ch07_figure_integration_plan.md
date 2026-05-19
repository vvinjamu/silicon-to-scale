# Chapter 7 Figure and Table Integration Plan  
## GPU Kernels and CUDA Optimization  
### AI/ML Infrastructure from Silicon to Scale — Production v1.0

**Chapter:** Ch07 — GPU Kernels and CUDA Optimization  
**Target source slug:** `ch07_gpu_kernels_cuda_optimization`  
**Current as of:** 2026 edition  
**Planning status:** Figure/table integration plan for production source pack

---

## 1. Figure Plan Overview

Chapter 7 needs visual reinforcement because GPU-kernel performance is hard to understand from prose alone. The chapter should include **8 figures** and **7 tables**.

Design goals:

1. Make execution hierarchy visible.
2. Show memory access patterns, not just define them.
3. Explain GEMM and FlashAttention as data-movement problems.
4. Teach profiling as a workflow, not a tool list.
5. Keep diagrams print-safe and web-responsive.
6. Use existing assets where they match; create new figures when the existing diagrams are too broad or belong to another chapter.

---

## 2. Figure Integration Plan

### Figure 7.1 — GPU Kernel Execution Hierarchy

| Field | Value |
|---|---|
| Number | Fig. 7.1 |
| Title | GPU Kernel Execution Hierarchy: Grid → CTA/Block → Warp → Thread |
| Existing source file if available | `diagrams_batch3.html#d27` may contain CUDA hierarchy; otherwise must be created |
| Exists or must be created | **Must be created or adapted** |
| Exact section placement | §7.1 The GPU Kernel Execution Model |
| Caption | A CUDA kernel launches a grid of blocks. Each block runs on an SM as one or more warps of 32 threads. The hierarchy determines scheduling, memory access, synchronization, and occupancy. |
| Intro paragraph | Before optimizing a GPU kernel, you need to know what the hardware is actually scheduling. A PyTorch operation becomes one or more kernels; each kernel becomes a grid; each grid is divided into blocks; each block is executed as warps. |
| Explanation paragraph | The figure should show a top-level kernel launch feeding a grid of blocks, blocks assigned to SMs, each block decomposed into warps, and each warp containing 32 CUDA threads. Include a note that NVIDIA CUDA warp size is 32 threads; AMD wavefront concepts are similar but not identical. |
| Key takeaway | Kernel performance starts with execution mapping: enough independent work must exist to fill SMs and hide latency. |
| Web-readiness | SVG must scale to `max-width: 100%`; labels must remain legible at mobile widths. |
| Print-readiness | Use high-contrast outlines and avoid tiny thread labels; include a caption that stands alone in grayscale. |
| Required production fixes | Create a clean Ch07-specific version; avoid overloading with GPU architecture details already covered in Ch03A. |

---

### Figure 7.2 — Memory Coalescing: Good vs Bad Global Loads

| Field | Value |
|---|---|
| Number | Fig. 7.2 |
| Title | Memory Coalescing: Consecutive Threads, Consecutive Addresses |
| Existing source file if available | None identified |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.2 Memory Coalescing — The First Rule of GPU Performance |
| Caption | Coalesced memory access lets neighboring threads in a warp load neighboring addresses, minimizing memory transactions. Strided or scattered access wastes bandwidth even when arithmetic work is unchanged. |
| Intro paragraph | Many slow kernels do not perform too many FLOPs. They move the same amount of data in a pattern that prevents the memory system from serving it efficiently. |
| Explanation paragraph | The figure should show two rows: coalesced access, where thread 0 reads element 0, thread 1 reads element 1, and so on; and strided/scattered access, where each thread hits a distant cache line. Use arrows from warp lanes to memory addresses. |
| Key takeaway | Memory bandwidth is not just a number on the spec sheet; the access pattern determines how much of that bandwidth a kernel can actually use. |
| Web-readiness | Use simple horizontal layout; avoid dense grids on mobile. |
| Print-readiness | Label “coalesced” and “strided” directly rather than relying on color. |
| Required production fixes | Use representative transaction labels without overclaiming exact transaction size for all architectures. |

---

### Figure 7.3 — GEMM Tiling Pipeline

| Field | Value |
|---|---|
| Number | Fig. 7.3 |
| Title | GEMM Tiling: HBM → Shared Memory → Registers → Tensor Cores |
| Existing source file if available | May need to be created; `diagrams_batch1.html#d1` roofline can be cross-referenced |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.5 Tensor Cores, GEMM, and Library-First Optimization |
| Caption | High-performance GEMM is a data-reuse problem. Tiles are loaded from HBM into shared memory, fragments are staged into registers, Tensor Cores perform matrix multiply-accumulate, and output tiles are written back. |
| Intro paragraph | Transformer layers are dominated by matrix multiplications. The reason GEMM can approach hardware peak is not magic — it is tiled reuse of data close to the compute units. |
| Explanation paragraph | Show matrices A, B, C; tile arrows from HBM to shared memory; smaller fragments into registers/Tensor Cores; accumulation into C. Annotate reuse: each loaded tile participates in many multiply-accumulate operations. |
| Key takeaway | GEMM is fast because it turns memory traffic into repeated compute through tiling and reuse. |
| Web-readiness | Use large labels and minimal math. |
| Print-readiness | Avoid gradient-only contrast; include arrows with text labels. |
| Required production fixes | Label as conceptual; do not imply a single universal tiling strategy across all GEMM libraries/hardware. |

---

### Figure 7.4 — FlashAttention IO-Aware Attention

| Field | Value |
|---|---|
| Number | Fig. 7.4 |
| Title | FlashAttention: Compute Attention Without Materializing the S×S Score Matrix in HBM |
| Existing source file if available | `diagrams_batch1.html#d5` likely contains FlashAttention; can be adapted |
| Exists or must be created | **Exists conceptually; adapt for Ch07** |
| Exact section placement | §7.6 FlashAttention — IO-Aware Kernel Design |
| Caption | FlashAttention tiles Q, K, and V so attention can be computed through on-chip memory without writing the full S×S attention matrix to HBM. The math is exact; the memory traffic is reduced. |
| Intro paragraph | The naive attention algorithm is expensive not only because of computation, but because it materializes a large attention-score matrix. FlashAttention changes the memory schedule while preserving the attention result. |
| Explanation paragraph | Show naive path on the left: QKᵀ → S×S scores in HBM → softmax → scores×V. Show FlashAttention path on the right: Q tile streams over K/V tiles, uses online softmax, writes only the final output. |
| Key takeaway | FlashAttention is the canonical example of IO-aware algorithm design: same mathematical output, less HBM traffic. |
| Web-readiness | Existing SVG should be embedded or redrawn with Ch07 labels. |
| Print-readiness | Needs a text caption because the flow is subtle in grayscale. |
| Required production fixes | Avoid saying O(S) HBM traffic without explaining “relative to not materializing the full S×S score matrix”; use safe wording. |

---

### Figure 7.5 — Shared Memory Bank Conflicts

| Field | Value |
|---|---|
| Number | Fig. 7.5 |
| Title | Shared Memory Bank Conflicts: When Fast Memory Serializes |
| Existing source file if available | None identified |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.3 Shared Memory, Registers, and Bank Conflicts |
| Caption | Shared memory is fast when a warp accesses distinct banks or broadcast-friendly addresses. Conflicting bank access patterns can serialize requests and reduce effective bandwidth. |
| Intro paragraph | Shared memory is not automatically fast. It is fast when the access pattern cooperates with the bank organization. |
| Explanation paragraph | Show 32 banks conceptually. In the good case, lanes map to different banks. In the bad case, multiple lanes contend for the same bank, causing serialization. Include a note that exact bank width and conflict behavior vary by architecture. |
| Key takeaway | Shared memory is a tool for data reuse, but bad access patterns can turn it into a bottleneck. |
| Web-readiness | Use simple bank blocks and lane arrows; avoid showing all 32 lanes if too dense. |
| Print-readiness | Label “parallel” vs “serialized” explicitly. |
| Required production fixes | Use architecture-agnostic wording and cite CUDA documentation. |

---

### Figure 7.6 — Occupancy Limiters

| Field | Value |
|---|---|
| Number | Fig. 7.6 |
| Title | Occupancy Is Limited by Threads, Registers, Shared Memory, and Blocks |
| Existing source file if available | None identified |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.4 Warp Divergence, Occupancy, and Latency Hiding |
| Caption | Occupancy is constrained by multiple per-SM resources. More occupancy is useful only until enough independent work exists to hide latency. |
| Intro paragraph | Occupancy is one of the most overused and misunderstood GPU metrics. Low occupancy can be a problem, but high occupancy is not automatically a win. |
| Explanation paragraph | Show an SM resource budget divided into register file, shared memory, maximum resident blocks, and maximum resident warps. Demonstrate one kernel limited by registers and another limited by shared memory. |
| Key takeaway | Treat occupancy as a diagnostic clue, not a performance objective by itself. |
| Web-readiness | Should be a compact conceptual diagram. |
| Print-readiness | Use patterns or labels, not just color regions. |
| Required production fixes | Add caveat that exact limits vary by GPU architecture and kernel compilation. |

---

### Figure 7.7 — Profiling Workflow: nsys → ncu → Fix → Verify

| Field | Value |
|---|---|
| Number | Fig. 7.7 |
| Title | The Kernel Profiling Workflow |
| Existing source file if available | `diagrams_batch3.html#d30` may contain profiling tree; can be adapted |
| Exists or must be created | **Exists conceptually; adapt for Ch07** |
| Exact section placement | §7.9 Kernel Profiling Workflow |
| Caption | Start with Nsight Systems to find where time goes. Use Nsight Compute only after a specific kernel is proven important. Then fix one hypothesis and verify end-to-end impact. |
| Intro paragraph | The most common profiling mistake is opening a kernel profiler before proving that the kernel is on the critical path. |
| Explanation paragraph | Show the loop: system symptom → Nsight Systems timeline → identify critical kernel or gap → Nsight Compute counters → classify bottleneck → apply fix → re-run system benchmark. Include a red branch for “kernel not on critical path: stop.” |
| Key takeaway | Nsight Systems answers “where did time go?” Nsight Compute answers “why was this kernel slow?” |
| Web-readiness | Should fit above the fold on desktop; use a vertical layout on mobile. |
| Print-readiness | Flow arrows and branch labels must remain clear. |
| Required production fixes | Add note that profiling settings can perturb performance; measure reproducibly. |

---

### Figure 7.8 — Kernel Optimization Decision Tree

| Field | Value |
|---|---|
| Number | Fig. 7.8 |
| Title | Should You Write a Custom Kernel? |
| Existing source file if available | None identified |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.10 Principal-Level Optimization Prioritization |
| Caption | A custom kernel is justified only after library, layout, batching, fusion, compiler, and configuration options are ruled out or shown insufficient. |
| Intro paragraph | Principal engineers protect teams from expensive low-leverage optimization projects. The decision tree makes that judgment explicit. |
| Explanation paragraph | Decision nodes: Is the operation >10% of end-to-end time? Is a vendor/library kernel available? Is the bottleneck data movement or launch overhead? Can fusion remove the memory round trip? Is the shape stable enough for CUDA Graphs or compiler capture? Only then consider Triton/CUDA custom work. |
| Key takeaway | Custom kernels are powerful, but the highest-leverage optimization is often to avoid writing one. |
| Web-readiness | Decision tree can be collapsible in HTML if long. |
| Print-readiness | Use compact nodes; avoid tiny text. |
| Required production fixes | Ensure this does not duplicate Ch09; frame compiler/fusion branches as routing decisions only. |

---

## 3. Table Integration Plan

### Table 7.1 — Kernel Bottleneck Taxonomy

| Field | Value |
|---|---|
| Number | Table 7.1 |
| Title | Kernel Bottleneck Taxonomy |
| Existing source file if available | None identified |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.0 or §7.1 after chapter overview |
| Caption | Common kernel bottlenecks, symptoms, profiler signals, and first fixes. |
| Intro paragraph | Kernel performance problems usually fall into a small number of regimes. Naming the regime prevents random tuning. |
| Explanation paragraph | Columns should include bottleneck, symptom, profiler signal, likely causes, and first fix. |
| Key takeaway | Start with classification before code changes. |
| Web-readiness | Responsive table wrapper required. |
| Print-readiness | Consider splitting into two tables if too wide. |
| Required production fixes | Use representative metrics; avoid hard thresholds unless validated. |

**Draft table content:**

| Bottleneck | Typical symptom | Profiler signal | Likely cause | First fix |
|---|---|---|---|---|
| Memory bandwidth | High DRAM bytes, low compute utilization | High HBM traffic vs useful FLOPs | Poor reuse, unfused ops, strided loads | Coalesce, fuse, tile, reduce bytes |
| Compute / Tensor Core | High tensor pipe utilization | Tensor cores active, little idle | True math-heavy GEMM | Use library kernels, ensure alignment/precision |
| Launch overhead | Gaps between small kernels | Nsight Systems shows CPU gaps | Many small ops, Python dispatch | Fuse, `torch.compile`, CUDA Graphs |
| Latency / dependency | Stalls despite low bandwidth | Warp stall metrics dominate | Dependent loads, insufficient parallelism | More independent work, better tiling |
| Occupancy-limited | Few active warps/blocks | Occupancy constrained by registers/SMEM | Too many registers, too much SMEM | Tune tile sizes, reduce register pressure |
| Communication overlap | NCCL after compute | Timeline shows serial collectives | Poor stream scheduling/bucketing | Overlap, bucket, topology fix |

---

### Table 7.2 — CUDA Memory Spaces Reference

| Field | Value |
|---|---|
| Number | Table 7.2 |
| Title | CUDA Memory Spaces and What They Are Good For |
| Existing source file if available | `diagram_01_memory_hierarchy.html` can be cross-referenced |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.3 Shared Memory, Registers, and Bank Conflicts |
| Caption | Registers, shared memory, caches, and HBM differ in latency, bandwidth, capacity, and programmability. |
| Intro paragraph | Kernel optimization is mostly the art of using the right memory space for the right reuse pattern. |
| Explanation paragraph | Keep it qualitative and refer to Ch04 for exact hierarchy details. |
| Key takeaway | Faster memory is smaller and harder to use; slower memory is larger and easier to overuse. |
| Web-readiness | Responsive table wrapper. |
| Print-readiness | Short cell content. |
| Required production fixes | Avoid exact latency numbers unless sourced and architecture-specific. |

**Draft table content:**

| Memory space | Scope | Typical use | Risk |
|---|---|---|---|
| Registers | Per thread | Fragments, accumulators, loop variables | Register pressure reduces occupancy |
| Shared memory | Per block/CTA | Reused tiles, staging, reductions | Bank conflicts, limited capacity |
| L1 / texture cache | Per SM | Cached global/local accesses | Hit rate depends on access pattern |
| L2 cache | GPU-wide | Cross-SM reuse, global cache | Capacity pressure |
| HBM/global memory | GPU-wide | Tensors, model weights, activations | Bandwidth bottleneck |
| Host memory | CPU/system | Data staging, offload | PCIe/NVLink transfer latency |

---

### Table 7.3 — Nsight Compute Metric Cheat Sheet

| Field | Value |
|---|---|
| Number | Table 7.3 |
| Title | Nsight Compute Metrics: What to Look At First |
| Existing source file if available | Existing Ch12/Ch17 snippets mention profiler metrics; create Ch07-specific table |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.9 Kernel Profiling Workflow |
| Caption | Start with high-level sections, then drill into counters only when you have a hypothesis. |
| Intro paragraph | Nsight Compute can overwhelm readers with metrics. This table gives a small starting set. |
| Explanation paragraph | Use metric families and safe descriptions because exact metric names change by tool version and architecture. |
| Key takeaway | Use counters to confirm a bottleneck classification, not to hunt randomly. |
| Web-readiness | Table may be wide; wrap. |
| Print-readiness | Compact descriptions. |
| Required production fixes | Add warning that metric names vary by Nsight Compute version. |

**Draft table content:**

| Metric family | What it indicates | When to care | Typical response |
|---|---|---|---|
| Tensor/FP pipe utilization | Whether compute units are busy | GEMM/attention kernels | Check precision, alignment, library kernel |
| DRAM/HBM throughput | Whether memory bandwidth is limiting | Elementwise, norm, attention | Coalesce, fuse, tile |
| L2 hit rate / cache throughput | Reuse across accesses | Reused weights/activations | Improve locality/layout |
| Warp stall reasons | Why warps cannot issue | Any slow kernel | Map dominant stall to specific fix |
| Occupancy / launch stats | Active warps/blocks | Latency hiding problems | Tune registers/SMEM/block size |
| Memory workload analysis | Load/store pattern quality | Suspected bad coalescing | Reorder layout or access pattern |

---

### Table 7.4 — FlashAttention Version Comparison

| Field | Value |
|---|---|
| Number | Table 7.4 |
| Title | FlashAttention Version Comparison |
| Existing source file if available | `diagrams_batch1.html#d5`; external papers needed |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.7 FlashAttention-2 and FlashAttention-3 |
| Caption | FlashAttention versions improve parallelism, work partitioning, and hardware-specific utilization while preserving exact attention semantics. |
| Intro paragraph | FlashAttention is not a single optimization frozen in time; it is a family of kernels that evolved with GPU architecture. |
| Explanation paragraph | Present version-level differences without promising universal speedups. |
| Key takeaway | Use the newest validated implementation supported by your stack and hardware, but measure it in your workload. |
| Web-readiness | Small table. |
| Print-readiness | Good. |
| Required production fixes | Mark performance as reported/environment-specific. |

**Draft table content:**

| Version | Main idea | Best suited for | Safe performance wording | Confidence |
|---|---|---|---|---|
| FlashAttention | IO-aware exact attention, tiled to reduce HBM traffic | Long sequence attention | Reduces memory traffic vs naive attention | `[SHIPPED]` |
| FlashAttention-2 | Better parallelism and work partitioning | A100/H100-class GPUs | Often faster than v1; exact gain varies | `[ENV-SPECIFIC]` |
| FlashAttention-3 | Hopper-specific WGMMA/TMA/asynchrony and FP8 support | H100/Hopper | Paper reports up to ~740 TFLOP/s FP16 and 1.5–2.0× reported speedups | `[ENV-SPECIFIC]` |

---

### Table 7.5 — GEMM Tuning Levers

| Field | Value |
|---|---|
| Number | Table 7.5 |
| Title | GEMM Tuning Levers and When They Matter |
| Existing source file if available | None identified |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.5 Tensor Cores, GEMM, and Library-First Optimization |
| Caption | GEMM performance usually depends on shape, layout, precision, alignment, and library selection before custom kernel work. |
| Intro paragraph | Most transformer compute is GEMM, but most engineers should not start by writing a GEMM kernel. |
| Explanation paragraph | The table should guide readers toward library-first debugging. |
| Key takeaway | Use cuBLAS/CUTLASS/vendor libraries first; custom GEMM is rare and specialized. |
| Web-readiness | Medium-width table. |
| Print-readiness | Good. |
| Required production fixes | Ensure vendor-neutral wording where possible. |

**Draft table content:**

| Lever | Why it matters | First check |
|---|---|---|
| Matrix dimensions | Tensor Cores prefer aligned tile shapes | Are dimensions multiples of hardware-friendly tile sizes? |
| Precision | BF16/FP16/FP8 map to different Tensor Core paths | Is the intended precision actually active? |
| Layout | Strides/transposes affect memory coalescing | Are tensors contiguous or in expected layout? |
| Batch size | Small batch may underutilize compute | Can batching increase arithmetic intensity? |
| Library path | Vendor kernels are heavily optimized | Is cuBLAS/cuBLASLt/CUTLASS path active? |
| Fusion | Separate epilogues cause extra memory traffic | Can bias/activation/residual be fused? |

---

### Table 7.6 — Occupancy Limiters and Mitigations

| Field | Value |
|---|---|
| Number | Table 7.6 |
| Title | Occupancy Limiters and Mitigation Options |
| Existing source file if available | None identified |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.4 Warp Divergence, Occupancy, and Latency Hiding |
| Caption | Occupancy is constrained by per-SM resource limits; the right response depends on which resource is limiting. |
| Intro paragraph | Occupancy is useful only when interpreted with its limiting factor. |
| Explanation paragraph | Show register, shared memory, block, and thread limits with mitigations. |
| Key takeaway | Improve the limiter, not the occupancy number. |
| Web-readiness | Good. |
| Print-readiness | Good. |
| Required production fixes | Add caveat that exact occupancy limits are architecture-specific. |

**Draft table content:**

| Limiter | Symptom | Mitigation |
|---|---|---|
| Registers per thread | Low resident warps due to register pressure | Reduce unrolling, split kernel, tune tile size |
| Shared memory per block | Few blocks resident per SM | Smaller tiles, double-buffer only when beneficial |
| Threads per block | Too few or too many active warps | Tune block shape |
| Blocks per SM | CTA limit reached | Adjust block granularity |
| Long dependency chains | Warps resident but stalled | Increase independent work, prefetch, restructure |

---

### Table 7.7 — Principal-Level Kernel Prioritization Matrix

| Field | Value |
|---|---|
| Number | Table 7.7 |
| Title | Should This Kernel Be Optimized? |
| Existing source file if available | None identified |
| Exists or must be created | **Must be created** |
| Exact section placement | §7.10 Principal-Level Optimization Prioritization |
| Caption | A kernel optimization is justified only when it improves an end-to-end metric enough to beat simpler alternatives. |
| Intro paragraph | Principal engineers prioritize by business/system impact, not by technical curiosity. |
| Explanation paragraph | Compare optimization targets by end-to-end leverage and risk. |
| Key takeaway | The best kernel optimization may be a library upgrade, batching change, or fusion that removes the kernel entirely. |
| Web-readiness | Good. |
| Print-readiness | Good. |
| Required production fixes | Avoid claiming universal speedups; all gains are workload-specific. |

**Draft table content:**

| Situation | Optimize kernel now? | Better first move | Priority |
|---|---|---|---|
| Kernel is <1% of wall time | No | Ignore or batch with nearby work | P2 |
| Kernel is 10–30% and memory-bound | Maybe | Check fusion/coalescing/library path | P1/P0 |
| GEMM underperforms | Usually not custom | Verify cuBLASLt precision/layout/alignment | P0 |
| Many small kernels with CPU gaps | Not individual kernels | CUDA Graphs / compiler capture / fusion | P0 |
| FlashAttention disabled | No custom attention yet | Enable supported FlashAttention path | P0 |
| Unique op unsupported by libraries | Yes, if on critical path | Triton prototype, then CUDA/CUTLASS if needed | P1 |

---

## 4. Figure/Table Cross-Reference by Section

| Section | Figure(s) | Table(s) |
|---|---|---|
| §7.0 Chapter Overview | — | Table 7.1 |
| §7.1 Kernel Execution Model | Fig. 7.1 | — |
| §7.2 Memory Coalescing | Fig. 7.2 | — |
| §7.3 Shared Memory / Registers / Bank Conflicts | Fig. 7.5 | Table 7.2 |
| §7.4 Warp Divergence / Occupancy | Fig. 7.6 | Table 7.6 |
| §7.5 Tensor Cores / GEMM | Fig. 7.3 | Table 7.5 |
| §7.6 FlashAttention | Fig. 7.4 | — |
| §7.7 FlashAttention-2/3 | — | Table 7.4 |
| §7.8 Triton | Fig. 7.8 optional branch | — |
| §7.9 Profiling Workflow | Fig. 7.7 | Table 7.3 |
| §7.10 Principal Prioritization | Fig. 7.8 | Table 7.7 |

---

## 5. Production Notes for Figure Creation

### 5.1 Visual style

Use the existing GitHub Pages design language:

- Dark background: `#0d1117` / `#080c10`
- Surface: `#161b22`
- Border: `#30363d`
- Accent blue: `#58a6ff`
- Green: `#3fb950`
- Gold/yellow: `#f7c948`
- Purple: `#bc8cff`
- Red/orange for warnings: `#f85149`, `#ff6b35`
- Monospace labels for technical annotations

### 5.2 Accessibility

Every figure must include:

- Descriptive caption
- Text explanation before and after
- No color-only meaning
- Sufficient contrast in grayscale
- SVG `viewBox`
- Responsive width

### 5.3 Print handling

In print CSS:

```css
.figure-card, .table-wrap {
  break-inside: avoid;
}
svg {
  max-width: 100%;
  height: auto;
}
```

### 5.4 Web handling

For responsive tables:

```css
.table-wrap {
  overflow-x: auto;
  margin: 1.5rem 0;
}
table {
  min-width: 720px;
}
```

---

## 6. Production Priority Summary

### P0

- Fig. 7.1 execution hierarchy
- Fig. 7.2 coalescing
- Fig. 7.3 GEMM tiling
- Fig. 7.4 FlashAttention
- Fig. 7.7 profiling workflow
- Table 7.1 bottleneck taxonomy
- Table 7.2 memory spaces
- Table 7.3 profiler metrics
- Table 7.4 FlashAttention comparison
- Table 7.7 prioritization matrix

### P1

- Fig. 7.5 bank conflicts
- Fig. 7.6 occupancy limiters
- Fig. 7.8 custom-kernel decision tree
- Table 7.5 GEMM tuning levers
- Table 7.6 occupancy mitigations

### P2

- CUDA vs ROCm terminology table
- Triton mini example figure
- Roofline mini placement figure for LayerNorm/GEMM/FlashAttention
- Hands-on profiling exercise card
