# Chapter 7 Technical Validation Plan  
## GPU Kernels and CUDA Optimization  
### AI/ML Infrastructure from Silicon to Scale — Production v1.0

**Chapter:** Ch07 — GPU Kernels and CUDA Optimization  
**Target source slug:** `ch07_gpu_kernels_cuda_optimization`  
**Current as of:** 2026 edition  
**Validation goal:** Ensure that hardware, formula, profiler, benchmark, and architecture claims are accurate, properly scoped, and labeled with confidence labels.

---

## 1. Confidence Labels Used

| Label | Meaning in Ch07 |
|---|---|
| `[SHIPPED]` | Publicly documented shipping hardware/software behavior or stable API/tool behavior. |
| `[ANNOUNCED]` | Vendor-announced or roadmap behavior not broadly verifiable in production. |
| `[DERIVED FROM SHIPPED]` | Calculated from documented hardware specs or stable formulas. |
| `[ESTIMATED]` | Engineering estimate with shown methodology. |
| `[REPRESENTATIVE]` | Illustrative example; useful for intuition but not a universal claim. |
| `[ENV-SPECIFIC]` | Depends on workload, model, tensor shape, software version, hardware SKU, or cluster configuration. |

---

## 2. Validation Summary

| Category | Overall status | Notes |
|---|---|---|
| Hardware specs | Mostly validated | Need dense vs sparsity wording for peak FLOPS. |
| CUDA execution model | Validated | Keep NVIDIA/CUDA-specific wording precise. |
| Memory coalescing / shared memory | Validated conceptually | Avoid exact transaction/bank behavior unless sourced for a specific architecture. |
| Occupancy / warp divergence | Validated conceptually | Avoid “higher occupancy is always better.” |
| GEMM / Tensor Cores | Validated conceptually | Use library-first recommendation; avoid custom GEMM over-scope. |
| FlashAttention | Validated with papers | Report paper claims as reported results, not universal. |
| Triton | Validated conceptually | Treat backend maturity/performance as environment-specific. |
| Nsight Systems / Nsight Compute | Validated conceptually | Metric names can evolve; use metric families and representative names. |
| CUDA Graphs | Validated conceptually | Gains are workload-specific; requires graph-capture-compatible execution. |

---

## 3. Claim Validation Matrix

| # | Claim | Current value/formula | Validation status | Corrected value / safe wording | Confidence label | Source type needed | Recommended final wording | Priority |
|---:|---|---|---|---|---|---|---|---|
| 1 | CUDA kernel execution hierarchy is grid → block/CTA → warp → thread | Grid, block, warp, thread | Valid | NVIDIA CUDA warp is 32 threads; blocks are scheduled on SMs; exact scheduling details are architecture/toolchain dependent | `[SHIPPED]` | NVIDIA CUDA Programming Guide | “In CUDA, a kernel launch creates a grid of thread blocks. Blocks execute on SMs as warps of 32 threads. This hierarchy is the mental model for scheduling, synchronization, and memory access.” | P0 |
| 2 | Warp size is 32 | 32 threads | Valid for NVIDIA CUDA | Scope explicitly to NVIDIA CUDA; do not generalize to all GPUs | `[SHIPPED]` | CUDA Programming Guide | “For NVIDIA CUDA, a warp contains 32 threads. Other vendors use similar SIMD/SIMT group concepts with different naming and sometimes different width.” | P0 |
| 3 | Memory coalescing is critical for global memory bandwidth | Coalesced access improves transaction efficiency | Valid | Use qualitative wording; exact transaction behavior varies | `[SHIPPED]` | CUDA Best Practices Guide | “Coalesced access lets neighboring lanes access neighboring addresses, improving effective global-memory bandwidth.” | P0 |
| 4 | Shared memory is low latency and programmer-managed per block | Shared memory per CTA/block | Valid | Avoid exact latency numbers; mention limited capacity and bank behavior | `[SHIPPED]` | CUDA Programming Guide | “Shared memory is a low-latency, per-block scratchpad used to stage reused data. It is fast only when the access pattern avoids avoidable bank conflicts.” | P0 |
| 5 | Shared memory bank conflicts serialize accesses | Bank conflicts reduce effective bandwidth | Valid conceptually | Architecture-specific details vary; use conceptual explanation | `[SHIPPED]` | CUDA Programming Guide / NVIDIA developer blog | “If multiple lanes in a warp contend for the same shared-memory bank in a non-broadcast-friendly pattern, the access can serialize and reduce effective bandwidth.” | P1 |
| 6 | Occupancy should be maximized | Higher occupancy is better | Needs correction | Occupancy helps hide latency but is not the end goal | `[SHIPPED]` / `[ENV-SPECIFIC]` | CUDA Best Practices / Nsight Compute docs | “Occupancy is a diagnostic metric: too little can fail to hide latency, but maximum occupancy is not always optimal.” | P0 |
| 7 | Warp divergence hurts performance | Divergence serializes paths within a warp | Valid | Scope to branch divergence inside a warp; predication/control flow details vary | `[SHIPPED]` | CUDA Programming Guide | “When lanes in a warp take different control-flow paths, execution can serialize across those paths, reducing effective throughput.” | P1 |
| 8 | H100 SXM HBM bandwidth is 3.35 TB/s | 3.35 TB/s | Valid for H100 SXM/HBM3 | State SKU and peak bandwidth | `[SHIPPED]` | NVIDIA H100 product/architecture docs | “H100 SXM5 is documented with 80 GB HBM3 and up to 3.35 TB/s peak HBM bandwidth.” | P0 |
| 9 | H100 BF16/FP16 Tensor Core peak is 989 TFLOP/s | 989 TFLOPS BF16 | Needs precision | Dense vs sparsity and FP16/BF16/TF32 values are often mixed | `[SHIPPED]` | NVIDIA H100 datasheet / architecture docs | “For H100 SXM, dense FP16/BF16 Tensor Core peak is commonly listed around ~989 TFLOP/s; sparsity-enabled figures can be roughly 2×. Always state whether sparsity is included.” | P0 |
| 10 | H100 NVLink bandwidth is 900 GB/s | 900 GB/s | Valid for H100 SXM-class configuration | State bidirectional aggregate per GPU if used | `[SHIPPED]` | NVIDIA H100 specs | “H100 SXM-class systems advertise up to 900 GB/s aggregate NVLink bandwidth per GPU; exact topology depends on platform.” | P1 |
| 11 | Arithmetic intensity formula | AI = FLOPs / bytes moved | Valid | Keep as foundational formula | `[DERIVED FROM SHIPPED]` | Roofline paper / Ch01 | “Arithmetic intensity is FLOPs divided by bytes moved to/from the relevant memory level.” | P0 |
| 12 | Roofline bound formula | Perf ≤ min(peak FLOPS, bandwidth × AI) | Valid | Clarify peak and bandwidth must match precision/memory level | `[DERIVED FROM SHIPPED]` | Roofline paper | “The roofline bound is `attainable_perf ≤ min(peak_compute, memory_bandwidth × arithmetic_intensity)`.” | P0 |
| 13 | GEMM FLOPs formula | 2MNK | Valid | Include dimensions and note multiply-add convention | `[DERIVED FROM SHIPPED]` | Linear algebra / standard HPC | “A dense matrix multiply C[M,N] = A[M,K] × B[K,N] performs approximately 2MNK FLOPs under the multiply-add counting convention.” | P0 |
| 14 | GEMM is usually Tensor Core dominated in transformer models | GEMM dominates transformer compute | Valid generally | Use “often” and “for transformer dense layers” | `[DERIVED FROM SHIPPED]` / `[ENV-SPECIFIC]` | Transformer arithmetic + profiler evidence | “In transformer blocks, QKV projections, output projections, and MLP layers are dense GEMMs and usually dominate total FLOPs.” | P0 |
| 15 | Most engineers should use cuBLAS/cuBLASLt/CUTLASS before writing GEMM kernels | Library-first | Valid best practice | Keep as recommendation, not law | `[SHIPPED]` / `[ENV-SPECIFIC]` | NVIDIA library docs / CUTLASS docs | “For GEMM, start with vendor libraries. Custom GEMM is justified only for unsupported shapes, fused epilogues, or specialized research/library work.” | P0 |
| 16 | Tensor Cores require alignment/friendly shapes | Tensor Core performance depends on shape/layout/precision | Valid | Avoid exact multiples unless tied to architecture/API | `[ENV-SPECIFIC]` | CUDA/CUTLASS/cuBLAS docs | “Tensor Core throughput depends on precision, layout, alignment, and problem shape. Poor shapes can leave peak hardware unused.” | P1 |
| 17 | FlashAttention computes exact attention with lower HBM traffic | Avoids materializing S×S attention matrix | Valid | Explain exactness and online softmax | `[SHIPPED]` | FlashAttention paper | “FlashAttention preserves exact attention semantics while changing the memory schedule so the full S×S attention matrix is not materialized in HBM.” | P0 |
| 18 | Naive attention materializes S×S scores | S × S attention score matrix | Valid as naive baseline | Clarify optimized frameworks may already avoid this | `[REPRESENTATIVE]` | FlashAttention paper | “The naive formulation materializes the S×S score matrix; modern optimized kernels avoid or reduce this materialization.” | P0 |
| 19 | FlashAttention reduces attention memory complexity | O(S²) → lower IO | Needs careful wording | Do not state universal O(S) without context | `[DERIVED FROM SHIPPED]` / `[ENV-SPECIFIC]` | FlashAttention paper | “For long sequences, FlashAttention reduces HBM reads/writes by tiling and avoiding S×S score materialization; exact IO depends on implementation and shape.” | P0 |
| 20 | FlashAttention-2 improves work partitioning | Better parallelism/work partitioning | Valid | Use qualitative wording | `[SHIPPED]` | FlashAttention-2 paper | “FlashAttention-2 improves parallelism and work partitioning over the original algorithm, with measured gains depending on shape and hardware.” | P1 |
| 21 | FlashAttention-3 uses Hopper WGMMA/TMA/asynchrony | WGMMA, TMA, FP8, warp specialization | Valid | Tie to Hopper/H100; not universal to all GPUs | `[SHIPPED]` / `[ENV-SPECIFIC]` | FlashAttention-3 paper / PyTorch blog | “FlashAttention-3 targets Hopper by using asynchronous Tensor Core and TMA features, warp-specialization, and FP8 support where applicable.” | P0 |
| 22 | FlashAttention-3 reaches ~740 TFLOP/s FP16 on H100 | Up to ~740 TFLOP/s | Valid as paper-reported | Say “paper reports,” not “always achieves” | `[ENV-SPECIFIC]` | FlashAttention-3 paper | “The FlashAttention-3 paper reports FP16 performance up to roughly 740 TFLOP/s on H100 for reported configurations.” | P0 |
| 23 | FlashAttention-3 achieves 1.5–2.0× speedup on H100 | 1.5–2.0× | Valid as reported | Use reported environments | `[ENV-SPECIFIC]` | FlashAttention-3 paper | “In reported H100 experiments, FlashAttention-3 shows 1.5–2.0× speedups for selected cases; production gains must be measured per workload.” | P0 |
| 24 | “FAv3 is 34% faster than FAv2” | 34% | Needs correction | Too specific unless tied to a benchmark case | `[ENV-SPECIFIC]` | Internal benchmark or paper table | “FlashAttention-3 can outperform earlier versions on Hopper-class GPUs; the exact delta depends on sequence length, head dimension, precision, and implementation.” | P0 |
| 25 | CUDA Graphs can reduce launch overhead | 15–25% TPOT | Needs correction | Do not give universal percent | `[ENV-SPECIFIC]` | NVIDIA CUDA Graphs docs / framework docs | “CUDA Graphs can reduce CPU launch overhead for static or graph-capturable execution paths; benefit depends on kernel count, shape stability, and framework integration.” | P1 |
| 26 | Nsight Systems should precede Nsight Compute | nsys → ncu | Valid practice | Strong recommendation, not absolute law | `[SHIPPED]` / `[ENV-SPECIFIC]` | NVIDIA Nsight docs | “Use Nsight Systems first to identify where time goes. Use Nsight Compute after a specific kernel is proven important.” | P0 |
| 27 | Nsight Compute can slow execution substantially | 100–1000× | Needs safe wording | Avoid exact slowdown without source | `[ENV-SPECIFIC]` | Nsight Compute docs | “Nsight Compute can perturb and slow execution because it collects detailed kernel counters; profile focused kernels rather than entire long workloads.” | P1 |
| 28 | `sm__pipe_tensor_cycles_active` indicates Tensor Core utilization | Metric name | Version-dependent | Use representative metric family | `[ENV-SPECIFIC]` | Nsight Compute docs | “Tensor-pipe utilization metrics indicate whether Tensor Core pipelines are active; exact metric names vary by architecture and Nsight Compute version.” | P0 |
| 29 | `dram__bytes_read` / DRAM throughput metrics indicate HBM traffic | Metric name | Version-dependent | Use metric family | `[ENV-SPECIFIC]` | Nsight Compute docs | “DRAM/HBM throughput and byte counters show how much global-memory traffic a kernel generates relative to the theoretical minimum.” | P0 |
| 30 | Warp stall metrics identify why warps cannot issue | Stall metrics | Valid but nuanced | Use as clues, not final answers | `[ENV-SPECIFIC]` | Nsight Compute docs | “Warp stall reason metrics help classify bottlenecks, but they must be interpreted with the source code, memory pattern, and roofline model.” | P1 |
| 31 | Triton is Python-based GPU kernel programming | Python-based language/compiler | Valid | Mention compiler and backend dependence | `[SHIPPED]` | Triton docs | “Triton is a Python-based language and compiler for writing custom GPU kernels, especially tiled tensor programs.” | P0 |
| 32 | Triton removes need to understand GPU architecture | Not safe | Correct | Triton simplifies syntax, not performance reasoning | `[SHIPPED]` / `[ENV-SPECIFIC]` | Triton docs | “Triton reduces boilerplate but does not remove the need to understand memory coalescing, tiling, occupancy, and data movement.” | P0 |
| 33 | Custom Triton kernels are usually faster than PyTorch | Not safe | Correct | Depends on operation and compiler/library path | `[ENV-SPECIFIC]` | Benchmark required | “A Triton kernel can beat an unfused eager PyTorch sequence for some memory-bound patterns, but vendor libraries may outperform hand-written kernels for GEMM and convolution.” | P0 |
| 34 | Fused kernels save memory traffic | One load/store instead of multiple | Valid conceptually | Quantify only with example | `[DERIVED FROM SHIPPED]` | Roofline/memory traffic derivation | “Fusion helps when it removes intermediate reads/writes to HBM. The gain is largest for memory-bound chains of small operations.” | P0 |
| 35 | LayerNorm is memory-bound | Often memory-bound | Valid generally | Use “typically” and qualify | `[ENV-SPECIFIC]` | Profiling evidence / roofline | “LayerNorm and RMSNorm are typically memory-bandwidth-sensitive because they perform modest arithmetic per byte moved.” | P1 |
| 36 | Optimizing a <1% kernel rarely matters | Amdahl’s law | Valid | Use Amdahl framing | `[DERIVED FROM SHIPPED]` | Amdahl’s Law | “A 2× speedup to a kernel that is 1% of wall time improves total time by at most ~0.5%; prioritize larger bottlenecks first.” | P0 |
| 37 | Kernel launch overhead matters for many small ops | CPU gaps between kernels | Valid | Depends on runtime and shape stability | `[ENV-SPECIFIC]` | Nsight Systems / CUDA Graph docs | “If the timeline shows many short kernels separated by CPU-side gaps, launch overhead or Python/runtime dispatch may dominate.” | P0 |
| 38 | Memory coalescing should be checked before advanced tricks | First rule | Valid as advice | Keep as heuristic | `[SHIPPED]` | CUDA Best Practices Guide | “Before tuning shared memory or occupancy, verify global-memory access is reasonably coalesced.” | P0 |
| 39 | Shared memory tiling improves reuse | Tile reuse | Valid | Not always worth it for cache-friendly/simple ops | `[ENV-SPECIFIC]` | CUDA guide / CUTLASS docs | “Shared-memory tiling improves performance when reused data justifies the extra staging and synchronization cost.” | P1 |
| 40 | Bank-conflict fixes include padding/transposition | Padding can reduce conflicts | Valid generally | Use as example, not universal | `[ENV-SPECIFIC]` | CUDA guide / examples | “Padding or changing layout can reduce bank conflicts for some shared-memory access patterns.” | P2 |
| 41 | Register pressure reduces occupancy | More registers per thread can reduce resident warps | Valid | Architecture-specific limits | `[SHIPPED]` | CUDA occupancy docs | “High register use per thread can reduce the number of active warps resident on an SM.” | P1 |
| 42 | More shared memory per block reduces occupancy | SMEM per block limits resident blocks | Valid | Architecture-specific limits | `[SHIPPED]` | CUDA occupancy docs | “A block that uses more shared memory can reduce how many blocks fit concurrently on an SM.” | P1 |
| 43 | Tensor Core path may fail due to layout/precision/shape | Underperformance due to shape/layout | Valid | Avoid exact rules unless API-specific | `[ENV-SPECIFIC]` | cuBLAS/cuBLASLt/CUTLASS docs | “If a GEMM underperforms, verify it is using the intended Tensor Core path, precision, layout, and alignment.” | P0 |
| 44 | NCU roofline can classify compute/memory bound kernels | Roofline analysis in profiler | Valid | Treat profiler roofline as one tool | `[SHIPPED]` | Nsight Compute docs | “Nsight Compute’s roofline analysis can help classify whether a kernel is closer to compute or memory limits.” | P1 |
| 45 | Use `ncu` only after identifying kernel with `nsys` | Recommended workflow | Valid | Use as best practice | `[ENV-SPECIFIC]` | Nsight docs | “Do not start with full-counter kernel profiling; first identify the critical kernel or gap at the timeline level.” | P0 |
| 46 | CUDA Graphs require static-ish graph/shape behavior | Graph capture restrictions | Valid | Framework support may hide complexity | `[SHIPPED]` / `[ENV-SPECIFIC]` | CUDA docs / PyTorch docs | “CUDA Graphs work best when shapes, memory addresses, and control flow are stable enough for capture/replay.” | P1 |
| 47 | CUTLASS is a reference for tiled GEMM design | CUTLASS library | Valid | Mention as advanced reference | `[SHIPPED]` | NVIDIA CUTLASS docs/GitHub | “CUTLASS is a useful reference for understanding how high-performance GEMM tiling maps to CUDA.” | P2 |
| 48 | `torch.compile` belongs in Ch09, not deep Ch07 | Ch07 only references it | Editorial validation | Keep boundary clear | `[REPRESENTATIVE]` | Book structure | “Ch07 mentions compiler/fusion as routing options; Ch09 explains compiler internals.” | P0 |
| 49 | AMD/ROCm analogs exist but metric names differ | HIP/ROCm profilers | Valid | Add caveat sidebar | `[SHIPPED]` / `[ENV-SPECIFIC]` | ROCm docs if included | “The kernel-performance mental model applies beyond CUDA, but profiler names, wavefront semantics, and library behavior differ across vendors.” | P1 |
| 50 | A principal engineer optimizes end-to-end metrics, not microbenchmarks alone | Architecture principle | Valid | Tie to chapter review/interview | `[REPRESENTATIVE]` | Internal book principle | “A microbenchmark win counts only after the end-to-end workload improves under production-like conditions.” | P0 |

---

## 4. Formulas to Include and Validate

### 4.1 Arithmetic intensity

```text
Arithmetic Intensity = FLOPs / Bytes Moved
```

| Validation | Label | Notes |
|---|---|---|
| Valid | `[DERIVED FROM SHIPPED]` | Specify memory level when possible: HBM, L2, or shared memory. |

Recommended wording:

> Arithmetic intensity is the ratio between useful arithmetic work and bytes moved at the relevant memory level. The same kernel can have different intensity depending on whether you count HBM traffic, L2 traffic, or shared-memory traffic.

---

### 4.2 Roofline bound

```text
Attainable Performance ≤ min(Peak Compute, Memory Bandwidth × Arithmetic Intensity)
```

| Validation | Label | Notes |
|---|---|---|
| Valid | `[DERIVED FROM SHIPPED]` | Use as a classifier, not a perfect predictor. |

Recommended wording:

> The roofline model tells you what regime you are in before you tune code. If `BW × AI` is below peak compute, memory traffic is the likely limit; if it exceeds peak compute, compute throughput is the likely limit.

---

### 4.3 GEMM FLOPs

```text
FLOPs_GEMM ≈ 2 × M × N × K
```

| Validation | Label | Notes |
|---|---|---|
| Valid | `[DERIVED FROM SHIPPED]` | Under multiply-add counting convention. |

Recommended wording:

> For `C[M,N] = A[M,K] × B[K,N]`, GEMM performs approximately `2MNK` FLOPs under the standard multiply-add convention.

---

### 4.4 Amdahl prioritization

```text
Max total speedup from optimizing component p by speedup s:
Speedup_total = 1 / ((1 - p) + p / s)
```

| Validation | Label | Notes |
|---|---|---|
| Valid | `[DERIVED FROM SHIPPED]` | Useful for “should we optimize this kernel?” decisions. |

Recommended wording:

> If a kernel is only 1% of wall time, even a perfect 2× speedup improves total wall time by about 0.5%. That is why profiling order matters.

---

### 4.5 Memory traffic saved by fusion

```text
Unfused chain traffic ≈ read input + write intermediate + read intermediate + write output
Fused chain traffic ≈ read input + write output
```

| Validation | Label | Notes |
|---|---|---|
| Valid as model | `[REPRESENTATIVE]` | Actual traffic depends on cache, compiler, and operation shape. |

Recommended wording:

> Fusion helps most when it eliminates intermediate HBM writes and reads. The formula is a traffic model, not a universal benchmark result.

---

## 5. Claims to Avoid or Rewrite

| Risky wording | Why risky | Safe replacement |
|---|---|---|
| “FlashAttention-3 is 34% faster than FlashAttention-2.” | Benchmark-specific; may not generalize. | “FlashAttention-3 can outperform earlier versions on Hopper-class GPUs; measure the delta for your sequence length, precision, and implementation.” |
| “CUDA Graphs reduce TPOT by 15–25%.” | Workload-specific. | “CUDA Graphs can reduce CPU launch overhead for graph-capturable paths; observed TPOT improvement is workload-dependent.” |
| “Occupancy should be maximized.” | False in many optimized kernels. | “Enough occupancy is needed to hide latency; maximum occupancy is not always optimal.” |
| “Triton is always faster than PyTorch.” | False for vendor-library-backed operations. | “Triton can help for custom or fused patterns not already optimized by vendor libraries.” |
| “Shared memory is always faster.” | Bank conflicts, staging, and synchronization can hurt. | “Shared memory is fast when reuse is high and access patterns are bank-friendly.” |
| “GEMM is always the bottleneck.” | Depends on workload; decode can be memory-bound, scheduling/launch can dominate. | “GEMM often dominates transformer FLOPs, but profiling determines the actual bottleneck.” |
| “Nsight Compute tells you what to optimize.” | It only explains a kernel, not system criticality. | “Nsight Systems identifies critical path; Nsight Compute diagnoses selected kernels.” |
| “FlashAttention changes attention complexity from O(S²) to O(S).” | Compute complexity still has attention-dependent terms; statement can be misleading. | “FlashAttention reduces HBM IO by avoiding materialization of the full S×S score matrix while preserving exact attention.” |

---

## 6. Source Types Required

| Claim category | Required source type |
|---|---|
| CUDA execution hierarchy, warp size, coalescing, shared memory | NVIDIA CUDA Programming Guide / CUDA Best Practices Guide |
| H100 specs | NVIDIA H100 official product page, architecture whitepaper, or official datasheet |
| Tensor Core, WGMMA, TMA | NVIDIA Hopper architecture docs, FlashAttention-3 paper |
| FlashAttention algorithms and reported performance | FlashAttention, FlashAttention-2, FlashAttention-3 papers |
| Triton language/compiler | Triton official documentation |
| Nsight Systems / Nsight Compute workflow | NVIDIA Nsight Systems and Nsight Compute documentation |
| CUTLASS / GEMM tiling | NVIDIA CUTLASS documentation/GitHub |
| Amdahl / roofline formulas | Standard performance modeling sources; book Ch01 can cross-reference |
| Performance gains | Internal benchmark or published benchmark; otherwise mark `[ENV-SPECIFIC]` |

---

## 7. Production Validation Checklist

### P0 validation before source pack

- [ ] H100 BF16/FP16 peak wording separates dense vs sparsity.
- [ ] H100 HBM bandwidth uses SKU-specific wording.
- [ ] FlashAttention-3 performance claims are tied to paper-reported results.
- [ ] CUDA execution hierarchy uses NVIDIA-specific warp wording.
- [ ] Occupancy explanation avoids “maximize occupancy.”
- [ ] `nsys → ncu` workflow is clearly explained.
- [ ] Nsight Compute metric names are described as representative/version-dependent.
- [ ] Triton is presented as useful, not magically faster.
- [ ] Custom kernel recommendation is library-first and end-to-end-driven.
- [ ] All quantitative or variable claims have confidence labels.

### P1 validation before HTML publication

- [ ] Figure captions are accurate and not over-specific.
- [ ] All tables use safe wording for architecture-dependent behavior.
- [ ] Print version contains source-safe captions and avoids unsupported visuals.
- [ ] CUDA/ROCm caveat included.
- [ ] Further reading section includes current docs and papers.

### P2 validation before final print edition

- [ ] Add optional ROCm profiler mapping if AMD-specific expansion is desired.
- [ ] Add mini exercises with verified commands.
- [ ] Include a profiler-output interpretation walkthrough from a representative workload.
- [ ] Add a glossary entry for CTA, warp, occupancy, coalescing, bank conflict, Tensor Core, TMA, WGMMA, and roofline.

---

## 8. Recommended Final Wording Blocks

### 8.1 Chapter opening safe wording

> This chapter is not a complete CUDA programming course. It is a production-performance chapter. The goal is to give you enough kernel-level understanding to read profiles, reason about memory traffic, understand why Tensor Cores matter, explain FlashAttention, and decide whether a custom CUDA or Triton kernel is the right optimization lever.

### 8.2 FlashAttention safe wording

> FlashAttention is an IO-aware exact attention algorithm. It preserves the mathematical result of attention while changing the execution schedule so that the full attention-score matrix does not need to be written to HBM. This is why it is a kernel-design landmark: it improves the memory movement, not the model definition.

### 8.3 Profiling safe wording

> Use Nsight Systems first. It tells you where time is going across CPU, GPU kernels, memory copies, and collectives. Use Nsight Compute only after a specific kernel is known to matter. Nsight Compute explains why that kernel is slow; it does not tell you whether that kernel is important to the end-to-end system.

### 8.4 Occupancy safe wording

> Occupancy is the number of active warps resident on an SM relative to the maximum. It helps hide latency, but it is not a performance goal by itself. A kernel with moderate occupancy and high data reuse can outperform a high-occupancy kernel that wastes memory bandwidth.

### 8.5 Principal-level safe wording

> A principal engineer does not ask, “Can I make this kernel faster?” first. The first question is, “If this kernel became infinitely fast, how much would the workload improve?” That Amdahl-style question prevents weeks of beautiful but irrelevant optimization work.

---

## 9. Validation Decision

**Proceed to Ch07 production source pack after P0 validation items are applied.**

This chapter can be technically strong and interview-useful if it avoids three traps:

1. Becoming a CUDA tutorial.
2. Stating environment-specific speedups as universal facts.
3. Optimizing kernels without tying them to end-to-end workload impact.

The production source should emphasize measurement-first reasoning, library-first optimization, and principal-level prioritization.
