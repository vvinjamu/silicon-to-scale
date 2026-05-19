# Chapter 9 Figure Integration Plan — Operator Fusion and the Compiler Stack

**Book:** AI/ML Infrastructure from Silicon to Scale  
**Chapter:** Ch09 — Operator Fusion and the Compiler Stack  
**Workflow Stage:** Step 1 — Figure/Table Planning  
**Primary source of truth:** `ch09_fusion_compiler.pdf`  
**Formatting references:** Approved Ch04/Ch05/Ch06/Ch07 Markdown + HTML chapter format  
**Current as of:** 2026 edition

---

## Figure Plan Summary

Chapter 9 needs figures that make compiler optimization visible. The reader should see that compilers improve performance by reducing unnecessary HBM movement, reducing launch overhead, selecting kernels, controlling layouts, and choosing save/recompute boundaries.

| Figure | Title | Existing Source | Status |
|---|---|---|---|
| Fig 9.1 | Three Performance Regimes Before Compilation | `diagrams_batch3.html#d30` can be adapted | Must create / adapt |
| Fig 9.2 | Operator Fusion: HBM Traffic Waterfall | `diagrams_batch2.html#d21` | Existing source available |
| Fig 9.3 | PyTorch 2.x Compiler Stack | `diagram_04_compiler_stack.html` and `diagrams_batch3.html#d28` | Existing source available |
| Fig 9.4 | Graph Breaks and Guard-Based Recompilation | None | Must be created |
| Fig 9.5 | TensorRT/TRT-LLM Offline Compilation Pipeline | None | Must be created |
| Fig 9.6 | Graph-Level Optimization Passes | None | Must be created |
| Fig 9.7 | Min-Cut Recomputation Save vs Recompute | None | Must be created |
| Fig 9.8 | CUDA Graph Capture and Replay | None | Must be created |
| Fig 9.9 | Compiler Selection Decision Tree | None | Must be created |

---

## Fig 9.1 — Three Performance Regimes Before Compilation

**Number:** Fig 9.1  
**Title:** Three Performance Regimes Before Compilation  
**Existing source file if available:** `../diagrams/diagrams_batch3.html#d30` may be adapted from the profiling decision tree  
**Exists or must be created:** Must be created or adapted  
**Exact section placement:** After Section `9.1 The Three Performance Regimes — Diagnosing Before Compiling`

**Caption:**  
**Fig 9.1 — Three Performance Regimes Before Compilation.** Before applying compiler optimizations, classify the workload as memory-bandwidth-bound, compute-bound, or overhead-bound. Fusion, kernel selection, and CUDA Graphs solve different problems.

**Intro paragraph:**  
Compiler optimization should not start with a tool name. It should start with a bottleneck classification. A memory-bound workload needs fewer HBM round-trips. A compute-bound workload needs better kernels, tile choices, or precision. An overhead-bound workload needs fewer launches and fewer graph breaks.

**Explanation paragraph:**  
The figure should show a decision flow from profiler symptoms to compiler action: `nsys` gaps indicate launch overhead; `ncu` roofline and HBM counters indicate memory-bound kernels; high Tensor Core activity indicates compute-bound execution. Each regime maps to a different optimization family.

**Key takeaway:**  
Use compilers to fix the regime you actually measured.

**Web-readiness:**  
Use large labels and minimal text. It should fit inside the standard figure placeholder until rendered.

**Print-readiness:**  
Must be readable in grayscale; do not rely only on color to distinguish regimes.

**Required production fixes:**  
Create SVG and 300-DPI PNG. Ensure labels use `memory-bound`, `compute-bound`, and `overhead-bound` consistently.

---

## Fig 9.2 — Operator Fusion: HBM Traffic Waterfall

**Number:** Fig 9.2  
**Title:** Operator Fusion: HBM Traffic Waterfall — Unfused vs Fused Kernel  
**Existing source file if available:** `../diagrams/diagrams_batch2.html#d21`  
**Exists or must be created:** Existing source available; print export needed  
**Exact section placement:** Inside Section `9.2 Operator Fusion — Types, Mechanics, and HBM Savings`, after fusion types

**Caption:**  
**Fig 9.2 — Operator Fusion: HBM Traffic Waterfall.** Fusion improves performance by keeping intermediate values in registers or shared memory instead of writing them to HBM and reading them back in separate kernels.

**Intro paragraph:**  
The most visible compiler win is often not more math. It is fewer bytes moved. Fusion removes intermediate HBM writes and reads between operations that can run together.

**Explanation paragraph:**  
The diagram should show an unfused chain with repeated HBM writes and reads, then a fused chain that reads the input once and writes the final output once. This directly supports the PDF's vertical fusion, horizontal fusion, and kernel replacement discussion.

**Key takeaway:**  
Fusion is memory-traffic reduction.

**Web-readiness:**  
Existing HTML diagram is usable as a linked source. Add placeholder with source link.

**Print-readiness:**  
Export to PNG/SVG; ensure arrows and byte labels remain readable.

**Required production fixes:**  
Export `ch09_fig_9_2_operator_fusion_waterfall.svg` and `png_300dpi/ch09_fig_9_2_operator_fusion_waterfall.png`.

---

## Fig 9.3 — PyTorch 2.x Compiler Stack

**Number:** Fig 9.3  
**Title:** PyTorch 2.x Compiler Stack — TorchDynamo → AOTAutograd → TorchInductor → Triton  
**Existing source file if available:** `../diagrams/diagram_04_compiler_stack.html`; also `../diagrams/diagrams_batch3.html#d28`  
**Exists or must be created:** Existing source available; use one as primary  
**Exact section placement:** Section `9.3 The PyTorch 2.x Compiler Stack`, immediately after the four-layer explanation

**Caption:**  
**Fig 9.3 — PyTorch 2.x Compiler Stack.** `torch.compile` captures Python execution into graphs, traces forward/backward where applicable, lowers optimized graphs to backend code, and emits GPU kernels through backends such as Triton.

**Intro paragraph:**  
`torch.compile` is not a single magic switch. It is a stack: Dynamo captures graphs, AOTAutograd enables forward/backward optimization, Inductor lowers graphs, and Triton or other backends generate kernels.

**Explanation paragraph:**  
The figure should preserve the PDF's four-layer architecture while avoiding over-specific implementation promises. It should teach what each layer is responsible for: capture, autograd graphing, graph lowering, and GPU code generation.

**Key takeaway:**  
Understand which compiler layer failed before blaming `torch.compile` as a whole.

**Web-readiness:**  
Use the existing compiler-stack HTML as a source link.

**Print-readiness:**  
Export the diagram; remove tiny annotations if they do not print clearly.

**Required production fixes:**  
Use `diagram_04_compiler_stack.html` or `diagrams_batch3.html#d28` as source; export stable image assets.

---

## Fig 9.4 — Graph Breaks and Guard-Based Recompilation

**Number:** Fig 9.4  
**Title:** Graph Breaks and Guard-Based Recompilation  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** Section `9.3.2 torch.compile in Practice`, after graph breaks / dynamic shapes explanation

**Caption:**  
**Fig 9.4 — Graph Breaks and Guard-Based Recompilation.** TorchDynamo captures eligible tensor operations into graphs, but unsupported Python control flow or guard failures can split graphs or trigger recompilation.

**Intro paragraph:**  
A common reason `torch.compile` underperforms is not the generated kernel. It is graph capture failure. Python-side logic can split the model into small compiled islands separated by eager execution.

**Explanation paragraph:**  
The figure should show a model forward pass split into compiled graph segments and eager gaps. A second branch should show shape guards: known shape uses cached graph; new shape triggers new compile or fallback.

**Key takeaway:**  
Graph breaks turn a compiled model back into many smaller eager segments.

**Web-readiness:**  
Keep labels simple: `captured graph`, `graph break`, `eager fallback`, `guard miss`, `recompile`.

**Print-readiness:**  
Use clear lane diagrams and visible break markers.

**Required production fixes:**  
Create SVG and PNG. Validate terminology with PyTorch docs.

---

## Fig 9.5 — TensorRT/TRT-LLM Offline Compilation Pipeline

**Number:** Fig 9.5  
**Title:** TensorRT/TRT-LLM Offline Compilation Pipeline  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** Section `9.5 TensorRT and TRT-LLM — Offline Production Compilation`

**Caption:**  
**Fig 9.5 — TensorRT/TRT-LLM Offline Compilation Pipeline.** Stable production models can amortize offline compilation that performs graph cleanup, layer fusion, precision assignment, kernel selection, memory planning, and runtime-profile generation.

**Intro paragraph:**  
TensorRT and TensorRT-LLM should be introduced as production-oriented compilation and runtime stacks. They trade longer build time for faster, more predictable serving execution.

**Explanation paragraph:**  
The figure should show an input model/checkpoint flowing through graph cleanup, fusion, precision assignment, kernel selection, memory planning, engine build, and runtime serving. Add a note that output artifacts are hardware/runtime-specific.

**Key takeaway:**  
Offline compilation is worth it when build time is amortized across production traffic.

**Web-readiness:**  
Use simple stage cards rather than dense text.

**Print-readiness:**  
Use a horizontal or vertical pipeline with high-contrast labels.

**Required production fixes:**  
Create SVG/PNG and align with current TensorRT/TRT-LLM official docs.

---

## Fig 9.6 — Graph-Level Optimization Passes

**Number:** Fig 9.6  
**Title:** Graph-Level Optimization Passes  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** Section `9.7 Graph-Level Optimizations in Depth`

**Caption:**  
**Fig 9.6 — Graph-Level Optimization Passes.** Compilers simplify the graph before generating kernels through dead code elimination, constant folding, common subexpression elimination, layout propagation, fusion, and memory planning.

**Intro paragraph:**  
Graph optimization is not only kernel fusion. Before code generation, the compiler can remove unused work, precompute constants, eliminate duplicate expressions, and choose layouts that reduce transposes.

**Explanation paragraph:**  
The figure should show an unoptimized graph on the left and an optimized graph on the right, with callouts for DCE, constant folding, CSE, layout propagation, and fusion.

**Key takeaway:**  
Good kernels start with a cleaner graph.

**Web-readiness:**  
Use callout bubbles. Avoid too many node names.

**Print-readiness:**  
Ensure graph arrows and labels print clearly.

**Required production fixes:**  
Create source diagram after final section text is settled.

---

## Fig 9.7 — Min-Cut Recomputation Save vs Recompute

**Number:** Fig 9.7  
**Title:** Min-Cut Recomputation: Save Expensive, Recompute Cheap  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** Section `9.8 Min-Cut Recomputation — Fusion and Activation Checkpointing`

**Caption:**  
**Fig 9.7 — Min-Cut Recomputation Save vs Recompute.** Compiler-aware recomputation can save expensive tensors and recompute cheap fused operations, reducing activation memory with less overhead than naive checkpointing.

**Intro paragraph:**  
Activation checkpointing trades compute for memory. A compiler can make this trade more intelligently by assigning each activation a memory size and recompute cost.

**Explanation paragraph:**  
The figure should show a computation graph with a cut line. On one side, large GEMM inputs/outputs are saved; on the other, cheap elementwise/norm intermediates are recomputed. The visual should explain why fusion changes recompute cost.

**Key takeaway:**  
The best checkpointing plan is not “save everything” or “recompute everything.” It is selective.

**Web-readiness:**  
Add simple annotations for `memory saved`, `recompute cost`, and `save set`.

**Print-readiness:**  
Use shapes and labels, not color alone.

**Required production fixes:**  
Create diagram and avoid overpromising exact compute overhead.

---

## Fig 9.8 — CUDA Graph Capture and Replay

**Number:** Fig 9.8  
**Title:** CUDA Graph Capture and Replay  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** Section `9.9 CUDA Graphs — Eliminating CPU Dispatch Overhead`

**Caption:**  
**Fig 9.8 — CUDA Graph Capture and Replay.** CUDA Graphs capture a sequence of GPU operations once and replay it with a single graph launch, reducing CPU dispatch overhead for repeated execution patterns.

**Intro paragraph:**  
CUDA Graphs do not reduce GPU math. They reduce CPU-side launch overhead by replacing many individual kernel submissions with a replayable graph submission.

**Explanation paragraph:**  
The figure should compare an eager timeline with many CPU launch arrows against a CUDA Graph timeline with capture and replay. Add a callout for static-shape/address constraints and dynamic-shape fallback.

**Key takeaway:**  
CUDA Graphs improve overhead-bound workloads by reducing launch overhead, not by changing the math.

**Web-readiness:**  
Timeline visual works well in web format.

**Print-readiness:**  
Ensure timing arrows and launch counts are readable in grayscale.

**Required production fixes:**  
Validate limitations with CUDA Programming Guide. Create SVG and PNG.

---

## Fig 9.9 — Compiler Selection Decision Tree

**Number:** Fig 9.9  
**Title:** Compiler Selection Decision Tree  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** Section `9.10 Compiler Selection Framework — When to Use What`

**Caption:**  
**Fig 9.9 — Compiler Selection Decision Tree.** Choose the compiler path based on model framework, hardware target, shape stability, development velocity, production volume, and latency/cost goals.

**Intro paragraph:**  
There is no universal best compiler. The right choice depends on whether the model is PyTorch or JAX, whether the target is NVIDIA GPU, AMD GPU, or TPU, whether shapes are stable, and whether the model is a research artifact or production service.

**Explanation paragraph:**  
The decision tree should route stable NVIDIA production LLM serving toward TensorRT-LLM; PyTorch dynamic or research workloads toward `torch.compile`/Triton; JAX/TPU toward XLA; and AMD PyTorch workloads toward `torch.compile` with ROCm/Triton support where validated.

**Key takeaway:**  
Compiler choice is a deployment decision, not a brand preference.

**Web-readiness:**  
Use compact branches so sidebar/HTML does not feel cluttered.

**Print-readiness:**  
Keep branch text short and readable.

**Required production fixes:**  
Create SVG/PNG after final table wording is stable.

---

## Table Plan

### Table 9.1 — Three Compiler Optimization Regimes

**Placement:** Section 9.1  
**Purpose:** Convert the PDF's three-regime text into a readable table.

| Regime | Symptom | Compiler Target | Metrics | Common Wrong Fix |
|---|---|---|---|---|
| Memory-bandwidth-bound | HBM high, TFLOPS low | Fusion, fewer HBM round-trips | HBM bytes, roofline, cache behavior | More compute tuning only |
| Compute-bound | Tensor Core utilization high | Kernel selection, tiling, precision | Tensor active %, GEMM throughput | Fusion-only approach |
| Overhead-bound | GPU gaps between kernels | CUDA Graphs, graph-break reduction | Nsys gaps, launch count, CPU time | Kernel tuning before launch overhead |

### Table 9.2 — Operator Fusion Types

**Placement:** Section 9.2  
**Purpose:** Summarize vertical, horizontal, and kernel replacement fusion.

| Fusion Type | Pattern | Main Benefit | Example | Caveat |
|---|---|---|---|---|
| Vertical | Sequential ops | Keeps intermediates out of HBM | residual + RMSNorm | Register pressure |
| Horizontal | Parallel ops | Reads common input once | Q/K/V → QKV projection | Shape/layout compatibility |
| Kernel replacement | New algorithm | Reduces asymptotic or practical HBM traffic | FlashAttention | Requires specialized kernel |

### Table 9.3 — LLaMA Block Fusion Opportunities

**Placement:** Section 9.2.2  
**Purpose:** Convert the PDF's unfused/fused block analysis into production-friendly table form.

| PDF Operation Area | Fusion Opportunity | Bytes Saved Concept | Confidence |
|---|---|---|---|
| Q/K/V projections | Horizontal QKV projection | Avoids multiple reads of `x` | `[ESTIMATED]` |
| RMSNorm | Fused norm | Avoids intermediate norm output | `[REPRESENTATIVE]` |
| RoPE | Fuse into attention path | Removes standalone RoPE pass | `[REPRESENTATIVE]` |
| Residual + Norm | Fused elementwise + reduction | Avoids residual intermediate | `[REPRESENTATIVE]` |
| SwiGLU | Fuse SiLU and multiply | Keeps gate intermediate local | `[REPRESENTATIVE]` |

### Table 9.4 — PyTorch Compiler Layers

**Placement:** Section 9.3  
**Purpose:** Make the four-layer PyTorch compiler stack easy to memorize.

| Layer | Role | What It Produces | Common Failure Mode |
|---|---|---|---|
| TorchDynamo | Captures Python/PyTorch ops into graph | FX graph | Graph breaks |
| AOTAutograd | Captures forward/backward graph | Joint graph | Unsupported backward patterns |
| TorchInductor | Lowers graph and fuses ops | Backend code | Dynamic shape or unsupported op issues |
| Triton/backend compiler | Generates GPU code | PTX/HSACO or backend output | Kernel compile/autotune overhead |

### Table 9.5 — XLA vs TorchInductor

**Placement:** Section 9.4  
**Purpose:** Clarify compiler choice for JAX/TPU vs PyTorch/GPU workflows.

| Dimension | XLA / JAX | TorchInductor / PyTorch |
|---|---|---|
| Primary workflow | JAX/TPU and JAX/GPU/CPU | PyTorch GPU/CPU workflows |
| Compilation style | JIT/AOT-like HLO lowering | Dynamic Python capture + backend lowering |
| Shape behavior | Best with stable/static shapes | Supports dynamic shapes with guards/recompile |
| Debuggability | HLO inspection | FX/TorchDynamo/Inductor diagnostics |
| Best fit | TPU/JAX scale workflows | PyTorch research and production workflows |

### Table 9.6 — TensorRT/TRT-LLM Optimization Passes

**Placement:** Section 9.5  
**Purpose:** Explain offline production compilation.

| Pass | What It Does | Why It Matters |
|---|---|---|
| Graph cleanup | DCE, constant folding, redundant op removal | Removes work before kernels exist |
| Layer fusion | Combines supported patterns | Reduces HBM traffic and launches |
| Precision assignment | Chooses FP8/INT8/FP16 etc. | Balances quality and throughput |
| Kernel selection | Benchmarks/selects implementations | Improves compute-bound kernels |
| Memory planning | Reuses buffers and plans lifetimes | Reduces peak memory |
| Runtime profiles | Captures/builds for supported shapes | Improves stable production latency |

### Table 9.7 — Compiler Selection Framework

**Placement:** Section 9.10  
**Purpose:** Preserve the PDF's compiler selection decision table.

| Use Case | Primary Choice | Secondary Choice | Avoid |
|---|---|---|---|
| Stable NVIDIA LLM serving | TensorRT-LLM | `torch.compile` | Eager hot path |
| PyTorch dynamic/research | `torch.compile` + Triton | Eager for unsupported segments | Offline-only build loops |
| JAX/TPU | XLA / `jax.jit` | N/A | CUDA-specific tools |
| AMD PyTorch GPU | `torch.compile`/Triton ROCm where supported | Native ROCm kernels | NVIDIA-only TRT-LLM |
| Memory-critical training | `torch.compile` + selective checkpointing | manual checkpointing | TRT-LLM inference path |
| Custom kernel research | Triton + PyTorch | CUDA/C++ custom op | Heavy offline engines |

### Table 9.8 — Benchmarking Methodology for Compiler Impact

**Placement:** Section 9.10.2  
**Purpose:** Prevent misleading compiler comparisons.

| Benchmark Rule | Why It Matters |
|---|---|
| Do not time first call | First call may include tracing, codegen, autotuning, and capture |
| Warm up before timing | Allows compile/capture and GPU clocks to stabilize |
| Use synchronization | Avoids measuring asynchronous launches incorrectly |
| Compare same precision | Separates compiler effect from quantization effect |
| Use same shapes and batch mix | Avoids guard/recompile artifacts |
| Report p50/p95/p99 | Compiler changes can affect tail latency differently |
| Inspect graph breaks | Explains unexpectedly small speedups or regressions |

---

## Production Notes for Source Pack

1. Use `ch09_fusion_compiler.pdf` as the source of truth for section order.
2. Convert PDF ASCII blocks into clean Markdown tables and short code snippets.
3. Use confidence labels on all benchmark and speedup examples.
4. Keep the HTML reader-facing and remove this production plan from the HTML.
5. Link existing diagrams as source references but keep placeholder cards until final image assets exist.
