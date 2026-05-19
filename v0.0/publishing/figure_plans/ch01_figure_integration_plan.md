# Chapter 1 Figure Integration Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Chapter 1 — *The AI/ML Performance Architecture Mindset*  
**Target file:** `publishing/figure_plans/ch01_figure_integration_plan.md`  
**Production status:** Draft integration plan for `production-v1.0`  
**Primary goal:** Turn Chapter 1 into the visual and mental-model anchor for the entire book.

---

## 0. Integration Strategy

Chapter 1 introduces the book’s core performance vocabulary: arithmetic intensity, memory bandwidth, communication volume, roofline analysis, the three performance regimes, MFU/HFU, and measurement discipline.

The chapter should not feel like a formula dump. It should visually teach the reader how to think like a Principal AI/ML Performance Architect:

1. Start with the whole-system question.
2. Classify the workload.
3. Identify the limiting regime.
4. Pick the correct metric.
5. Measure before optimizing.
6. Translate findings into architecture decisions.

The current repository already has a strong **Roofline Model — H100 SXM5** diagram in the diagram pack. Chapter 1 should use that as the primary anchor figure. Several additional Chapter 1 figures/tables should be created or adapted so the chapter has a complete visual learning path.

---

# 1. Proposed Chapter 1 Visual Sequence

Recommended flow:

| Order | Figure/Table | Purpose |
|---:|---|---|
| 1 | Fig 1.1 — Three Quantities of AI Performance | Establish the book’s master mental model |
| 2 | Fig 1.2 — Seven-Layer Performance Stack | Show full-stack reasoning from silicon to fleet |
| 3 | Fig 1.3 — H100 Roofline Model | Teach the universal performance model |
| 4 | Table 1.1 — H100 Roofline Quick Reference | Convert diagram into numbers readers can memorize |
| 5 | Fig 1.4 — Three Performance Regimes | Help readers classify bottlenecks |
| 6 | Table 1.2 — Optimization Regime Decision Table | Convert classification into action |
| 7 | Table 1.3 — Wrong Fix vs Right First Question | Train practical engineering judgment |
| 8 | Table 1.4 — MFU vs HFU | Clarify two commonly confused metrics |
| 9 | Fig 1.5 — MFU/HFU System View | Show how useful FLOPs become system-level efficiency |
| 10 | Fig 1.6 — Profiling Hypothesis Loop | Reinforce measurement discipline |
| 11 | Table 1.5 — Metrics by Stack Layer | Map symptoms to metrics/tools |
| 12 | Optional Fig 1.7 — Communication Volume Scaling | Introduce cluster-scale limits early |

---

# 2. Detailed Figure and Table Plan

---

## Fig 1.1 — Three Quantities of AI Performance

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch01_fig_1_1_three_quantities.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch01_fig_1_1_three_quantities.png`  
**Exact section placement:** Immediately after the Chapter 1 opening manifesto, before §1.1.

### Caption

**Fig 1.1 — The Three Quantities of AI Performance.**  
Every AI/ML performance problem can be reduced to arithmetic intensity, memory bandwidth, and communication volume. Compute tells you how much math exists, memory bandwidth tells you how fast data can move locally, and communication volume tells you how much data must cross device, node, rack, or cluster boundaries.

### Intro paragraph before figure

Before looking at any profiler trace, a principal performance engineer asks a simpler question: what quantity is most likely limiting this workload? For a single kernel, the answer is often arithmetic intensity or memory bandwidth. For distributed training and serving, the third quantity — communication volume — becomes just as important. These three quantities form the mental model used throughout the book.

### Explanation paragraph after figure

The value of this model is that it prevents random optimization. A compute-bound GEMM needs better Tensor Core utilization. A memory-bound decode step needs less HBM traffic or better cache behavior. A communication-bound AllReduce needs overlap, topology awareness, or reduced synchronization volume. The rest of Chapter 1 teaches the reader how to classify these cases before reaching for tools.

### Key takeaway box

> **Key Takeaway:** Do not begin with “How do I make this faster?” Begin with “Which quantity is limiting the system: compute, memory movement, or communication?”

### Web-readiness status

**Not ready.** Needs SVG/HTML version, alt text, and responsive layout.

### Print-readiness status

**Not ready.** Needs 300-DPI PNG or vector PDF export.

### Required fixes before production

- Create triangle or three-column visual.
- Use consistent Chapter 1 visual styling.
- Add grayscale-safe labels.
- Add alt text: “Diagram showing arithmetic intensity, memory bandwidth, and communication volume as the three core quantities of AI performance.”
- Export print-safe version.

---

## Fig 1.2 — Seven-Layer Performance Stack

**Type:** New or adapted figure  
**Existing source file:** Possible conceptual reuse from `diagrams/diagram_08_observability_stack.html`, but that file is currently Chapter 17 focused.  
**Status:** Must be created or adapted  
**Recommended asset path:** `assets/diagrams/svg/ch01_fig_1_2_performance_stack.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch01_fig_1_2_performance_stack.png`  
**Exact section placement:** End of §1.1, after introducing the seven-layer performance stack.

### Caption

**Fig 1.2 — The Seven-Layer AI/ML Performance Stack.**  
AI/ML performance problems can originate at any layer: silicon, memory, kernel, compiler/runtime, model architecture, distributed system, or fleet operations. Principal-level performance engineering means reasoning across all seven layers instead of optimizing only the nearest function.

### Intro paragraph before figure

A common difference between senior and principal performance engineering is scope. A senior engineer may optimize a kernel or tune a model-serving parameter. A principal engineer has to decide whether the real bottleneck lives in the kernel, runtime, scheduler, network, power envelope, storage pipeline, or fleet policy. The seven-layer stack gives a structured way to avoid tunnel vision.

### Explanation paragraph after figure

Use the stack as a diagnostic map. If GPU utilization is low, the problem might be a data pipeline, a scheduler gap, communication serialization, or power throttling — not a bad CUDA kernel. If a kernel is slow, the cause might be memory coalescing, shared-memory bank conflicts, Tensor Core underutilization, or compiler-generated code. The stack helps the reader ask the right question at the right layer.

### Key takeaway box

> **Key Takeaway:** The best optimization is often not located where the symptom appears. Always map the symptom to a layer before choosing a fix.

### Web-readiness status

**Not ready.** Existing observability stack is browser-ready but not conceptually correct for Chapter 1 without adaptation.

### Print-readiness status

**Not ready.** Needs a clean vertical layout that fits 7×10 print.

### Required fixes before production

- Create a Chapter 1-specific stack, not just the Chapter 17 observability stack.
- Use these layers:
  1. Silicon / accelerator hardware
  2. Memory hierarchy
  3. Kernel execution
  4. Compiler / runtime
  5. Model architecture
  6. Distributed system
  7. Fleet / business / TCO
- Add one metric and one tool per layer.
- Export as SVG and 300-DPI PNG.
- Ensure labels are readable in print.

---

## Fig 1.3 — H100 Roofline Model

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch1.html`  
**Likely anchor:** `#d1` / Fig 01 in Pack 1  
**Status:** Exists but must be integrated into Chapter 1  
**Recommended source mapping:** `diagrams/diagrams_batch1.html#d1`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch01_fig_1_3_h100_roofline.png`  
**Exact section placement:** Middle of §1.2, immediately after introducing:

```text
Achievable Performance <= min(Peak_FLOPS, Arithmetic_Intensity × Peak_Memory_Bandwidth)
```

### Caption

**Fig 1.3 — Roofline Model for H100 SXM5.**  
The roofline model separates memory-bound operations from compute-bound operations. The ridge point is the arithmetic intensity at which memory bandwidth is no longer the limiter and peak compute becomes the roof.

### Intro paragraph before figure

The roofline model is the first calculation every AI/ML performance engineer should learn. It answers a simple question: given the arithmetic intensity of an operation and the peak capabilities of the GPU, what is the maximum performance we should expect before measuring anything?

### Explanation paragraph after figure

Operations to the left of the ridge point are memory-bound: they do not perform enough math per byte moved to fully use the compute units. Operations to the right of the ridge point are compute-bound: they have enough reuse to approach Tensor Core limits. The position of an operation on this chart tells the engineer whether to reduce memory traffic, increase data reuse, improve kernel efficiency, or focus on compute utilization.

### Key takeaway box

> **Key Takeaway:** The roofline model tells you whether more FLOPs will help. If the workload is below the ridge point, adding compute will not fix the bottleneck.

### Web-readiness status

**Mostly ready.** Existing HTML/SVG diagram works in the browser.

### Print-readiness status

**Not ready.** Needs print export and print-size validation.

### Required fixes before production

- Export the figure to 300-DPI PNG.
- Create vector PDF if possible.
- Verify all labels are readable at 7×10 trim.
- Add alt text.
- Confirm H100 numbers with official vendor documentation.
- Add confidence labels:
  - H100 BF16 peak: `[SHIPPED]`
  - H100 HBM bandwidth: `[SHIPPED]`
  - Ridge point calculation: `[DERIVED FROM SHIPPED]`

---

## Table 1.1 — H100 Roofline Quick Reference

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Immediately after Fig 1.3 and the H100 ridge point derivation.

### Caption

**Table 1.1 — H100 Roofline Quick Reference.**  
A small set of numbers is enough to classify common AI operations on H100-class hardware.

### Proposed table content

| Quantity | Value | Label | Why It Matters |
|---|---:|---|---|
| Peak BF16 Tensor Core throughput | 989 TFLOPS | `[SHIPPED]` after validation | Compute roof |
| HBM bandwidth | 3.35 TB/s | `[SHIPPED]` after validation | Memory roof |
| Ridge point | ~295 FLOP/byte | `[DERIVED]` | Separates memory-bound from compute-bound |
| Decode attention example | ~1 FLOP/byte | `[REPRESENTATIVE]` | Strongly memory-bound |
| LayerNorm example | ~5 FLOP/byte | `[REPRESENTATIVE]` | Memory-bound |
| Small-batch GEMM example | ~30 FLOP/byte | `[REPRESENTATIVE]` | Often memory-limited |
| Large-batch GEMM example | 300+ FLOP/byte | `[REPRESENTATIVE]` | Can become compute-bound |

### Intro paragraph before table

The diagram gives intuition, but the interview and architecture-review version must fit in the reader’s head. The table below gives the minimum set of numbers needed to classify common operations on an H100-class GPU.

### Explanation paragraph after table

The exact arithmetic intensity depends on shape, datatype, implementation, and reuse. The point is not to memorize every operation. The point is to learn the habit of estimating whether the operation is likely below or above the ridge point before opening the profiler.

### Key takeaway box

> **Key Takeaway:** Memorize the ridge point, not every benchmark. The ridge point lets you classify new workloads quickly.

### Web-readiness status

**Ready after table is authored.** Use responsive table styling.

### Print-readiness status

**Needs layout check.** Keep table narrow enough for 7×10.

### Required fixes before production

- Validate H100 numbers.
- Keep table to 4 columns or fewer.
- Avoid long prose in cells.
- Add confidence labels.

---

## Fig 1.4 — Three Performance Regimes

**Type:** New figure or simplified derivative of roofline  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch01_fig_1_4_three_regimes.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch01_fig_1_4_three_regimes.png`  
**Exact section placement:** Beginning of §1.4, before explaining compute-bound, memory-bound, and communication/overhead-bound regimes.

### Caption

**Fig 1.4 — The Three Performance Regimes.**  
A workload can be compute-bound, memory-bound, or communication/overhead-bound. The correct optimization depends on which regime dominates.

### Intro paragraph before figure

Once the reader understands the roofline model, the next step is to translate it into an action framework. Not every slow workload should be optimized the same way. A compute-bound problem, a memory-bound problem, and a communication-bound problem require different fixes.

### Explanation paragraph after figure

The purpose of regime classification is to stop wrong-regime optimization. If decode is memory-bound, optimizing a GEMM kernel may not improve end-to-end latency. If distributed training is communication-bound, buying faster GPUs may worsen utilization. If the system is overhead-bound, both compute and memory may look underutilized while the real issue is scheduling, launch overhead, data loading, or synchronization.

### Key takeaway box

> **Key Takeaway:** Classify first, optimize second. Wrong-regime optimization creates impressive local improvements that do not move system KPIs.

### Web-readiness status

**Not ready.** Needs new SVG/HTML asset.

### Print-readiness status

**Not ready.** Needs clean print export.

### Required fixes before production

- Create a three-column or decision-tree visual.
- Include “symptom,” “metric,” and “typical fix” for each regime.
- Keep labels short and print-safe.
- Add alt text.

---

## Table 1.2 — Optimization Regime Decision Table

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Immediately after Fig 1.4.

### Caption

**Table 1.2 — Optimization Regime Decision Table.**  
The fastest way to choose the right optimization is to identify the regime and match it to the correct metric and tool.

### Proposed table content

| Regime | Common Symptom | Metric to Check | First Tools | Likely Fix |
|---|---|---|---|---|
| Compute-bound | High Tensor Core utilization but still below peak | Achieved TFLOPS, occupancy, TC utilization | Nsight Compute, rocprof-compute | Improve tiling, fusion, Tensor Core use |
| Memory-bound | High HBM bandwidth, low FLOP utilization | DRAM bytes, cache hit rate, arithmetic intensity | Nsight Compute, roofline | Reduce memory traffic, improve reuse, fuse ops |
| Communication-bound | Scaling drops with more GPUs | AllReduce time, bus bandwidth, overlap | Nsight Systems, NCCL tests, IB counters | Overlap comm/compute, topology-aware placement |
| Overhead-bound | GPU gaps, low utilization, short kernels | Kernel gaps, launch overhead, queue time | Nsight Systems, PyTorch profiler | CUDA Graphs, batching, scheduling fixes |
| I/O-bound | GPU waiting for data | DataLoader wait, storage throughput | PyTorch profiler, storage metrics | Prefetch, cache, parallelize data pipeline |

### Intro paragraph before table

The decision table is the operational form of the roofline mindset. It maps observed symptoms to metrics, tools, and likely fixes.

### Explanation paragraph after table

This table should be used before any deep optimization effort. It prevents the common mistake of tuning the component that is easiest to see rather than the component that controls end-to-end throughput or latency.

### Key takeaway box

> **Key Takeaway:** A profiler trace is only useful when paired with a hypothesis. The regime tells you what hypothesis to test.

### Web-readiness status

**Ready after table is authored.** Needs responsive table style.

### Print-readiness status

**Medium risk.** Table has five columns; may need smaller font or split into two tables for 7×10.

### Required fixes before production

- Keep rows short.
- Consider splitting into “Regime → Metric” and “Regime → Fix” if print width is tight.
- Add confidence label where examples are representative.

---

## Table 1.3 — Wrong Fix vs Right First Question

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** End of §1.4, after the regime decision table.

### Caption

**Table 1.3 — Wrong Fix vs Right First Question.**  
Many optimization efforts fail because the first fix targets the wrong layer or the wrong regime.

### Proposed table content

| Symptom | Common Wrong Fix | Right First Question |
|---|---|---|
| Decode latency is high | Optimize GEMM kernel | Is decode memory-bandwidth-bound due to KV cache reads? |
| Training MFU is low | Tune one CUDA kernel | Is the GPU idle due to communication, data loading, or pipeline bubbles? |
| More GPUs do not improve throughput | Add faster GPUs | What percentage of step time is communication? |
| HBM OOM during serving | Reduce batch size only | Can KV cache be paged, quantized, shared, or tiered? |
| Kernel is slow | Rewrite in CUDA immediately | Is it memory-bound, compute-bound, or launch-overhead-bound? |
| P99 latency is high | Increase average throughput | Which queue or stage dominates tail latency? |

### Intro paragraph before table

The table below captures the habit Chapter 1 is trying to build: do not jump from symptom to fix. Insert one diagnostic question in between.

### Explanation paragraph after table

This is the difference between local optimization and system performance engineering. A local fix can improve a kernel while leaving the product KPI unchanged. A principal engineer cares about the bottleneck that controls the measured outcome.

### Key takeaway box

> **Key Takeaway:** Every symptom deserves a diagnostic question before an optimization proposal.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Low to medium risk.** Table is text-heavy; may need page-width layout.

### Required fixes before production

- Keep wording concise.
- Use this table as a visual break after dense formula sections.
- Avoid too many rows in print version.

---

## Table 1.4 — MFU vs HFU

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Start of §1.5, before MFU/HFU formulas.

### Caption

**Table 1.4 — MFU vs HFU.**  
Model FLOPs Utilization and Hardware FLOPs Utilization measure different things. MFU measures useful model work; HFU measures how busy the hardware is, including recomputation or extra work.

### Proposed table content

| Metric | Measures | Includes Recomputation? | Best Used For | Common Misread |
|---|---|---|---|---|
| MFU | Useful model FLOPs / peak hardware FLOPs | Usually no | Training efficiency and scaling quality | Can look low even if hardware is busy doing overhead |
| HFU | Actual hardware FLOPs / peak hardware FLOPs | Yes | Hardware saturation and kernel activity | Can look good while useful model progress is poor |

### Intro paragraph before table

MFU and HFU are often confused because both look like utilization metrics. The distinction matters because a system can keep GPUs busy doing work that does not improve model progress.

### Explanation paragraph after table

A high HFU with a low MFU can indicate recomputation, inefficient kernels, redundant work, or overhead hidden inside hardware activity. A low MFU with low HFU often indicates idle gaps, communication stalls, data stalls, or scheduling inefficiency. Both metrics are useful, but they answer different questions.

### Key takeaway box

> **Key Takeaway:** HFU asks “Are the GPUs busy?” MFU asks “Are the GPUs busy doing useful model work?”

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Low risk.** Table is compact and should fit 7×10.

### Required fixes before production

- Ensure formulas are shown in equation boxes after the table.
- Label healthy MFU ranges as `[ENV-SPECIFIC]`.
- Add one worked example after the table.

---

## Fig 1.5 — MFU/HFU System View

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch01_fig_1_5_mfu_hfu_system_view.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch01_fig_1_5_mfu_hfu_system_view.png`  
**Exact section placement:** Middle of §1.5, after MFU/HFU formulas and before the worked example.

### Caption

**Fig 1.5 — From Tokens to Utilization: MFU and HFU in the Training Loop.**  
MFU converts model progress into useful FLOPs per second. HFU measures how much hardware work was executed. The gap between the two helps identify recomputation, overhead, and inefficiency.

### Intro paragraph before figure

A training job is not just a sequence of kernels. It is a repeated loop: forward pass, backward pass, gradient communication, optimizer update, checkpointing, and input pipeline. MFU and HFU summarize how effectively that loop uses the available hardware.

### Explanation paragraph after figure

When MFU is low, the next question is not automatically “which kernel is slow?” The system may be losing time to AllReduce, pipeline bubbles, data loading, checkpointing, stragglers, or framework overhead. The figure should show useful model work flowing through the training step and separate it from extra hardware activity and idle time.

### Key takeaway box

> **Key Takeaway:** MFU is a system-level KPI. Treat it as a starting point for investigation, not as a final diagnosis.

### Web-readiness status

**Not ready.** Needs new figure.

### Print-readiness status

**Not ready.** Needs clean 300-DPI/vector export.

### Required fixes before production

- Create training-loop visual.
- Show “useful model FLOPs,” “extra/recomputed FLOPs,” and “idle/wait time.”
- Add formula callout:
  - `MFU = Useful Model FLOPs / Peak Hardware FLOPs`
  - `HFU = Actual Hardware FLOPs / Peak Hardware FLOPs`
- Ensure formulas are readable in print.
- Add alt text.

---

## Fig 1.6 — Profiling Hypothesis Loop

**Type:** New or adapted figure  
**Existing source file:** Possible adaptation from Pack 3 profiling decision tree, likely `diagrams/diagrams_batch3.html` Fig 30.  
**Status:** Should be created or adapted  
**Recommended asset path:** `assets/diagrams/svg/ch01_fig_1_6_profiling_hypothesis_loop.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch01_fig_1_6_profiling_hypothesis_loop.png`  
**Exact section placement:** End of §1.6, after measurement discipline discussion.

### Caption

**Fig 1.6 — The Performance Engineer’s Hypothesis Loop.**  
Production optimization should follow a loop: classify the workload, form a hypothesis, measure the right metric, change one variable, validate the outcome, and decide whether to continue or stop.

### Intro paragraph before figure

Profilers are powerful, but they should not be used as fishing tools. A profiler should confirm or reject a hypothesis. The hypothesis loop turns optimization into a reproducible engineering process.

### Explanation paragraph after figure

This loop is what separates debugging from performance engineering. It records the reason for each experiment, the metric being moved, the expected direction, and the actual result. It also creates a written trail that can be reviewed by hardware teams, framework teams, leadership, or interviewers.

### Key takeaway box

> **Key Takeaway:** The profiler confirms the hypothesis; it should not be the only source of the hypothesis.

### Web-readiness status

**Partially ready if adapted from existing profiling decision tree.** Needs Chapter 1-specific simplification.

### Print-readiness status

**Not ready.** Existing diagram may be too detailed if reused directly.

### Required fixes before production

- Create a simple loop diagram:
  1. Estimate
  2. Classify
  3. Hypothesize
  4. Measure
  5. Change one variable
  6. Validate
  7. Document
- Add small “do not optimize randomly” note.
- Export print-safe version.
- Add alt text.

---

## Table 1.5 — Metrics by Stack Layer

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** End of §1.6 or beginning of §1.7.

### Caption

**Table 1.5 — Metrics by Stack Layer.**  
The correct metric depends on the layer where the bottleneck lives.

### Proposed table content

| Layer | Symptom | Metric | Tool |
|---|---|---|---|
| Silicon / GPU | Low math throughput | SM utilization, Tensor Core utilization | DCGM, Nsight Compute |
| Memory | Low FLOPs, high traffic | HBM bandwidth, cache hit rate | Nsight Compute |
| Kernel | Slow operation | Kernel duration, occupancy, stalls | Nsight Compute |
| Runtime / Compiler | Many tiny kernels or graph breaks | Launch gaps, graph breaks | Nsight Systems, PyTorch profiler |
| Distributed system | Poor scaling | AllReduce time, busbw, overlap | NCCL tests, Nsight Systems |
| Storage / input | GPU idle between steps | Data wait, loader time | PyTorch profiler, storage metrics |
| Fleet | Poor utilization/cost | MFU, queue time, $/token | Prometheus, Grafana, cost model |

### Intro paragraph before table

Once a hypothesis exists, the next question is which metric can test it. The table below maps common layers of the performance stack to observable symptoms, metrics, and tools.

### Explanation paragraph after table

The table should be used as a diagnostic routing guide. If the symptom is GPU idle time, a kernel-level counter may not be the first tool. If the symptom is poor scaling, a single-GPU benchmark may not reveal the issue. Start with the layer that matches the symptom.

### Key takeaway box

> **Key Takeaway:** Use the metric that matches the layer. A kernel profiler cannot fully explain a queueing, communication, or fleet-utilization problem.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Medium risk.** Seven rows and four columns should fit, but text must be concise.

### Required fixes before production

- Keep tool names short.
- Use line breaks carefully in print.
- Cross-reference Appendix B.

---

## Optional Fig 1.7 — Communication Volume Scaling

**Type:** Optional new figure  
**Existing source file:** None  
**Status:** Optional but recommended  
**Recommended asset path:** `assets/diagrams/svg/ch01_fig_1_7_communication_scaling.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch01_fig_1_7_communication_scaling.png`  
**Exact section placement:** §1.8, where scaling laws and performance implications are introduced.

### Caption

**Fig 1.7 — Communication Volume Becomes the Bottleneck at Scale.**  
As GPU count grows, local compute capacity increases faster than the ability to synchronize, reduce, and move data across devices and nodes.

### Intro paragraph before figure

Roofline analysis explains single-device performance, but large-scale AI infrastructure adds another limiter: communication volume. A workload that is efficient on one GPU can become communication-bound at 256, 1,024, or 10,000 GPUs.

### Explanation paragraph after figure

This figure should show that scaling is not just “more GPUs equals more throughput.” The system must move gradients, activations, parameters, KV cache blocks, checkpoints, and telemetry. At sufficient scale, communication and orchestration become first-order performance constraints.

### Key takeaway box

> **Key Takeaway:** At cluster scale, the question is not only “How much compute do we have?” It is “How much useful compute survives communication, synchronization, and orchestration overhead?”

### Web-readiness status

**Not ready.** Needs new figure.

### Print-readiness status

**Not ready.** Needs export.

### Required fixes before production

- Keep simple.
- Do not duplicate Chapter 14 networking detail.
- Use this as a teaser for distributed training and networking chapters.
- Add cross-reference to Ch10 and Ch14.

---

# 3. Figure Numbering Recommendation

Use the following final numbering for Chapter 1:

| Number | Asset |
|---|---|
| Fig 1.1 | Three Quantities of AI Performance |
| Fig 1.2 | Seven-Layer AI/ML Performance Stack |
| Fig 1.3 | H100 Roofline Model |
| Fig 1.4 | Three Performance Regimes |
| Fig 1.5 | MFU/HFU System View |
| Fig 1.6 | Profiling Hypothesis Loop |
| Fig 1.7 | Communication Volume Scaling, optional |

Use the following final table numbering:

| Number | Table |
|---|---|
| Table 1.1 | H100 Roofline Quick Reference |
| Table 1.2 | Optimization Regime Decision Table |
| Table 1.3 | Wrong Fix vs Right First Question |
| Table 1.4 | MFU vs HFU |
| Table 1.5 | Metrics by Stack Layer |

---

# 4. Required Updates to `publishing/figure_inventory.md`

Add or update these rows:

```markdown
| Figure | Title | Current Asset | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|---|
| Fig 1.1 | Three Quantities of AI Performance | TBD | Ch01 | No | No | Create SVG + print export |
| Fig 1.2 | Seven-Layer AI/ML Performance Stack | TBD / adapt diagram_08_observability_stack.html | Ch01 | No | No | Create Ch01-specific version |
| Fig 1.3 | H100 Roofline Model | diagrams/diagrams_batch1.html#d1 | Ch01 | Yes | No | Export 300-DPI/vector and validate labels |
| Fig 1.4 | Three Performance Regimes | TBD | Ch01 | No | No | Create decision/regime figure |
| Fig 1.5 | MFU/HFU System View | TBD | Ch01 | No | No | Create system-view figure |
| Fig 1.6 | Profiling Hypothesis Loop | TBD / adapt diagrams_batch3.html Fig 30 | Ch01 | Partial | No | Create simplified Ch01 loop |
| Fig 1.7 | Communication Volume Scaling | TBD | Ch01 | No | No | Optional; create if section 1.8 remains dense |
```

---

# 5. Production Checklist for Chapter 1 Visuals

## Web Checklist

- [ ] Embed Fig 1.3 roofline inside Chapter 1 HTML page.
- [ ] Add SVG or responsive image versions for all figures.
- [ ] Add alt text for each figure.
- [ ] Add anchors for each figure and table.
- [ ] Add “Back to top” or mini-TOC behavior.
- [ ] Ensure tables scroll horizontally on mobile.
- [ ] Use lazy loading for images.

## Print Checklist

- [ ] Export each figure at 300 DPI or vector PDF.
- [ ] Validate figure readability at 7×10 trim.
- [ ] Check grayscale readability.
- [ ] Keep captions with figures.
- [ ] Avoid figure splits across pages.
- [ ] Test tables for page-width overflow.
- [ ] Confirm all formulas render cleanly.

## Technical Validation Checklist

- [ ] H100 BF16 TFLOPS validated.
- [ ] H100 HBM bandwidth validated.
- [ ] H100 ridge point recomputed.
- [ ] MI300X reference values validated if used.
- [ ] MFU/HFU definitions validated.
- [ ] FLOPs/token assumptions documented.
- [ ] Representative examples labeled correctly.

---

# 6. Recommended Next Commit

After saving this file as:

```text
publishing/figure_plans/ch01_figure_integration_plan.md
```

Run:

```powershell
git add publishing\figure_plans\ch01_figure_integration_plan.md
git commit -m "Add Chapter 1 figure integration plan"
git push origin production-v1.0
```

Then update the master figure inventory:

```powershell
git add publishing\figure_inventory.md
git commit -m "Update figure inventory for Chapter 1"
git push origin production-v1.0
```

---

# 7. Next Production Step After This Plan

The next production task should be:

```text
Chapter 1 Technical Validation Plan
```

Recommended file:

```text
publishing/validation/ch01_technical_validation.md
```

The validation should cover:

1. H100 BF16 peak TFLOPS
2. H100 HBM bandwidth
3. H100 ridge point
4. MI300X BF16 and HBM values if referenced
5. MFU/HFU definitions
6. Training FLOPs/token formula
7. Inference FLOPs/token formula
8. Communication-bound examples
9. Confidence labels
10. Required citations or source notes
