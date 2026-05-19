# Chapter 4 Figure Integration Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Ch04 — *GPU Memory Hierarchy and HBM Deep Dive*  
**Target file:** `publishing/figure_plans/ch04_figure_integration_plan.md`  
**Production status:** Production Planning Pack  
**Last reviewed:** 2026-04-30

---

## 0. Visual Strategy

Chapter 4 should visually teach:

```text
where data lives → how far it moves → how much bandwidth is available → which optimization reduces movement
```

The chapter already has strong existing assets:

- Memory Hierarchy Pyramid — H100 SXM5
- GPU Memory Hierarchy — Bandwidth View
- HBM3e Die Stacking vs GDDR6X
- FlashAttention tiling diagram as a cross-reference
- KV cache / PagedAttention diagram as a cross-reference

New visuals should focus on:

- memory-bound vs compute-bound decision flow
- HBM traffic waterfall
- KV cache growth
- memory optimization decision tree

---

# 1. Proposed Figure/Table Sequence

| Order | Figure/Table | Purpose |
|---:|---|---|
| 1 | Fig 4.1 — Memory Hierarchy Pyramid | Establish memory tiers |
| 2 | Table 4.1 — GPU Memory Tier Summary | Make tiers practical |
| 3 | Fig 4.2 — GPU Memory Hierarchy Bandwidth View | Show bandwidth cliff |
| 4 | Fig 4.3 — HBM3e Die Stacking vs GDDR6X | Explain HBM physically |
| 5 | Table 4.2 — HBM Reference Values | Compare product-specific HBM |
| 6 | Fig 4.4 — Memory-Bound vs Compute-Bound Decision Flow | Connect to Roofline |
| 7 | Table 4.3 — Memory Bottleneck Diagnostic Signals | Profiling guidance |
| 8 | Table 4.4 — KV Cache Formula Variables | Prevent formula confusion |
| 9 | Fig 4.5 — KV Cache Memory Growth Curve | Show context/concurrency pressure |
| 10 | Fig 4.6 — FlashAttention HBM Traffic Reduction | Show HBM traffic reduction |
| 11 | Table 4.5 — Memory Optimization Techniques | Practical optimization menu |
| 12 | Table 4.6 — Wrong Fix vs Right First Question | Principal-level diagnostic table |
| 13 | Fig 4.7 — Memory Optimization Decision Tree | Final synthesis |

---

# 2. Detailed Figure and Table Plan

---

## Fig 4.1 — Memory Hierarchy Pyramid: H100 SXM5

**Type:** Existing figure  
**Existing source file:** `diagrams/diagram_01_memory_hierarchy.html`  
**Status:** Exists; must be integrated  
**Recommended print export:** `assets/diagrams/png_300dpi/ch04_fig_4_1_memory_hierarchy_pyramid.png`  
**Exact section placement:** Opening memory hierarchy section, after the “Memory in One Page” explanation.

### Caption

**Fig 4.1 — GPU Memory Hierarchy Pyramid.**  
GPU memory is a hierarchy of capacity, latency, and bandwidth tradeoffs: registers and shared memory are closest to the math, while HBM, host memory, and network memory paths are farther away and more expensive to access.

### Intro paragraph before figure

Every memory optimization starts with one question: where does the data live when the math needs it? The answer determines whether the workload sees fast on-chip reuse, HBM bandwidth pressure, PCIe latency, or network communication overhead.

### Explanation paragraph after figure

The pyramid shows why locality matters. Data reused in registers or shared memory can feed compute efficiently. Data repeatedly fetched from HBM may become the bottleneck. Data that spills to host memory or crosses the network can dominate latency and destroy throughput.

### Key takeaway box

> **Key Takeaway:** The farther data is from the compute units, the more carefully the workload must reuse, compress, tile, fuse, or schedule it.

### Web-readiness status

**Ready.** Existing standalone HTML diagram available.

### Print-readiness status

**Not ready.** Needs 300-DPI PNG/vector export.

### Required production fixes

- Export print-safe version.
- Validate H100-specific labels.
- Add alt text.
- Test readability at print trim size.
- Ensure it does not duplicate Ch03A too heavily; Ch04 should go deeper into performance implications.

---

## Table 4.1 — GPU Memory Tier Summary

**Type:** New table  
**Existing source file:** Concept exists across Ch03A/Ch04; create clean Ch04 table  
**Status:** Must be created  
**Exact section placement:** Immediately after Fig 4.1.

### Caption

**Table 4.1 — GPU Memory Tier Summary.**  
Each memory tier has different scope, capacity, latency, bandwidth, and optimization strategy.

### Proposed table content

| Tier | Scope | Capacity | What It Is Good For | Main Risk |
|---|---|---:|---|---|
| Registers | Per thread | Tiny | Fast operands and accumulator fragments | Spilling if register pressure is high |
| Shared memory / L1 | Per SM | Small | Tiling and producer/consumer reuse | Bank conflicts, limited capacity |
| L2 cache | Whole GPU | Medium | Cross-SM reuse and reduced HBM traffic | Low reuse workloads bypass benefits |
| HBM | Whole GPU | Large | Model weights, activations, KV cache | Bandwidth and capacity bottlenecks |
| Host memory | CPU side | Very large | Staging and offload | PCIe latency/bandwidth penalty |
| Network memory path | Remote GPUs/nodes | Remote | Distributed workloads | Fabric latency, congestion, collectives |

### Intro paragraph before table

The figure shows the hierarchy visually. The table translates it into performance questions: who can access this memory, how big is it, and what goes wrong when the workload depends on it too heavily?

### Explanation paragraph after table

A memory-bound workload is not simply “using memory.” It is using the wrong memory tier too often, with too little reuse, or with too many bytes per unit of useful math.

### Key takeaway box

> **Key Takeaway:** The memory tier that supplies the critical path usually determines the optimization strategy.

### Web-readiness status

**Ready after authoring.**

### Print-readiness status

**Medium risk.** Keep compact.

### Required production fixes

- Keep table concise.
- Do not include too many numeric values.
- Move product-specific HBM specs to Table 4.2.

---

## Fig 4.2 — GPU Memory Hierarchy: Bandwidth View

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch1.html#d2`  
**Status:** Exists; must be integrated  
**Recommended print export:** `assets/diagrams/png_300dpi/ch04_fig_4_2_gpu_memory_bandwidth_view.png`  
**Exact section placement:** After Table 4.1 and before the memory-bound vs compute-bound discussion.

### Caption

**Fig 4.2 — GPU Memory Hierarchy Bandwidth View.**  
Bandwidth drops sharply as data moves farther from compute. AI performance often depends on minimizing HBM traffic and maximizing on-chip reuse.

### Intro paragraph before figure

The memory hierarchy is not only about capacity. It is also about bandwidth. A workload that repeatedly streams through HBM can become bandwidth-bound even on a GPU with enormous Tensor Core throughput.

### Explanation paragraph after figure

This bandwidth view connects directly to Roofline analysis. If an operation has low arithmetic intensity, it cannot reach the compute roof no matter how large the TFLOPS number is. Improving the workload often means increasing reuse or reducing bytes moved.

### Key takeaway box

> **Key Takeaway:** Peak compute matters only if the memory hierarchy can feed it.

### Web-readiness status

**Ready.** Existing diagram available in Pack 1.

### Print-readiness status

**Not ready.** Needs export.

### Required production fixes

- Export 300-DPI/vector.
- Validate bandwidth labels.
- Add alt text.
- Cross-reference Ch01 Roofline model.

---

## Fig 4.3 — HBM3e Die Stacking vs GDDR6X

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch2.html#d13`  
**Status:** Exists; must be integrated  
**Recommended print export:** `assets/diagrams/png_300dpi/ch04_fig_4_3_hbm3e_die_stacking.png`  
**Exact section placement:** HBM physical architecture section.

### Caption

**Fig 4.3 — HBM3e Die Stacking vs GDDR-Style Memory.**  
HBM uses stacked memory dies and a very wide interface close to the GPU package, enabling much higher bandwidth and energy efficiency than conventional off-package memory approaches.

### Intro paragraph before figure

HBM is not just “more VRAM.” It is a packaging and bandwidth technology. By stacking DRAM dies near the GPU and using a very wide interface, HBM delivers the bandwidth needed by modern AI accelerators.

### Explanation paragraph after figure

This physical structure explains why HBM capacity and bandwidth are expensive, power-sensitive, package-sensitive, and central to accelerator roadmap decisions. It also explains why HBM is one of the hardest resources to scale quickly.

### Key takeaway box

> **Key Takeaway:** HBM is a packaging-level performance feature, not just a memory-size feature.

### Web-readiness status

**Ready.** Existing diagram available.

### Print-readiness status

**Not ready.** Needs export.

### Required production fixes

- Export print-safe image.
- Avoid overclaiming generic HBM bandwidth.
- Use product-specific HBM values in Table 4.2.
- Add alt text.

---

## Table 4.2 — HBM Reference Values by Accelerator

**Type:** New validation-sensitive table  
**Existing source file:** None; use technical validation  
**Status:** Must be created after validation  
**Exact section placement:** After Fig 4.3.

### Caption

**Table 4.2 — HBM Reference Values by Accelerator.**  
HBM capacity and bandwidth are product-specific. Compare HBM values by accelerator SKU or system level, not by memory generation alone.

### Proposed table content

| Accelerator / Product | Product Level | Memory | Peak Bandwidth | Confidence |
|---|---|---:|---:|---|
| H100 SXM5 | GPU | 80 GB HBM3 | 3.35 TB/s | `[SHIPPED]` |
| H200 | GPU | 141 GB HBM3e | 4.8 TB/s | `[SHIPPED]` |
| MI300X | GPU / OAM accelerator | 192 GB HBM3 | ≈5.3 TB/s | `[SHIPPED]` |
| MI325X | GPU / OAM accelerator | 256 GB HBM3e | ≈6 TB/s | `[SHIPPED]` or `[ANNOUNCED]` |
| MI350 Series | GPU / OAM accelerator | 288 GB HBM3e | ≈8 TB/s | `[SHIPPED]` if official product source |
| DGX B200 | System | 1,440 GB total GPU memory | 64 TB/s total HBM3e | `[SHIPPED]` system-level |

### Intro paragraph before table

The easiest way to misuse HBM data is to compare different product levels. A per-GPU memory value is not the same as a system-level aggregate value. This table keeps product level visible.

### Explanation paragraph after table

For inference decisions, per-GPU HBM capacity affects model fit and KV-cache headroom. For system-level products, total HBM capacity is useful only when the workload and parallelism strategy can use it efficiently.

### Key takeaway box

> **Key Takeaway:** Always ask whether an HBM number is per GPU, per module, per system, or per rack.

### Web-readiness status

**Ready after validation.**

### Print-readiness status

**High risk.** Table may be wide; consider split current vs system-level.

### Required production fixes

- Validate all values in `ch04_technical_validation.md`.
- Add “Current as of 2026 edition.”
- Label product level clearly.
- Cross-reference Appendix A.

---

## Fig 4.4 — Memory-Bound vs Compute-Bound Decision Flow

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch04_fig_4_4_memory_compute_bound_flow.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch04_fig_4_4_memory_compute_bound_flow.png`  
**Exact section placement:** In section connecting Ch04 to Ch01 Roofline model.

### Caption

**Fig 4.4 — Memory-Bound vs Compute-Bound Decision Flow.**  
Use arithmetic intensity and ridge point to decide whether the next optimization should target math throughput or data movement.

### Intro paragraph before figure

A workload is not memory-bound because memory is involved. It is memory-bound when bytes moved per unit of math are too high for the hardware ridge point.

### Explanation paragraph after figure

The decision flow should route the reader from symptoms to action. Low Tensor Core utilization may mean compute underuse, but it may also mean memory stalls. The key is to compare arithmetic intensity to the ridge point and then confirm with profiler metrics.

### Key takeaway box

> **Key Takeaway:** Classify the bottleneck before optimizing. Memory-bound workloads need fewer bytes moved, not just more compute.

### Web-readiness status

**Not ready.** New SVG needed.

### Print-readiness status

**Not ready.**

### Required production fixes

- Create simple flowchart.
- Include formula:
  - arithmetic intensity = FLOPs / bytes moved
  - ridge point = peak FLOPs / memory bandwidth
- Add H100 example as label.
- Cross-reference Ch01.

---

## Table 4.3 — Memory Bottleneck Diagnostic Signals

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** After Fig 4.4.

### Caption

**Table 4.3 — Memory Bottleneck Diagnostic Signals.**  
Profiler symptoms should be mapped to memory hierarchy causes before selecting an optimization.

### Proposed table content

| Symptom | Possible Cause | What to Check | Possible Fix |
|---|---|---|---|
| Low compute utilization | Memory stalls | HBM bandwidth, cache hit rate, stall reasons | Fusion, tiling, quantization |
| High HBM bandwidth | Streaming low-reuse kernel | Bytes moved per token/layer | Fuse ops, reduce precision |
| OOM at long context | KV cache growth | KV bytes per token and concurrency | GQA/MQA, KV quantization, paging |
| High PCIe traffic | CPU/GPU fallback or offload | Host-device transfer counters | Keep hot path on GPU |
| Poor scaling | Remote memory/collectives | NCCL/RCCL traces | Topology-aware placement |
| Slow attention | Too much HBM traffic | Attention kernel memory reads/writes | FlashAttention-style tiling |

### Intro paragraph before table

A memory bottleneck should be diagnosed with evidence. The same low-utilization symptom can come from HBM bandwidth, cache misses, register spills, PCIe transfer, or communication.

### Explanation paragraph after table

The table helps prevent the common mistake of tuning GEMM when the bottleneck is actually KV-cache bandwidth, normalization traffic, or host-device movement.

### Key takeaway box

> **Key Takeaway:** Memory bottlenecks have signatures. Match the symptom to the memory tier before choosing the fix.

### Web-readiness status

**Ready after authoring.**

### Print-readiness status

**Medium risk.**

### Required production fixes

- Keep table compact.
- Avoid tool-specific metric names unless cross-referenced to Ch12/Ch17.
- Add profiler tool examples in prose.

---

## Table 4.4 — KV Cache Memory Formula Variables

**Type:** New table  
**Existing source file:** Ch11 has deeper KV cache content; Ch04 should include formula overview  
**Status:** Must be created  
**Exact section placement:** KV-cache memory pressure section.

### Caption

**Table 4.4 — KV Cache Memory Formula Variables.**  
KV-cache memory grows with layers, concurrent sequences, sequence length, KV heads, head dimension, and datatype size.

### Formula

```text
KV cache bytes ≈ 2 × L × B × S × n_kv × d_head × bytes_per_element
```

### Proposed table content

| Symbol | Meaning | Notes |
|---|---|---|
| `2` | Key and value tensors | One K and one V cache |
| `L` | Number of transformer layers | Model architecture dependent |
| `B` | Batch size / concurrent sequences | Serving concurrency |
| `S` | Sequence length | Grows with context |
| `n_kv` | Number of KV heads | Lower for GQA/MQA |
| `d_head` | Head dimension | Usually hidden size / attention heads |
| `bytes_per_element` | KV dtype size | BF16/FP16=2, FP8/INT8=1 |

### Intro paragraph before table

KV cache is one of the most important memory structures in LLM serving. During decode, every new token must read prior context state, so KV cache affects both capacity and bandwidth.

### Explanation paragraph after table

The formula shows why long-context serving can become memory-limited quickly. Reducing KV heads with GQA/MQA, reducing precision, paging KV blocks, and prefix caching all target this memory pressure.

### Key takeaway box

> **Key Takeaway:** KV-cache growth is linear in layers, concurrency, sequence length, KV heads, head dimension, and bytes per element.

### Web-readiness status

**Ready after authoring.**

### Print-readiness status

**Low risk if formula is boxed.**

### Required production fixes

- Use consistent symbols with Ch02 and Ch11.
- Explain GQA/MQA difference.
- Cross-reference Ch11 for full KV-cache chapter.

---

## Fig 4.5 — KV Cache Memory Growth Curve

**Type:** New figure  
**Existing source file:** None; can cross-reference Ch11 PagedAttention assets  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch04_fig_4_5_kv_cache_growth_curve.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch04_fig_4_5_kv_cache_growth_curve.png`  
**Exact section placement:** After Table 4.4.

### Caption

**Fig 4.5 — KV Cache Memory Growth with Sequence Length and Concurrency.**  
KV-cache memory grows linearly with sequence length and concurrent sequences, making long-context serving a memory-capacity and memory-bandwidth problem.

### Intro paragraph before figure

A model may fit in HBM at short context but fail at long context or high concurrency. KV cache is often the difference between a theoretical deployment and a production deployment.

### Explanation paragraph after figure

The growth curve should show why batch size, sequence length, and KV precision interact. A serving system needs headroom for weights, KV cache, temporary buffers, fragmentation, and scheduler overhead.

### Key takeaway box

> **Key Takeaway:** Long context is not free. It consumes HBM capacity and increases decode memory traffic.

### Web-readiness status

**Not ready.** New figure needed.

### Print-readiness status

**Not ready.**

### Required production fixes

- Create simple curve or stacked area chart.
- Use representative values only.
- Label `[ESTIMATED]`.
- Cross-reference Ch11 for exact model-specific calculations.

---

## Fig 4.6 — FlashAttention HBM Traffic Reduction

**Type:** Existing cross-reference or new adaptation  
**Existing source file:** `diagrams/diagrams_batch1.html#d5`  
**Status:** Existing Ch07 asset can be reused/cross-referenced; Ch04 may need simplified memory-traffic version  
**Recommended print export:** `assets/diagrams/png_300dpi/ch04_fig_4_6_flashattention_hbm_traffic.png`  
**Exact section placement:** FlashAttention case study section.

### Caption

**Fig 4.6 — FlashAttention and HBM Traffic Reduction.**  
FlashAttention-style kernels reduce HBM reads and writes by tiling attention and keeping intermediate state in on-chip memory instead of materializing the full attention matrix in HBM.

### Intro paragraph before figure

Attention is a perfect example of why memory hierarchy matters. A naive attention implementation can write and reread large intermediate matrices. FlashAttention changes the memory schedule.

### Explanation paragraph after figure

FlashAttention does not make attention free. It changes the bottleneck by reducing HBM traffic and increasing on-chip reuse. Exact speedups depend on sequence length, head dimension, dtype, GPU generation, and implementation.

### Key takeaway box

> **Key Takeaway:** FlashAttention is a memory optimization: it wins by moving fewer bytes through HBM.

### Web-readiness status

**Partially ready.** Existing Ch07 diagram available.

### Print-readiness status

**Not ready.** Needs export or Ch04-specific adaptation.

### Required production fixes

- Decide whether to reuse Ch07 diagram or create Ch04-specific traffic view.
- Label speedup claims `[REPRESENTATIVE]` or `[ENV-SPECIFIC]`.
- Add cross-reference to Ch07.

---

## Table 4.5 — Memory Optimization Techniques

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** After FlashAttention case study.

### Caption

**Table 4.5 — Memory Optimization Techniques.**  
Memory optimization means reducing bytes moved, improving reuse, lowering precision, avoiding offload, or changing the schedule.

### Proposed table content

| Technique | Reduces | Best For | Caveat |
|---|---|---|---|
| Tiling | HBM traffic | GEMM, attention | Requires kernel support |
| Fusion | Intermediate reads/writes | Norm, activation, elementwise chains | May increase register pressure |
| Quantization | Weight/KV bytes | Inference | Quality and kernel support matter |
| KV quantization | KV-cache memory/bandwidth | Decode and long context | Accuracy and implementation matter |
| GQA/MQA | KV heads | LLM inference | Architecture-specific |
| Paging | Fragmentation/OOM | Serving systems | Scheduler overhead |
| Recomputation | Activation memory | Training | More compute |
| CPU offload | HBM capacity | Oversized models | PCIe latency/bandwidth penalty |
| Topology-aware placement | Remote memory/comm | Distributed workloads | Requires scheduler/fabric awareness |

### Intro paragraph before table

Once the memory tier and bottleneck are known, the optimization menu becomes clearer. The best fix depends on whether the problem is HBM capacity, HBM bandwidth, on-chip reuse, fragmentation, or offload.

### Explanation paragraph after table

Some techniques trade memory for compute. Others trade quality risk for capacity. Others trade implementation complexity for bandwidth. A principal engineer should name the tradeoff explicitly.

### Key takeaway box

> **Key Takeaway:** Memory optimization is tradeoff management, not a single trick.

### Web-readiness status

**Ready after authoring.**

### Print-readiness status

**Medium risk.**

### Required production fixes

- Keep concise.
- Add confidence labels in prose.
- Cross-reference Ch06, Ch07, Ch08, Ch11.

---

## Table 4.6 — Wrong Fix vs Right First Question

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Near end of chapter before decision tree.

### Caption

**Table 4.6 — Wrong Fix vs Right First Question for Memory Problems.**  
Memory bottlenecks are often misdiagnosed. The right first question prevents wasted optimization effort.

### Proposed table content

| Symptom | Wrong Fix | Right First Question |
|---|---|---|
| Low GPU utilization | Add more GPUs | Is the kernel memory-bound or launch/overhead-bound? |
| High decode latency | Use a bigger GPU for TFLOPS | Is KV-cache bandwidth the bottleneck? |
| OOM at long context | Reduce batch blindly | How many bytes does KV cache consume per request? |
| Slow attention | Tune only GEMM | Is attention materializing too much HBM traffic? |
| Slow norm/activation chain | Increase Tensor Core focus | Are elementwise ops streaming HBM repeatedly? |
| Poor multi-GPU scaling | Add faster GPUs | Is remote communication or topology the bottleneck? |

### Intro paragraph before table

Memory problems often look like compute problems from far away. This table helps the reader slow down and ask the diagnostic question before changing hardware or rewriting kernels.

### Explanation paragraph after table

The table is especially useful in principal interviews because it shows disciplined problem framing. A senior answer jumps to a fix. A principal answer first identifies the resource and tradeoff.

### Key takeaway box

> **Key Takeaway:** The fastest way to waste optimization time is to fix the wrong bottleneck.

### Web-readiness status

**Ready after authoring.**

### Print-readiness status

**Medium risk.**

### Required production fixes

- Keep wording sharp.
- Place before principal-interview section or as part of it.
- Use in LinkedIn/social snippet later.

---

## Fig 4.7 — Memory Optimization Decision Tree

**Type:** New synthesis figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch04_fig_4_7_memory_optimization_decision_tree.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch04_fig_4_7_memory_optimization_decision_tree.png`  
**Exact section placement:** End of chapter before key takeaways and review questions.

### Caption

**Fig 4.7 — Memory Optimization Decision Tree.**  
Choose memory optimizations by identifying whether the bottleneck is HBM capacity, HBM bandwidth, on-chip reuse, PCIe/host offload, or interconnect.

### Intro paragraph before figure

The chapter should end with a practical decision tool. Once the bottleneck is classified, the solution space becomes smaller and more rational.

### Explanation paragraph after figure

The decision tree should route capacity problems toward quantization, sharding, paging, or offload; bandwidth problems toward fusion, tiling, KV quantization, and layout; and communication problems toward topology-aware placement and overlap.

### Key takeaway box

> **Key Takeaway:** Memory optimization starts by naming the limiting memory tier.

### Web-readiness status

**Not ready.** New SVG needed.

### Print-readiness status

**Not ready.**

### Required production fixes

- Create simple flowchart.
- Keep text readable in print.
- Add alt text.
- Cross-reference Ch12 profiling methodology.

---

# 3. Final Figure Numbering Recommendation

| Number | Title |
|---|---|
| Fig 4.1 | Memory Hierarchy Pyramid — H100 SXM5 |
| Fig 4.2 | GPU Memory Hierarchy — Bandwidth View |
| Fig 4.3 | HBM3e Die Stacking vs GDDR-Style Memory |
| Fig 4.4 | Memory-Bound vs Compute-Bound Decision Flow |
| Fig 4.5 | KV Cache Memory Growth Curve |
| Fig 4.6 | FlashAttention HBM Traffic Reduction |
| Fig 4.7 | Memory Optimization Decision Tree |

---

# 4. Final Table Numbering Recommendation

| Number | Title |
|---|---|
| Table 4.1 | GPU Memory Tier Summary |
| Table 4.2 | HBM Reference Values by Accelerator |
| Table 4.3 | Memory Bottleneck Diagnostic Signals |
| Table 4.4 | KV Cache Memory Formula Variables |
| Table 4.5 | Memory Optimization Techniques |
| Table 4.6 | Wrong Fix vs Right First Question |

---

# 5. Figure Inventory Updates

```markdown
| Figure | Title | Current Asset | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|---|
| Fig 4.1 | Memory Hierarchy Pyramid | diagrams/diagram_01_memory_hierarchy.html | Ch04 | Yes | No | Export 300-DPI/vector and caption |
| Fig 4.2 | GPU Memory Hierarchy Bandwidth View | diagrams/diagrams_batch1.html#d2 | Ch04 | Yes | No | Export and validate labels |
| Fig 4.3 | HBM3e Die Stacking vs GDDR6X | diagrams/diagrams_batch2.html#d13 | Ch04 | Yes | No | Export and caption |
| Fig 4.4 | Memory-Bound vs Compute-Bound Decision Flow | TBD | Ch04 | No | No | Create SVG |
| Fig 4.5 | KV Cache Memory Growth Curve | TBD | Ch04 | No | No | Create SVG |
| Fig 4.6 | FlashAttention HBM Traffic Reduction | diagrams/diagrams_batch1.html#d5 or TBD | Ch04 | Partial | No | Reuse/adapt and export |
| Fig 4.7 | Memory Optimization Decision Tree | TBD | Ch04 | No | No | Create SVG |
```

---

# 6. Recommended Commit

Save this file as:

```text
publishing/figure_plans/ch04_figure_integration_plan.md
```

Then run:

```powershell
git add publishing\figure_plans\ch04_figure_integration_plan.md
git commit -m "Add Chapter 4 figure integration plan"
git push origin production-v1.0
```
