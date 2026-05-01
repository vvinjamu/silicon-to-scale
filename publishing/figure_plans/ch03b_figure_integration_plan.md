# Chapter 3B Figure Integration Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Chapter 3B — *GPU Architecture Roadmap: NVIDIA and AMD Generations*  
**Target file:** `publishing/figure_plans/ch03b_figure_integration_plan.md`  
**Production status:** Draft integration plan for `production-v1.0`  
**Primary goal:** Turn Ch03B into a roadmap-reading chapter that teaches readers how to evaluate GPU generations safely, without mixing shipped specs, announced roadmaps, product-level numbers, benchmark claims, or marketing language.

---

## 0. Integration Strategy

Chapter 3B should visually answer one question:

> How should a performance architect evaluate GPU and accelerator generations over time?

The chapter should not become a raw hardware-spec catalog. It should teach the reader a repeatable roadmap evaluation method:

1. Evaluate each generation through four lenses:
   - Compute density
   - Memory capacity and bandwidth
   - Interconnect / scale-up domain
   - Software and deployment maturity
2. Separate shipping product specs from announced roadmap claims.
3. Avoid mixing product levels:
   - GPU
   - Board / module
   - Node / system
   - Rack
   - Cluster
4. Compare dense to dense, sparse to sparse, line rate to line rate, and effective bandwidth to effective bandwidth.
5. End with a workload-driven hardware selection decision tree.

Existing relevant diagram assets:

- `diagrams/diagrams_batch2.html#d13` — HBM3e die stacking cross-section vs GDDR6X
- `diagrams/diagrams_batch2.html#d14` — NVLink domain — DGX H100 + NVSwitch
- `diagrams/diagrams_batch2.html#d15` — AMD MI300X die stack
- `diagrams/diagram_01_memory_hierarchy.html` — memory hierarchy cross-reference
- `diagrams/diagrams_batch3.html#d25` — GPU vs CPU architecture comparison, mostly Ch03A cross-reference

Most Ch03B visuals must be created because roadmap and product-level guardrail diagrams are chapter-specific.

---

# 1. Proposed Chapter 3B Visual Sequence

Recommended flow:

| Order | Figure/Table | Purpose |
|---:|---|---|
| 1 | Table 3B.1 — Roadmap Confidence Labels | Teach shipped vs announced vs estimated labels before any product claims |
| 2 | Fig 3B.1 — Four-Lens GPU Generation Evaluation Framework | Establish how to evaluate any GPU generation |
| 3 | Fig 3B.2 — NVIDIA Roadmap Timeline | Show Ampere → Hopper → Blackwell → Rubin |
| 4 | Table 3B.2 — NVIDIA Generation Comparison | Validate and summarize NVIDIA generations |
| 5 | Fig 3B.3 — AMD Roadmap Timeline | Show CDNA 3 → CDNA 3+ → CDNA 4 → MI400 |
| 6 | Table 3B.3 — AMD Generation Comparison | Validate and summarize AMD generations |
| 7 | Fig 3B.4 — HBM Evolution | Explain memory generation trend |
| 8 | Table 3B.4 — Memory Evolution Table | Separate memory trend from product-specific values |
| 9 | Fig 3B.5 — NVLink / Infinity Fabric Evolution | Show interconnect domain evolution |
| 10 | Table 3B.5 — Interconnect Evolution Table | Compare NVLink and Infinity Fabric with directionality caution |
| 11 | Fig 3B.6 — MI300X Die Stack | Explain AMD chiplet + HBM strategy |
| 12 | Fig 3B.7 — GPU vs System vs Rack Numbering Guardrail | Prevent Blackwell/GB200 product-level confusion |
| 13 | Table 3B.6 — Product-Level Numbering Guardrail | Show GPU vs system vs rack examples |
| 14 | Fig 3B.8 — Hardware Selection Decision Tree | Final architecture decision flow |
| 15 | Table 3B.7 — Hardware Selection Matrix | Workload → hardware attribute mapping |
| 16 | Table 3B.8 — Claims Requiring Annual Refresh | Maintenance checklist for future editions |

---

# 2. Detailed Figure and Table Plan

---

## Table 3B.1 — Roadmap Confidence Labels

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact chapter section placement:** Immediately after chapter overview, before any NVIDIA or AMD roadmap discussion.

### Caption

**Table 3B.1 — Confidence Labels for Hardware Roadmap Claims.**  
Roadmap chapters must distinguish shipping product specifications, announced platform claims, derived calculations, estimates, representative guidance, and environment-specific benchmark behavior.

### Proposed table content

| Label | Use in Ch03B | Example |
|---|---|---|
| `[SHIPPED]` | Vendor-published shipping product spec | H200 141 GB HBM3e, 4.8 TB/s |
| `[ANNOUNCED]` | Vendor-announced roadmap or future platform | Rubin or MI400 roadmap direction |
| `[DERIVED FROM SHIPPED]` | Calculation from official shipping spec | Dense BF16 derived from sparse peak |
| `[ESTIMATED]` | Simplified engineering estimate | Memory needed for 70B BF16 weights |
| `[REPRESENTATIVE]` | Workload-dependent architecture guidance | H200 often helps memory-bound inference |
| `[ENV-SPECIFIC]` | Benchmark or measured deployment behavior | MLPerf result, internal cluster throughput |

### Intro paragraph before table

A roadmap chapter is useful only if readers understand the confidence level of each claim. A shipping GPU specification is not the same as an announced roadmap target, and a benchmark result is not the same as a production guarantee.

### Explanation paragraph after table

Use this table as the contract for the chapter. Every hardware number, roadmap claim, benchmark comparison, and architecture recommendation should carry the right confidence label. This protects the book from sounding speculative and helps readers know when to verify details against current vendor documentation.

### Key takeaway box

> **Key Takeaway:** Roadmap claims are not all equal. Before using a hardware number, identify whether it is shipped, announced, derived, estimated, representative, or environment-specific.

### Web-readiness status

**Ready after table is authored.** Needs responsive table styling.

### Print-readiness status

**Low risk.** Compact table should fit well.

### Required fixes before production

- Keep labels identical to the book-wide confidence-label system.
- Use the same definitions in Ch00, Ch01, Ch03A, and Appendix A.
- Add a note: “Current as of 2026 edition; verify before procurement.”

---

## Fig 3B.1 — Four-Lens GPU Generation Evaluation Framework

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch03b_fig_3b_1_four_lens_gpu_generation_framework.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03b_fig_3b_1_four_lens_gpu_generation_framework.png`  
**Exact chapter section placement:** After Table 3B.1 and before NVIDIA roadmap timeline.

### Caption

**Fig 3B.1 — Four-Lens GPU Generation Evaluation Framework.**  
A performance architect evaluates each accelerator generation through compute density, memory architecture, interconnect domain, and software/deployment maturity.

### Intro paragraph before figure

GPU generations should not be evaluated by one headline number. A new generation may improve peak FLOPs, memory capacity, memory bandwidth, precision support, scale-up topology, or software-visible capability. The architect’s job is to identify which dimension actually changes the workload bottleneck.

### Explanation paragraph after figure

The four-lens framework keeps roadmap discussion disciplined. H200 is mainly a memory-capacity and memory-bandwidth story. Blackwell is partly a precision and rack-scale interconnect story. MI300X is strongly a memory-capacity story. Future Rubin or MI400 claims should be treated as roadmap signals until product-level specifications are verified.

### Key takeaway box

> **Key Takeaway:** A GPU generation matters only when its changed resource maps to your workload bottleneck.

### Web-readiness status

**Not ready.** New SVG needed.

### Print-readiness status

**Not ready.** Needs 300-DPI PNG or vector PDF.

### Required fixes before production

- Create a four-quadrant framework:
  - Compute density
  - Memory capacity/bandwidth
  - Interconnect / scale-up domain
  - Software maturity
- Add small examples under each lens.
- Avoid product-specific values in this figure.
- Add alt text.
- Use grayscale-safe colors.

---

## Fig 3B.2 — NVIDIA Roadmap Timeline: Ampere → Hopper → Blackwell → Rubin

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch03b_fig_3b_2_nvidia_roadmap_timeline.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03b_fig_3b_2_nvidia_roadmap_timeline.png`  
**Exact chapter section placement:** Beginning of NVIDIA roadmap section.

### Caption

**Fig 3B.2 — NVIDIA GPU Roadmap Timeline.**  
NVIDIA accelerator generations should be read as a sequence of compute, memory, precision, interconnect, and system-level changes: Ampere, Hopper, Blackwell, and Rubin.

### Intro paragraph before figure

The NVIDIA roadmap is not simply “new GPU faster than old GPU.” Each generation changes a different part of the AI infrastructure stack. Ampere established A100 as a major training platform. Hopper made FP8 and HBM3 central to LLM infrastructure. H200 refreshed Hopper with larger and faster HBM. Blackwell shifts more emphasis toward rack-scale NVLink domains and newer precision formats. Rubin should be treated as roadmap until specific product specs are available.

### Explanation paragraph after figure

The timeline helps the reader avoid comparing the wrong product level. H100 and H200 are GPU-level products. DGX B200 and GB200 NVL72 are system/rack-level products. Rubin is a future roadmap family unless official product-level specifications are available. The figure should visually separate shipped and announced generations.

### Key takeaway box

> **Key Takeaway:** Read NVIDIA generations by product level and confidence label: GPU, system, rack, or roadmap.

### Web-readiness status

**Not ready.** New timeline needed.

### Print-readiness status

**Not ready.** Timeline may need a stacked print layout.

### Required fixes before production

- Use visually distinct badges:
  - `[SHIPPED]`
  - `[ANNOUNCED]`
  - `[ROADMAP SIGNAL]`
- Separate GPU-level and rack-level products.
- Include A100, H100, H200, B200/GB200, Rubin.
- Avoid unverified dates or product availability.
- Add alt text.
- Ensure mobile layout stacks vertically.

---

## Table 3B.2 — NVIDIA Generation Comparison

**Type:** New validation-sensitive table  
**Existing source file:** None  
**Status:** Must be created after technical validation  
**Exact chapter section placement:** Immediately after Fig 3B.2.

### Caption

**Table 3B.2 — NVIDIA Generation Comparison.**  
A roadmap comparison must distinguish GPU-level, system-level, and rack-level products and must label each specification as shipped, announced, derived, or representative.

### Proposed table content

| Generation / Product | Product Level | Main Change | Memory Story | Interconnect Story | Confidence |
|---|---|---|---|---|---|
| A100 | GPU | Ampere Tensor Core platform | 40/80 GB HBM2e depending SKU | NVLink 3 class systems | `[SHIPPED]` |
| H100 | GPU | Hopper, FP8 Transformer Engine, HBM3 | 80 GB HBM3, 3.35 TB/s for SXM5 | NVLink 4, up to 900 GB/s aggregate per GPU | `[SHIPPED]` |
| H200 | GPU | Hopper memory refresh | 141 GB HBM3e, 4.8 TB/s | Hopper platform continuity | `[SHIPPED]` |
| B200 | GPU / module depending context | Blackwell compute and precision evolution | Product-specific HBM3e | NVLink 5 class systems | `[SHIPPED]` or `[ANNOUNCED]` depending source |
| DGX B200 | System | 8-GPU Blackwell system | System-level total HBM | System-level NVLink fabric | `[SHIPPED]` if official product page |
| GB200 NVL72 | Rack-scale platform | Rack-scale NVLink domain | Rack-level HBM | Rack-scale NVLink domain | `[SHIPPED]` if official product page |
| Rubin / Vera Rubin | Roadmap | Future generation | Roadmap-specific | Roadmap-specific | `[ANNOUNCED]` |

### Intro paragraph before table

The NVIDIA generation comparison should not mix per-GPU, per-system, and per-rack values. This table forces each row to declare its product level before listing hardware attributes.

### Explanation paragraph after table

The table is a roadmap reading tool. It shows that H200 should not be described as a completely new architecture in the same way Blackwell is. It also shows that GB200 NVL72 should not be compared directly to a single GPU number.

### Key takeaway box

> **Key Takeaway:** Every NVIDIA roadmap number must answer: per GPU, per system, per rack, or roadmap?

### Web-readiness status

**Ready after table is authored.** Needs responsive table wrapper.

### Print-readiness status

**High risk.** Wide table may need split into two print tables.

### Required fixes before production

- Validate each value in `ch03b_technical_validation.md`.
- Add “current as of” note.
- Consider split:
  - Table 3B.2A — GPU-level products
  - Table 3B.2B — System/rack-level products
- Avoid single “Blackwell number” language.

---

## Fig 3B.3 — AMD Roadmap Timeline: CDNA 3 → CDNA 3+ → CDNA 4 → MI400

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch03b_fig_3b_3_amd_roadmap_timeline.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03b_fig_3b_3_amd_roadmap_timeline.png`  
**Exact chapter section placement:** Beginning of AMD roadmap section.

### Caption

**Fig 3B.3 — AMD Instinct Roadmap Timeline.**  
AMD’s AI accelerator roadmap should be read through memory capacity, HBM bandwidth, chiplet architecture, precision support, and rack-scale platform direction.

### Intro paragraph before figure

The AMD roadmap tells a different but important story. MI300X emphasized very large HBM capacity and bandwidth. MI325X extended the memory story. MI350/MI355 moved the platform toward CDNA 4, larger HBM3e capacity, higher bandwidth, and new low-precision formats. MI400 should be treated as roadmap unless final product specifications are available.

### Explanation paragraph after figure

The timeline should show AMD’s memory-first relevance for LLM serving and long-context workloads. It should also avoid implying software maturity, benchmark parity, or deployment availability without evidence. The roadmap should be treated as a sequence of architectural signals, not a universal NVIDIA-vs-AMD conclusion.

### Key takeaway box

> **Key Takeaway:** AMD roadmap evaluation should focus on memory fit, bandwidth, interconnect, precision support, ROCm/software readiness, and platform availability.

### Web-readiness status

**Not ready.** New timeline needed.

### Print-readiness status

**Not ready.** Timeline may need stacked print version.

### Required fixes before production

- Include MI300X, MI325X, MI350/MI355, MI400.
- Use confidence badges.
- Avoid vendor-comparison claims in the timeline.
- Add alt text.
- Validate generation names and product status before final export.

---

## Table 3B.3 — AMD Generation Comparison

**Type:** New validation-sensitive table  
**Existing source file:** None  
**Status:** Must be created after technical validation  
**Exact chapter section placement:** Immediately after Fig 3B.3.

### Caption

**Table 3B.3 — AMD Instinct Generation Comparison.**  
AMD accelerator generations should be compared by memory capacity, memory bandwidth, precision formats, chiplet architecture, interconnect, software maturity, and product status.

### Proposed table content

| Product | Generation / Family | Main Change | Memory Story | Compute / Precision Story | Confidence |
|---|---|---|---|---|---|
| MI300X | CDNA 3 | Large-HBM chiplet accelerator | 192 GB HBM3, ≈5.3 TB/s | Dense BF16 ≈1,307.4 TFLOPS | `[SHIPPED]` |
| MI325X | CDNA 3+ / refresh | Memory capacity/bandwidth refresh | 256 GB HBM3e, validate bandwidth | Same general generation class, validate values | `[SHIPPED]` if official product page |
| MI350X / MI355X | CDNA 4 / MI350 Series | Larger HBM and newer low-precision support | 288 GB HBM3e, 8 TB/s class bandwidth if validated | MXFP4/MXFP6 or FP4-class claims need validation | `[SHIPPED]` if official product page |
| MI400 | Roadmap / next generation | Future rack-scale/platform direction | TBD until official specs | TBD until official specs | `[ANNOUNCED]` |

### Intro paragraph before table

The AMD comparison should make clear which values are shipping product specs and which are roadmap direction. This prevents the chapter from overclaiming MI400 or underexplaining MI300X/MI325X/MI350 Series.

### Explanation paragraph after table

The table should emphasize AMD’s large-memory story without making universal performance claims. A memory-rich accelerator can be extremely attractive for model-fit and KV-cache-heavy workloads, but production choice also depends on kernels, framework maturity, interconnect, support model, and operational risk.

### Key takeaway box

> **Key Takeaway:** AMD’s roadmap should be evaluated through large HBM capacity, bandwidth, chiplet design, software maturity, and workload fit — not through peak numbers alone.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**High risk.** Wide table may need split.

### Required fixes before production

- Validate MI325X and MI350/MI355 values.
- Avoid using MI400 as if final specs are known.
- Add “current as of” note.
- Keep benchmark comparisons out of this table unless footnoted.

---

## Fig 3B.4 — HBM Evolution: HBM2e → HBM3 → HBM3e → HBM4

**Type:** Existing partial figure plus new roadmap visual  
**Existing source file:** `diagrams/diagrams_batch2.html#d13`  
**Existing diagram title:** HBM3e Die Stacking Cross-Section vs GDDR6X  
**Status:** Existing HBM3e visual can be used, but a roadmap-specific HBM evolution figure/table should also be created  
**Recommended asset path:** `assets/diagrams/svg/ch03b_fig_3b_4_hbm_evolution.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03b_fig_3b_4_hbm_evolution.png`  
**Exact chapter section placement:** In memory evolution section after NVIDIA/AMD product timelines.

### Caption

**Fig 3B.4 — HBM Evolution and AI Accelerator Roadmaps.**  
AI accelerator generations increasingly depend on HBM capacity and bandwidth. HBM evolution changes model-fit, KV-cache capacity, long-context serving, and training memory economics.

### Intro paragraph before figure

The AI accelerator roadmap is also a memory roadmap. Many LLM workloads are constrained not by peak compute, but by whether the model and KV cache fit in HBM and whether decode can stream data fast enough.

### Explanation paragraph after figure

HBM2e, HBM3, HBM3e, and HBM4 should be discussed as technology generations, while product-specific values must remain tied to a GPU or accelerator SKU. H100, H200, B200/GB200, MI300X, MI325X, and MI350 Series use memory differently at GPU, system, and rack levels.

### Key takeaway box

> **Key Takeaway:** For LLM infrastructure, memory generation can matter as much as compute generation.

### Web-readiness status

**Partially ready.** Existing HBM3e visual exists, but roadmap figure should be created.

### Print-readiness status

**Not ready.** Needs dedicated print-safe export.

### Required fixes before production

- Use existing HBM3e diagram as supporting visual.
- Create a separate HBM evolution timeline or ladder.
- Avoid universal bandwidth values for each HBM generation unless sourced.
- Tie capacity/bandwidth values to products in tables, not generic HBM labels.
- Add alt text.

---

## Table 3B.4 — Memory Evolution Table

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact chapter section placement:** Immediately after Fig 3B.4.

### Caption

**Table 3B.4 — Memory Evolution in AI Accelerators.**  
HBM evolution changes AI infrastructure by increasing memory capacity, bandwidth, and packaging density.

### Proposed table content

| Memory Generation | Used In / Associated With | What It Changes | Production Caution |
|---|---|---|---|
| HBM2e | A100-era accelerators | Large memory bandwidth relative to prior GPU generations | Product values vary by SKU |
| HBM3 | H100, MI300X class | Higher bandwidth and capacity for LLM training/inference | Compare product-specific values |
| HBM3e | H200, B200/GB200, MI325X/MI350 class | Larger capacity and higher bandwidth | Do not mix GPU, system, and rack totals |
| HBM4 | Future roadmap | Expected future capacity/bandwidth direction | Use `[ANNOUNCED]` or `[ESTIMATED]` unless final product specs exist |

### Intro paragraph before table

The figure shows the direction of memory evolution. The table turns that direction into production reasoning. Each HBM generation matters because it changes whether workloads fit and how fast memory-bound workloads can run.

### Explanation paragraph after table

The safest approach is to avoid generic HBM claims when making system decisions. Use product-specific HBM capacity and bandwidth values from official specs, then map those values to model size, KV cache, batch size, context length, and cost.

### Key takeaway box

> **Key Takeaway:** HBM generation is a useful signal, but product-specific HBM capacity and bandwidth are what matter for architecture decisions.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Low to medium risk.**

### Required fixes before production

- Validate product examples.
- Add “current as of 2026 edition” note.
- Cross-reference Chapter 4 and Appendix A.
- Keep HBM4 wording roadmap-safe.

---

## Fig 3B.5 — NVLink / Infinity Fabric Evolution

**Type:** New or adapted figure  
**Existing source file:** `diagrams/diagrams_batch2.html#d14` can be used for NVLink domain example  
**Status:** Must create roadmap-specific comparison figure  
**Recommended asset path:** `assets/diagrams/svg/ch03b_fig_3b_5_interconnect_evolution.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03b_fig_3b_5_interconnect_evolution.png`  
**Exact chapter section placement:** In interconnect evolution section after memory evolution.

### Caption

**Fig 3B.5 — Scale-Up Fabric Evolution: NVLink, NVSwitch, and Infinity Fabric.**  
AI accelerator roadmaps increasingly expand the scale-up communication domain, moving from GPU-to-GPU links to system-level and rack-level fabrics.

### Intro paragraph before figure

Large AI systems are not just collections of fast GPUs. They are communication machines. As models grow, the boundary of high-bandwidth communication shifts from one GPU to one node and, increasingly, toward rack-scale domains.

### Explanation paragraph after figure

NVLink/NVSwitch and AMD Infinity Fabric should be discussed by scope and directionality. Is the number per link, per GPU, per system, per rack, per direction, bidirectional aggregate, or effective application bandwidth? Without that clarity, interconnect comparisons become misleading.

### Key takeaway box

> **Key Takeaway:** Interconnect evolution is about expanding the efficient communication domain, not just increasing a bandwidth number.

### Web-readiness status

**Not ready.** New figure needed.

### Print-readiness status

**Not ready.** Needs print-safe export.

### Required fixes before production

- Create visual with communication domains:
  - GPU
  - Node
  - Rack
  - Cluster
- Include NVLink 4, NVLink 5, NVSwitch, Infinity Fabric as concepts.
- Avoid unvalidated bandwidth values in the figure.
- Put numeric values in validation tables.
- Add alt text.
- Cross-reference Chapter 14.

---

## Table 3B.5 — Interconnect Evolution Table

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created after technical validation  
**Exact chapter section placement:** Immediately after Fig 3B.5.

### Caption

**Table 3B.5 — Interconnect Evolution and Directionality Caution.**  
Interconnect values must be compared by scope and directionality: per link, per GPU, per node, per rack, line rate, or effective bandwidth.

### Proposed table content

| Fabric / Generation | Associated Products | Scope | What Changed | Directionality Caution |
|---|---|---|---|---|
| NVLink 3 | A100 systems | Intra-node GPU fabric | Ampere scale-up fabric | State per GPU/aggregate |
| NVLink 4 | H100/H200 systems | Intra-node GPU fabric | Hopper scale-up bandwidth | H100 SXM often listed as 900 GB/s aggregate per GPU |
| NVLink 5 | Blackwell systems | Node/rack-scale fabric depending platform | Higher bandwidth, larger NVLink domains | Product-specific |
| NVSwitch | DGX/HGX/NVL systems | Switched local/rack fabric | Many-GPU communication domain | System/rack specific |
| AMD Infinity Fabric | MI300/MI350 systems | Accelerator fabric / package/platform | GPU-to-GPU and package-level communication | Product/platform specific |
| InfiniBand/Ethernet | Multi-node fabric | Cluster | Scale-out training/serving | Line rate ≠ effective NCCL bandwidth |

### Intro paragraph before table

The interconnect table should prevent a common roadmap mistake: comparing one vendor’s per-GPU aggregate number to another vendor’s per-link or per-system number.

### Explanation paragraph after table

The right comparison starts with scope. A tensor-parallel group cares about the local scale-up fabric. Data parallelism cares about scale-out bandwidth and collective efficiency. Expert parallelism may care about all-to-all patterns and topology. The interconnect cannot be evaluated outside the parallelism strategy.

### Key takeaway box

> **Key Takeaway:** Interconnect numbers are meaningful only after you know the communication pattern and the scope of the bandwidth number.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**High risk.** Wide table may require splitting.

### Required fixes before production

- Validate NVLink 4/5 values in technical validation.
- Validate Infinity Fabric values per product.
- Avoid line-rate/application-bandwidth confusion.
- Cross-reference Ch10 and Ch14.

---

## Fig 3B.6 — AMD MI300X Die Stack

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch2.html#d15`  
**Status:** Exists but must be integrated into Ch03B  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03b_fig_3b_6_mi300x_die_stack.png`  
**Exact chapter section placement:** In AMD CDNA 3 / MI300X section, after explaining MI300X’s memory and chiplet story.

### Caption

**Fig 3B.6 — AMD MI300X Die Stack: Chiplets and HBM.**  
MI300X uses a chiplet-based accelerator architecture with multiple compute dies and large HBM capacity, making it especially relevant for memory-heavy AI workloads.

### Intro paragraph before figure

MI300X is important not only because of its compute number, but because of its packaging and memory story. Large HBM capacity changes model-fit and KV-cache economics for LLM inference.

### Explanation paragraph after figure

The die-stack figure should be used to explain why MI300X is often discussed in the context of large-memory serving. The architectural lesson is that memory capacity, bandwidth, packaging, and software support together determine production usefulness.

### Key takeaway box

> **Key Takeaway:** MI300X is a roadmap milestone because it made large-HBM accelerator design central to AI infrastructure discussions.

### Web-readiness status

**Ready.** Existing diagram is browser-ready.

### Print-readiness status

**Not ready.** Needs 300-DPI/vector export.

### Required fixes before production

- Export print-safe PNG/vector.
- Validate die-count labels before final publication.
- Add alt text.
- Avoid claiming MI300X universally beats other GPUs; keep workload-specific framing.
- Cross-reference Ch03A and Ch04.

---

## Fig 3B.7 — GPU vs Node vs Rack Product-Level Numbering Guardrail

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch03b_fig_3b_7_product_level_guardrail.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03b_fig_3b_7_product_level_guardrail.png`  
**Exact chapter section placement:** In Blackwell/GB200 section before any DGX B200, GB200 Superchip, or GB200 NVL72 numbers.

### Caption

**Fig 3B.7 — GPU vs System vs Rack Product-Level Guardrail.**  
Roadmap comparisons must not mix per-GPU, per-system, per-rack, and cluster-level numbers. Product level must be declared before any hardware value is compared.

### Intro paragraph before figure

Blackwell-era systems make product-level clarity mandatory. A B200 GPU, a DGX B200 system, a GB200 Superchip, and a GB200 NVL72 rack are not the same comparison unit.

### Explanation paragraph after figure

This figure should prevent one of the most common roadmap mistakes: comparing a rack-level HBM capacity or NVLink-domain number to a single-GPU number from a previous generation. Always declare whether a number is per GPU, per module, per node, per rack, or per cluster.

### Key takeaway box

> **Key Takeaway:** Before comparing roadmap numbers, ask: “What is the unit — GPU, module, system, rack, or cluster?”

### Web-readiness status

**Not ready.** New figure needed.

### Print-readiness status

**Not ready.** Needs print-safe export.

### Required fixes before production

- Create hierarchy visual:
  - GPU
  - Board/module
  - Node/system
  - Rack
  - Cluster
- Include examples:
  - B200 GPU
  - DGX B200
  - GB200 Superchip
  - GB200 NVL72
- Avoid putting too many numbers in figure.
- Add alt text.
- Use this figure before Blackwell comparison tables.

---

## Table 3B.6 — Product-Level Numbering Guardrail

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact chapter section placement:** Immediately after Fig 3B.7.

### Caption

**Table 3B.6 — Product-Level Numbering Guardrail.**  
Hardware values must be compared at the same product level. Per-GPU values are not comparable to system-level or rack-level aggregate values.

### Proposed table content

| Level | Example | Valid Comparison | Common Mistake |
|---|---|---|---|
| GPU | H100, H200, B200, MI300X | GPU-to-GPU specs | Comparing one GPU to a full rack |
| Module / Superchip | GB200 Superchip, accelerator module | Module-to-module | Treating module value as per-GPU |
| System / Node | DGX B200, HGX system, 8-GPU server | System-to-system | Comparing system total HBM to per-GPU HBM |
| Rack | GB200 NVL72, rack-scale platforms | Rack-to-rack | Comparing rack NVLink domain to node fabric |
| Cluster | Multi-rack training system | Cluster-to-cluster | Confusing network bisection with local fabric |

### Intro paragraph before table

The figure shows the product-level hierarchy. The table gives a quick check for whether a comparison is valid.

### Explanation paragraph after table

This guardrail should be repeated whenever the chapter discusses Blackwell, GB200, NVL72, MI400, or rack-scale platforms. It is also useful for avoiding misleading procurement or interview explanations.

### Key takeaway box

> **Key Takeaway:** Most roadmap comparison errors happen because the comparison unit changed without the reader noticing.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Medium risk.**

### Required fixes before production

- Keep examples current.
- Use confidence labels for future products.
- Add “current as of 2026 edition” note.
- Cross-reference Appendix A.

---

## Fig 3B.8 — Hardware Selection Decision Tree

**Type:** New synthesis figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch03b_fig_3b_8_hardware_selection_decision_tree.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03b_fig_3b_8_hardware_selection_decision_tree.png`  
**Exact chapter section placement:** Near the end of the chapter before key takeaways and review questions.

### Caption

**Fig 3B.8 — Hardware Selection Decision Tree.**  
A performance architect selects hardware by matching workload bottlenecks to hardware resources: memory capacity, memory bandwidth, compute density, interconnect, precision support, software stack, power, cost, and operational risk.

### Intro paragraph before figure

The purpose of roadmap analysis is not to memorize product names. The purpose is to choose the right system for a workload. This decision tree turns roadmap knowledge into architecture judgment.

### Explanation paragraph after figure

A 70B inference service may prioritize HBM capacity and KV-cache headroom. A large training cluster may prioritize local and scale-out interconnect. A cost-sensitive deployment may prioritize software maturity, availability, and cost per token. Future platforms such as Rubin or MI400 should be treated as planning signals until production details are verified.

### Key takeaway box

> **Key Takeaway:** Hardware selection starts with the workload bottleneck, not the newest product name.

### Web-readiness status

**Not ready.** New figure needed.

### Print-readiness status

**Not ready.** Needs simple layout and print export.

### Required fixes before production

- Decision tree should fit on one page.
- Include yes/no or priority branches:
  - Does the model fit in HBM?
  - Is decode memory-bound?
  - Is training communication-bound?
  - Does workload need rack-scale NVLink?
  - Does software stack support kernels?
  - What is cost/power risk?
- Add alt text.
- Cross-reference Ch03A, Ch04, Ch05, Ch10, Ch14, Ch17.

---

## Table 3B.7 — Hardware Selection Matrix

**Type:** New synthesis table  
**Existing source file:** None  
**Status:** Must be created  
**Exact chapter section placement:** Immediately after Fig 3B.8.

### Caption

**Table 3B.7 — Workload-to-Hardware Selection Matrix.**  
Different AI workloads reward different hardware attributes. The best accelerator is workload-specific.

### Proposed table content

| Workload Scenario | Most Important Hardware Attribute | Roadmap Implication |
|---|---|---|
| 70B inference with long context | HBM capacity, HBM bandwidth, KV-cache efficiency | H200, MI300X/MI325X/MI350-class memory-rich systems may be attractive |
| Multi-GPU tensor-parallel serving | Local scale-up fabric | NVLink/NVSwitch or equivalent high-bandwidth local fabric matters |
| Frontier training | Compute, HBM, scale-up + scale-out fabric, reliability | Hopper/Blackwell-class and rack-scale systems matter |
| Cost-sensitive inference | Software stack, availability, power, quantization support | Peak TFLOPS may matter less than cost per token |
| Long-context workloads | HBM capacity, bandwidth, attention kernels | HBM3e/HBM4 and FlashAttention-style kernels matter |
| MoE training/inference | All-to-all communication and routing balance | Interconnect and scheduler design matter |
| Enterprise deployment | Software maturity, support, observability | Ecosystem and operational maturity matter |

### Intro paragraph before table

The decision tree gives a flow. The matrix gives practical scenario mapping. Use it to avoid universal claims like “GPU X is better than GPU Y.”

### Explanation paragraph after table

The table should be used as architecture guidance, not a benchmark claim. Real decisions still require profiling, cost modeling, framework testing, power/cooling analysis, and support review.

### Key takeaway box

> **Key Takeaway:** The same hardware can be excellent for one workload and inefficient for another.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Medium to high risk.** May need split table.

### Required fixes before production

- Keep wording neutral.
- Do not imply recommendations without validation.
- Label scenario guidance `[REPRESENTATIVE]`.
- Cross-reference Ch05 for TCO.

---

## Table 3B.8 — Claims Requiring Annual Refresh

**Type:** New maintenance table  
**Existing source file:** None  
**Status:** Must be created  
**Exact chapter section placement:** End of chapter, in production source before source notes or appendix reference. Can be visually separated from main reader narrative.

### Caption

**Table 3B.8 — Roadmap Claims Requiring Annual Refresh.**  
Roadmap chapters age quickly. The following claims should be rechecked before every new edition or production release.

### Proposed table content

| Claim Category | Refresh Needed | Source Type |
|---|---|---|
| B200 / GB200 product status | Per edition | NVIDIA official product pages |
| Rubin / Rubin Ultra roadmap | Per edition | NVIDIA roadmap / official announcements |
| MI325X / MI350 / MI355 availability | Per edition | AMD official product pages |
| MI400 / Helios roadmap | Per edition | AMD official roadmap announcements |
| HBM4 timing and product association | Per edition | JEDEC/vendor roadmap |
| NVLink / Infinity Fabric generation values | Per edition | Vendor product docs |
| MLPerf benchmark comparisons | Per release | MLPerf results database |
| Software stack maturity | Per release | CUDA/ROCm/framework docs |
| Pricing / availability / power | Per procurement cycle | Vendor quotes, cloud SKUs, TCO model |

### Intro paragraph before table

This table is a production-maintenance asset. It helps future editions stay accurate and prevents roadmap chapters from becoming stale.

### Explanation paragraph after table

The table can be included in the web version and optionally shortened in the print edition. It signals to readers that roadmap content is intentionally labeled and maintained.

### Key takeaway box

> **Key Takeaway:** Roadmap content must be refreshed. The method is durable; the product details change.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Optional.** Consider moving to appendix or shortened source note in print.

### Required fixes before production

- Decide whether to include in main chapter or appendix.
- Keep it compact.
- Add “Current as of 2026 edition.”
- Link to Appendix A hardware reference tables.

---

# 3. Final Figure Numbering Recommendation

Use the following final figure numbering for Chapter 3B:

| Number | Asset |
|---|---|
| Fig 3B.1 | Four-Lens GPU Generation Evaluation Framework |
| Fig 3B.2 | NVIDIA Roadmap Timeline |
| Fig 3B.3 | AMD Roadmap Timeline |
| Fig 3B.4 | HBM Evolution |
| Fig 3B.5 | NVLink / Infinity Fabric Evolution |
| Fig 3B.6 | AMD MI300X Die Stack |
| Fig 3B.7 | GPU vs System vs Rack Product-Level Guardrail |
| Fig 3B.8 | Hardware Selection Decision Tree |

Optional figure:

| Number | Asset |
|---|---|
| Fig 3B.9 | Roadmap Signal vs Procurement Fact |

Recommendation: do not add Fig 3B.9 unless the chapter feels too text-heavy. The concept can be handled as a callout box.

---

# 4. Final Table Numbering Recommendation

Use the following final table numbering:

| Number | Table |
|---|---|
| Table 3B.1 | Roadmap Confidence Labels |
| Table 3B.2 | NVIDIA Generation Comparison |
| Table 3B.3 | AMD Generation Comparison |
| Table 3B.4 | Memory Evolution |
| Table 3B.5 | Interconnect Evolution |
| Table 3B.6 | Product-Level Numbering Guardrail |
| Table 3B.7 | Hardware Selection Matrix |
| Table 3B.8 | Claims Requiring Annual Refresh |

---

# 5. Required Updates to `publishing/figure_inventory.md`

Add or update these rows:

```markdown
| Figure | Title | Current Asset | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|---|
| Fig 3B.1 | Four-Lens GPU Generation Evaluation Framework | TBD | Ch03B | No | No | Create SVG + print export |
| Fig 3B.2 | NVIDIA Roadmap Timeline | TBD | Ch03B | No | No | Create timeline SVG |
| Fig 3B.3 | AMD Roadmap Timeline | TBD | Ch03B | No | No | Create timeline SVG |
| Fig 3B.4 | HBM Evolution | diagrams/diagrams_batch2.html#d13 + TBD | Ch03B | Partial | No | Create roadmap version and export |
| Fig 3B.5 | NVLink / Infinity Fabric Evolution | TBD / diagrams_batch2.html#d14 reference | Ch03B | No | No | Create scope/domain SVG |
| Fig 3B.6 | AMD MI300X Die Stack | diagrams/diagrams_batch2.html#d15 | Ch03B | Yes | No | Export 300-DPI/vector and caption |
| Fig 3B.7 | Product-Level Numbering Guardrail | TBD | Ch03B | No | No | Create SVG + print export |
| Fig 3B.8 | Hardware Selection Decision Tree | TBD | Ch03B | No | No | Create SVG + print export |
```

Add table tracking if desired:

```markdown
| Table | Title | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|
| Table 3B.1 | Roadmap Confidence Labels | Ch03B | Yes | Ready | Create in source |
| Table 3B.2 | NVIDIA Generation Comparison | Ch03B | Yes | High risk | Validate and split if needed |
| Table 3B.3 | AMD Generation Comparison | Ch03B | Yes | High risk | Validate and split if needed |
| Table 3B.4 | Memory Evolution | Ch03B | Yes | Needs check | Add current-as-of note |
| Table 3B.5 | Interconnect Evolution | Ch03B | Yes | High risk | Validate directionality |
| Table 3B.6 | Product-Level Numbering Guardrail | Ch03B | Yes | Needs check | Create concise version |
| Table 3B.7 | Hardware Selection Matrix | Ch03B | Yes | Medium risk | Keep wording neutral |
| Table 3B.8 | Claims Requiring Annual Refresh | Ch03B | Yes | Optional | Consider appendix |
```

---

# 6. Chapter 3B Visual Production Checklist

## Web Checklist

- [ ] Add Table 3B.1 near the opening.
- [ ] Create Fig 3B.1 four-lens evaluation framework.
- [ ] Create Fig 3B.2 NVIDIA roadmap timeline.
- [ ] Create Fig 3B.3 AMD roadmap timeline.
- [ ] Add Fig 3B.4 HBM evolution visual.
- [ ] Create Fig 3B.5 interconnect evolution visual.
- [ ] Add Fig 3B.6 MI300X die stack from existing diagram.
- [ ] Create Fig 3B.7 product-level guardrail visual.
- [ ] Create Fig 3B.8 hardware selection decision tree.
- [ ] Add responsive table wrappers.
- [ ] Add CSS styling for confidence labels.
- [ ] Add anchors for figures and tables.
- [ ] Add previous link to Ch03A and next link to Ch04.
- [ ] Add “Current as of 2026 edition” callout.
- [ ] Verify mobile layout for timelines.

## Print Checklist

- [ ] Export all created SVG diagrams as 300-DPI PNG or vector PDF.
- [ ] Export MI300X die-stack diagram from existing HTML/SVG source.
- [ ] Test roadmap timelines at 7×10 trim.
- [ ] Split wide tables if needed.
- [ ] Avoid tiny product labels.
- [ ] Keep figure captions with figures.
- [ ] Avoid mixing footnotes into dense tables.
- [ ] Add source refresh note.
- [ ] Ensure grayscale readability.

## Technical Validation Checklist

- [ ] Validate A100 80GB values.
- [ ] Validate H100 SXM5 values.
- [ ] Validate H200 values.
- [ ] Validate B200 values by product.
- [ ] Validate DGX B200 values as system-level.
- [ ] Validate GB200 NVL72 values as rack-level.
- [ ] Validate Rubin roadmap wording.
- [ ] Validate MI300X values.
- [ ] Validate MI325X values.
- [ ] Validate MI350 / MI355 values.
- [ ] Treat MI400 as roadmap unless official specs exist.
- [ ] Validate HBM generation claims.
- [ ] Validate NVLink 4/5 values and directionality.
- [ ] Validate Infinity Fabric values and directionality.
- [ ] Label benchmark/comparative claims `[ENV-SPECIFIC]`.

---

# 7. Recommended Next Commit

After saving this file as:

```text
publishing/figure_plans/ch03b_figure_integration_plan.md
```

Run:

```powershell
git add publishing\figure_plans\ch03b_figure_integration_plan.md
git commit -m "Add Chapter 3B figure integration plan"
git push origin production-v1.0
```

Then update the master figure inventory:

```powershell
git add publishing\figure_inventory.md
git commit -m "Update figure inventory for Chapter 3B"
git push origin production-v1.0
```

---

# 8. Next Production Step After This Plan

The next task should be:

```text
Chapter 3B Technical Validation Plan
```

Recommended file:

```text
publishing/validation/ch03b_technical_validation.md
```

The validation should cover:

1. A100 80GB HBM2e and BF16 values
2. H100 SXM5 HBM, BF16 dense/sparse, NVLink 4
3. H200 HBM3e capacity and bandwidth
4. B200 per-GPU values and product status
5. DGX B200 system-level values
6. GB200 Superchip and GB200 NVL72 rack-level values
7. NVLink 5 directionality and scope
8. Rubin / Vera Rubin roadmap wording
9. MI300X memory, BF16, chiplet details
10. MI325X memory/bandwidth
11. MI350X / MI355X memory, bandwidth, precision format claims
12. MI400 / Helios roadmap wording
13. HBM2e/HBM3/HBM3e/HBM4 evolution claims
14. Infinity Fabric and interconnect values
15. Benchmark comparison claims and confidence labels
