# Chapter 3A Figure Integration Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Chapter 3A — *GPU and Accelerator Architecture Fundamentals*  
**Target file:** `publishing/figure_plans/ch03a_figure_integration_plan.md`  
**Production status:** Draft integration plan for `production-v1.0`  
**Primary goal:** Make GPU architecture understandable as a practical performance system: CPU vs GPU mindset, memory hierarchy, SMs, warps, Tensor Cores, interconnect, form factor, accelerator choice, and spec-sheet reading.

---

## 0. Integration Strategy

Chapter 3A should visually answer one question:

> What does the accelerator physically provide, and how does that shape AI/ML performance?

The chapter should not become a list of hardware specs. It should teach the reader to reason from hardware structure to performance behavior:

1. CPU vs GPU mental model
2. GPU memory hierarchy
3. Streaming Multiprocessor (SM) as the core execution unit
4. SIMT and warp execution
5. CUDA grid/block/warp/thread hierarchy
6. Tensor Cores and matrix-multiply specialization
7. NVLink/NVSwitch and intra-node communication
8. PCIe vs SXM/OAM form-factor tradeoffs
9. Accelerator selection across NVIDIA, AMD, TPU, and Gaudi-style systems
10. GPU spec-sheet decision framework

The repository already contains several useful diagram assets:

- `diagrams/diagrams_batch3.html` — GPU vs CPU, SIMT warp execution, CUDA hierarchy
- `diagrams/diagrams_batch1.html` — GPU memory hierarchy pyramid, H100 SM internal block
- `diagrams/diagram_01_memory_hierarchy.html` — standalone memory hierarchy
- `diagrams/diagrams_batch2.html` — NVLink domain and MI300X die stack

The production task is to integrate these diagrams into the chapter flow, create missing decision visuals, and prepare web/print-safe exports.

---

# 1. Proposed Chapter 3A Visual Sequence

Recommended flow:

| Order | Figure/Table | Purpose |
|---:|---|---|
| 1 | Fig 3A.1 — GPU vs CPU Architecture Comparison | Establish the mental-model shift |
| 2 | Fig 3A.2 — GPU Memory Hierarchy | Show where data lives and why movement dominates |
| 3 | Table 3A.1 — GPU Memory Hierarchy Summary | Convert the visual into practical numbers/concepts |
| 4 | Fig 3A.3 — H100 SM Internal Block Diagram | Explain the SM as the unit of execution |
| 5 | Table 3A.2 — SM Resource Summary | Show what an SM contains and why it matters |
| 6 | Fig 3A.4 — SIMT Warp Execution | Explain lockstep execution, divergence, and occupancy |
| 7 | Fig 3A.5 — CUDA Thread Hierarchy | Connect programming hierarchy to hardware scheduling |
| 8 | Fig 3A.6 — Tensor Core / WGMMA Concept | Explain why AI matrix math is special |
| 9 | Table 3A.3 — Tensor Core Format Support by Generation | Show precision support evolution |
| 10 | Fig 3A.7 — NVLink / NVSwitch Domain | Explain intra-node GPU-to-GPU bandwidth |
| 11 | Table 3A.4 — Interconnect Comparison | Compare PCIe, NVLink, NVSwitch, InfiniBand |
| 12 | Fig 3A.8 — PCIe vs SXM / OAM Form Factor | Show why packaging changes system behavior |
| 13 | Table 3A.5 — Accelerator Selection Matrix | Compare NVIDIA, AMD, TPU, and Gaudi-style choices |
| 14 | Table 3A.6 — GPU Spec Sheet Decision Framework | Turn the chapter into an architecture decision tool |

---

# 2. Detailed Figure and Table Plan

---

## Fig 3A.1 — GPU vs CPU Architecture Comparison

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch3.html`  
**Likely existing diagram title:** GPU vs CPU Architecture Comparison  
**Status:** Exists but must be integrated into Chapter 3A  
**Recommended source mapping:** `diagrams/diagrams_batch3.html#d25` or equivalent anchor if present  
**Recommended web asset:** Existing HTML/SVG diagram  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03a_fig_3a_1_gpu_vs_cpu_architecture.png`  
**Exact section placement:** Immediately after the chapter overview and before the first technical hardware section.

### Caption

**Fig 3A.1 — GPU vs CPU Architecture Comparison.**  
A CPU is optimized for low-latency execution of a smaller number of complex threads. A GPU is optimized for high-throughput execution of many simpler threads. AI workloads benefit from the GPU model when they expose enough parallel matrix and tensor work.

### Intro paragraph before figure

The first mistake many engineers make is treating a GPU as a faster CPU. It is not. CPUs and GPUs make different architectural tradeoffs. CPUs spend more silicon on control, branch prediction, speculation, and low-latency execution. GPUs spend more silicon on parallel execution units and memory bandwidth.

### Explanation paragraph after figure

This difference explains why GPUs are powerful for transformer workloads. Training and prefill contain large GEMMs and attention kernels that expose massive parallelism. But the same difference also explains why poorly shaped kernels, branch divergence, small batches, and memory-bound decode paths can underutilize the GPU.

### Key takeaway box

> **Key Takeaway:** A CPU is a latency machine. A GPU is a throughput machine. AI performance depends on exposing enough parallel work to keep the GPU’s execution units and memory system busy.

### Web-readiness status

**Mostly ready.** Existing diagram is browser-ready in the diagram batch page.

### Print-readiness status

**Not ready.** Needs 300-DPI PNG or vector PDF export and label-size validation.

### Required fixes before production

- Export to `assets/diagrams/png_300dpi/ch03a_fig_3a_1_gpu_vs_cpu_architecture.png`.
- Verify all text labels are readable at 7×10 trim.
- Add alt text: “Diagram comparing CPU architecture optimized for latency with GPU architecture optimized for parallel throughput.”
- Ensure the figure does not imply CPUs are weak or obsolete; frame as different design tradeoffs.
- Place early in the chapter to set the mental model.

---

## Fig 3A.2 — GPU Memory Hierarchy

**Type:** Existing figure  
**Existing source file:** `diagrams/diagram_01_memory_hierarchy.html` and/or `diagrams/diagrams_batch1.html` Fig 02  
**Status:** Exists but must be integrated into Chapter 3A  
**Recommended source mapping:** `diagrams/diagram_01_memory_hierarchy.html` for standalone version; `diagrams/diagrams_batch1.html#d2` if using batch anchor  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03a_fig_3a_2_gpu_memory_hierarchy.png`  
**Exact section placement:** In the memory hierarchy section, before the memory hierarchy table.

### Caption

**Fig 3A.2 — GPU Memory Hierarchy.**  
GPU performance depends on how often data can be reused in registers, shared memory/L1, and L2 before falling back to HBM or host memory. The farther data travels, the more expensive it becomes in latency, bandwidth pressure, and energy.

### Intro paragraph before figure

Most GPU performance problems are data movement problems. Tensor Cores can execute enormous amounts of math, but they are only useful if data arrives in the right shape at the right time. The memory hierarchy explains why tiling, fusion, coalescing, caching, and KV-cache layout matter.

### Explanation paragraph after figure

Registers and shared memory are close to the math but limited in capacity. HBM provides high bandwidth but is much slower and more energy-expensive than on-chip storage. PCIe and host memory are farther away still. A strong GPU performance engineer tries to maximize reuse before data leaves the closest usable memory tier.

### Key takeaway box

> **Key Takeaway:** GPU optimization is often the art of moving data fewer times and reusing it closer to the compute units.

### Web-readiness status

**Ready.** Existing standalone HTML diagram is browser-ready.

### Print-readiness status

**Not ready.** Needs 300-DPI PNG or vector PDF export.

### Required fixes before production

- Export print-safe version.
- Confirm memory-capacity and bandwidth values are validated before final publication.
- Use large text for labels such as registers, shared memory, L2, HBM, PCIe, CPU DRAM.
- Add grayscale-safe styling.
- Add alt text.
- Cross-reference Chapter 4 for the deeper memory hierarchy and HBM chapter.

---

## Table 3A.1 — GPU Memory Hierarchy Summary

**Type:** New cleaned table  
**Existing source file:** Concept exists in current chapter, but should be rebuilt as a production table  
**Status:** Must be created  
**Exact section placement:** Immediately after Fig 3A.2.

### Caption

**Table 3A.1 — GPU Memory Hierarchy Summary.**  
Each memory tier trades capacity, bandwidth, latency, and programmability. Understanding these tradeoffs is the foundation of GPU performance engineering.

### Proposed table content

| Tier | Scope | Capacity | Performance Role | Common Optimization |
|---|---|---:|---|---|
| Registers | Per thread | Tiny | Fastest operand storage | Keep hot values in registers |
| Shared memory / L1 | Per SM | Small | Explicit tile reuse and low-latency data sharing | Tiling, bank-conflict avoidance |
| L2 cache | Whole GPU | Medium | Cross-SM data reuse and HBM traffic reduction | Locality, reuse, streaming behavior |
| HBM | Whole GPU | Large | Main high-bandwidth model/activation/KV storage | Coalescing, fusion, quantization, cache layout |
| PCIe / Host memory | CPU-GPU path | Very large | Data staging, offload, slow fallback | Avoid in critical path, prefetch, pin memory |
| Network memory path | Multi-node | Remote | Distributed training/serving communication | Overlap, topology-aware placement |

### Intro paragraph before table

The figure gives the visual hierarchy. The table converts it into an engineering checklist: where does the data live, who can access it, and what optimization moves it closer to the math?

### Explanation paragraph after table

The same transformer operation can behave very differently depending on which memory tier it stresses. A GEMM with good tile reuse can be compute-bound. A normalization kernel that streams through HBM with little reuse can be memory-bound. A distributed training step can become network-bound even when local HBM behavior is excellent.

### Key takeaway box

> **Key Takeaway:** The right optimization depends on which memory tier is the bottleneck. Do not treat all data movement as the same.

### Web-readiness status

**Ready after table is authored.** Needs responsive table wrapper.

### Print-readiness status

**Medium risk.** Table should fit if text is concise.

### Required fixes before production

- Keep table concise.
- Validate H100/H200/MI300X capacity and bandwidth values separately in the technical validation file.
- Avoid overloading this table with too many numeric specs.
- Cross-reference detailed HBM tables in Chapter 4 and Appendix A.

---

## Fig 3A.3 — H100 SM Internal Block Diagram

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch1.html`  
**Likely existing diagram title:** H100 SM Internal Block Diagram — Streaming Multiprocessor  
**Status:** Exists but must be integrated into Chapter 3A  
**Recommended source mapping:** `diagrams/diagrams_batch1.html#d3` or equivalent anchor if present  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03a_fig_3a_3_h100_sm_internal_block.png`  
**Exact section placement:** In the Streaming Multiprocessor section, after introducing SMs and before discussing warps/Tensor Cores.

### Caption

**Fig 3A.3 — H100 Streaming Multiprocessor Internal Block Diagram.**  
The SM is the fundamental execution unit of an NVIDIA GPU. It contains warp schedulers, CUDA cores, Tensor Cores, load/store units, registers, and shared memory/L1 resources.

### Intro paragraph before figure

The SM is where GPU work actually executes. When a kernel launches, blocks are scheduled onto SMs, warps are issued by warp schedulers, and instructions execute on functional units such as CUDA cores, Tensor Cores, and load/store pipelines.

### Explanation paragraph after figure

This diagram explains why GPU performance is multi-dimensional. A kernel can be limited by Tensor Core utilization, load/store throughput, register pressure, shared-memory usage, occupancy, instruction scheduling, or memory stalls. The SM is the bridge between code-level choices and hardware-level execution.

### Key takeaway box

> **Key Takeaway:** The SM is the unit where GPU performance becomes real. Occupancy, warp scheduling, Tensor Core use, memory stalls, and register pressure all meet inside the SM.

### Web-readiness status

**Ready.** Existing diagram is available in the batch page.

### Print-readiness status

**Not ready.** Needs print export and readability test.

### Required fixes before production

- Export to 300-DPI PNG/vector.
- Validate all H100-specific values such as SM count, schedulers, Tensor Cores, L2, shared memory/L1, and warp limits.
- Avoid excessive detail in the main figure if print readability is poor.
- Add alt text.
- Label H100-specific details as `[SHIPPED]` after validation.

---

## Table 3A.2 — SM Resource Summary

**Type:** New table  
**Existing source file:** Concept exists in chapter; should be rebuilt cleanly  
**Status:** Must be created  
**Exact section placement:** Immediately after Fig 3A.3.

### Caption

**Table 3A.2 — Streaming Multiprocessor Resource Summary.**  
SM resources determine how many blocks and warps can run concurrently and what hardware units a kernel can use.

### Proposed table content

| Resource | What It Does | Why It Matters |
|---|---|---|
| Warp schedulers | Issue instructions for resident warps | Determines ability to hide latency |
| CUDA cores | Execute scalar/vector FP32/INT-style instructions | Handles non-Tensor-Core math and control-heavy code |
| Tensor Cores | Execute matrix-multiply-accumulate operations | Main source of AI GEMM throughput |
| Load/store units | Move data between memory hierarchy and registers | Critical for memory-bound kernels |
| Registers | Fast per-thread storage | Too much register use can reduce occupancy |
| Shared memory / L1 | On-chip data sharing and tile reuse | Enables high-performance tiling |
| L2 cache | Shared cache across SMs | Reduces HBM traffic |
| Occupancy slots | Resident warps/blocks | Helps hide latency but is not the same as utilization |

### Intro paragraph before table

The SM diagram shows the hardware layout. The table converts that layout into the questions a performance engineer asks while reading profiler output.

### Explanation paragraph after table

When a kernel is slow, the SM resource summary helps route the investigation. High register usage may limit occupancy. Poor memory coalescing may stress load/store units. A GEMM that does not use Tensor Cores may leave most AI throughput unused. A memory-bound kernel may not improve even if occupancy is high.

### Key takeaway box

> **Key Takeaway:** SM performance is not one number. It is the interaction of schedulers, Tensor Cores, memory pipelines, registers, shared memory, and occupancy.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Low to medium risk.** Table is compact.

### Required fixes before production

- Keep hardware values out of this table unless validated.
- Use this table to explain concepts; place numeric H100 details in validation/source-note tables.
- Cross-reference Chapter 7 for kernel optimization.

---

## Fig 3A.4 — SIMT Warp Execution: Divergence and Occupancy

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch3.html`  
**Likely existing diagram title:** SIMT Warp Execution — Divergence & Occupancy  
**Status:** Exists but must be integrated into Chapter 3A  
**Recommended source mapping:** `diagrams/diagrams_batch3.html#d26` or equivalent anchor if present  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03a_fig_3a_4_simt_warp_execution.png`  
**Exact section placement:** In the SIMT / warp execution section, before explaining divergence and occupancy.

### Caption

**Fig 3A.4 — SIMT Warp Execution: Divergence and Occupancy.**  
A warp executes multiple threads in lockstep. When threads take different control-flow paths, execution serializes across branches, reducing effective throughput.

### Intro paragraph before figure

GPUs execute groups of threads called warps. On NVIDIA GPUs, a warp contains 32 threads. The SIMT model lets the programmer write scalar-looking code while the hardware issues instructions across a warp of lanes.

### Explanation paragraph after figure

Warp execution is efficient when threads follow the same path and access memory coherently. Divergence causes the warp to execute multiple paths serially. Occupancy measures how many warps are resident, but high occupancy does not guarantee high useful throughput. The real question is whether the workload is limited by compute, memory, latency hiding, or instruction efficiency.

### Key takeaway box

> **Key Takeaway:** Warp divergence wastes parallel lanes. Occupancy helps hide latency, but it is not the same as useful hardware utilization.

### Web-readiness status

**Ready.** Existing diagram is browser-ready.

### Print-readiness status

**Not ready.** Needs print export.

### Required fixes before production

- Export to 300-DPI PNG/vector.
- Add alt text.
- Ensure branch/divergence labels are readable in print.
- Add a callout: “Occupancy is not utilization.”
- Cross-reference Chapter 7 for branch divergence and kernel profiling.

---

## Fig 3A.5 — CUDA Thread Hierarchy

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch3.html`  
**Likely existing diagram title:** CUDA Thread Hierarchy — Grid → Block → Warp  
**Status:** Exists but must be integrated into Chapter 3A  
**Recommended source mapping:** `diagrams/diagrams_batch3.html#d27` or equivalent anchor if present  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03a_fig_3a_5_cuda_thread_hierarchy.png`  
**Exact section placement:** After Fig 3A.4, once the chapter explains warp execution.

### Caption

**Fig 3A.5 — CUDA Thread Hierarchy: Grid, Block, Warp, Thread.**  
CUDA exposes a hierarchy of grids, blocks, warps, and threads. Blocks are scheduled onto SMs, and warps are the basic scheduling unit inside the SM.

### Intro paragraph before figure

To connect code to hardware, the reader needs the CUDA execution hierarchy. A kernel launch creates a grid. The grid contains blocks. Blocks contain threads. Threads are executed in warps. Blocks are assigned to SMs based on available resources.

### Explanation paragraph after figure

This hierarchy explains why block size, shared-memory use, register count, and occupancy matter. A kernel with too many registers or too much shared memory per block may reduce the number of blocks that fit on an SM. A kernel with poor thread mapping may produce uncoalesced memory access or divergence.

### Key takeaway box

> **Key Takeaway:** CUDA programming choices become hardware scheduling choices. Grid, block, warp, and thread layout affect occupancy, memory access, and SM utilization.

### Web-readiness status

**Ready.** Existing diagram is browser-ready.

### Print-readiness status

**Not ready.** Needs print export.

### Required fixes before production

- Export to 300-DPI PNG/vector.
- Add alt text.
- Confirm that the diagram does not overload beginners with too many details.
- Cross-reference Chapter 7 for kernel-level optimization.

---

## Fig 3A.6 — Tensor Core / WGMMA Concept

**Type:** New or expanded figure  
**Existing source file:** None as a dedicated Tensor Core figure; H100 SM diagram shows Tensor Core placement  
**Status:** Must be created or derived from SM diagram  
**Recommended asset path:** `assets/diagrams/svg/ch03a_fig_3a_6_tensor_core_wgmma_concept.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03a_fig_3a_6_tensor_core_wgmma_concept.png`  
**Exact section placement:** In the Tensor Core section, after introducing Tensor Cores and before precision-format tables.

### Caption

**Fig 3A.6 — Tensor Core / WGMMA Conceptual Data Path.**  
Tensor Cores accelerate matrix multiply-accumulate operations by operating on tiles of input matrices. High AI throughput depends on using supported datatypes, aligned dimensions, efficient tiling, and kernel paths that dispatch Tensor Core instructions.

### Intro paragraph before figure

Tensor Cores are the hardware reason modern GPUs can deliver enormous AI throughput. But peak Tensor Core throughput is only available when the workload uses compatible datatypes, shapes, layouts, and kernels.

### Explanation paragraph after figure

A GEMM that maps cleanly to Tensor Core tiles can approach high throughput. A GEMM with poor alignment, unsupported dtype, excessive layout conversion, or a fallback kernel may use much less of the theoretical peak. Hopper and later GPUs also introduce more advanced matrix instructions and scheduling patterns such as WGMMA, which are important for high-end kernels but should be introduced conceptually here.

### Key takeaway box

> **Key Takeaway:** Tensor Core peak TFLOPS is not automatic. The kernel must use the right datatype, tile shape, alignment, memory layout, and instruction path.

### Web-readiness status

**Not ready.** New figure needed.

### Print-readiness status

**Not ready.** Needs SVG and 300-DPI/vector export.

### Required fixes before production

- Create a simple conceptual figure showing:
  - Matrix tile A
  - Matrix tile B
  - Accumulator tile C
  - Tensor Core / MMA unit
  - Datatype labels: FP16, BF16, TF32, FP8, INT8 as applicable
- Avoid overloading with exact instruction names unless validated.
- Add a note that WGMMA details are architecture-specific.
- Add alt text.
- Validate Tensor Core format support in technical validation.

---

## Table 3A.3 — Tensor Core Format Support by Generation

**Type:** New table  
**Existing source file:** Concept exists in chapter; must be validated  
**Status:** Must be created after technical validation  
**Exact section placement:** Immediately after Fig 3A.6 or within the Tensor Core section.

### Caption

**Table 3A.3 — Tensor Core Datatype Support by GPU Generation.**  
Tensor Core capability evolves across GPU generations. Supported datatypes determine which model precisions and quantization strategies can use the fastest matrix paths.

### Proposed table content

| Generation | Example GPUs | Representative Tensor Core Formats | Production Note |
|---|---|---|---|
| Volta | V100 | FP16 | First major Tensor Core generation |
| Ampere | A100 | TF32, FP16, BF16, INT8, sparsity paths | Important for mixed precision training |
| Hopper | H100/H200 | FP8, BF16, FP16, TF32, INT8, Transformer Engine | Key generation for FP8 LLM training/inference |
| Blackwell | B100/B200/GB200 | FP4/FP6/FP8 and newer Tensor Core paths as applicable | Label carefully as `[SHIPPED]` or `[ANNOUNCED]` |

### Intro paragraph before table

AI performance depends not only on peak TFLOPS but also on which numerical formats the hardware accelerates. A model using BF16, FP8, INT8, or FP4 must map to supported high-throughput paths.

### Explanation paragraph after table

This table is not a replacement for vendor documentation. It is a reader guide. The practical lesson is that precision choice, quantization strategy, and hardware generation must be evaluated together.

### Key takeaway box

> **Key Takeaway:** Precision is a hardware decision as much as a model decision. If the datatype does not map to fast Tensor Core paths, theoretical model efficiency may not become real system throughput.

### Web-readiness status

**Ready after validation.**

### Print-readiness status

**Medium risk.** Keep table concise.

### Required fixes before production

- Validate each format against official NVIDIA documentation.
- Mark future/roadmap values as `[ANNOUNCED]` if not broadly shipping.
- Avoid implying all formats have the same speed or availability across SKUs.
- Cross-reference Chapter 8 on quantization.

---

## Fig 3A.7 — NVLink / NVSwitch Domain

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch2.html`  
**Likely existing diagram title:** NVLink Domain — DGX H100 + NVSwitch  
**Status:** Exists but must be integrated into Chapter 3A  
**Recommended source mapping:** `diagrams/diagrams_batch2.html#d14` or equivalent anchor if present  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03a_fig_3a_7_nvlink_nvswitch_domain.png`  
**Exact section placement:** In the NVLink/NVSwitch section, before topology guidance and before PCIe vs SXM/OAM discussion.

### Caption

**Fig 3A.7 — NVLink / NVSwitch Domain.**  
High-bandwidth intra-node GPU-to-GPU fabric enables fast tensor-parallel and collective communication. The NVLink/NVSwitch domain is a key boundary for performance-sensitive multi-GPU workloads.

### Intro paragraph before figure

Single-GPU performance is not enough for large models. When a model is partitioned across GPUs, the interconnect becomes part of the model execution path. Tensor parallelism, pipeline parallelism, KV movement, and collective operations all depend on GPU-to-GPU bandwidth and latency.

### Explanation paragraph after figure

NVLink and NVSwitch provide much higher GPU-to-GPU bandwidth than PCIe-only paths. This is why HGX/SXM-class systems are often preferred for large multi-GPU training and tensor-parallel serving. However, topology guidance should remain workload-specific. Not every workload needs the highest-bandwidth fabric.

### Key takeaway box

> **Key Takeaway:** Multi-GPU AI performance depends on topology. A model split across GPUs is only as fast as the compute, memory, and communication schedule together.

### Web-readiness status

**Ready.** Existing diagram is browser-ready.

### Print-readiness status

**Not ready.** Needs 300-DPI/vector export.

### Required fixes before production

- Export to print-safe asset.
- Validate NVLink/NVSwitch bandwidth numbers and directionality.
- Avoid absolute language such as “never” unless clearly constrained.
- Add alt text.
- Cross-reference Chapter 10 and Chapter 14.

---

## Table 3A.4 — Interconnect Comparison

**Type:** New table  
**Existing source file:** Concept exists in chapter but needs validation and directionality clarity  
**Status:** Must be created  
**Exact section placement:** After Fig 3A.7 and before PCIe vs SXM/OAM comparison.

### Caption

**Table 3A.4 — GPU Interconnect Comparison.**  
Different interconnects serve different roles: local CPU-GPU attachment, intra-node GPU communication, and inter-node communication.

### Proposed table content

| Interconnect | Typical Scope | What It Connects | Production Use | Watchouts |
|---|---|---|---|---|
| PCIe | Host/device and sometimes GPU-GPU path | CPU ↔ GPU, GPU ↔ NIC, sometimes GPU ↔ GPU | General attachment, lower-cost systems, development | Lower bandwidth than NVLink; topology matters |
| NVLink | Intra-node GPU fabric | GPU ↔ GPU | Tensor parallelism, fast collectives, large model serving/training | SKU and generation specific |
| NVSwitch | Intra-node or rack-scale switch fabric | Many GPUs through switched NVLink | All-to-all GPU communication within a domain | System design and generation specific |
| InfiniBand / Ethernet fabric | Inter-node | GPU nodes ↔ GPU nodes | Distributed training, multi-node serving | Latency, congestion, collectives, overlap |
| GPUDirect RDMA | GPU ↔ NIC path | GPU memory ↔ network | Reduces CPU staging overhead | Requires platform and driver support |

### Intro paragraph before table

A GPU system is not only a GPU. It is a communication system. The link used for CPU attachment, local GPU-to-GPU communication, and node-to-node traffic can determine whether a distributed workload scales.

### Explanation paragraph after table

The important distinction is scope. PCIe is not the same role as NVLink. NVLink is not the same role as InfiniBand. A strong architecture decision maps each workload communication pattern to the correct interconnect tier.

### Key takeaway box

> **Key Takeaway:** Always ask what the link connects and whether that link sits on the critical path of the workload.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Medium to high risk.** Table may be wide; consider splitting into local and network interconnects.

### Required fixes before production

- Validate bandwidth values in a separate technical validation table.
- Clearly state directionality: per direction, bidirectional, aggregate, or theoretical peak.
- Avoid mixing protocol line rate and application-level bandwidth.
- Cross-reference Chapter 14 for deep networking treatment.

---

## Fig 3A.8 — PCIe vs SXM / OAM Form Factor Comparison

**Type:** New figure  
**Existing source file:** None dedicated  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch03a_fig_3a_8_pcie_vs_sxm_oam.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch03a_fig_3a_8_pcie_vs_sxm_oam.png`  
**Exact section placement:** In the form-factor section, after explaining PCIe, SXM/HGX, and AMD OAM-style accelerator modules.

### Caption

**Fig 3A.8 — PCIe vs SXM / OAM Form Factors.**  
Form factor affects power delivery, cooling, memory bandwidth options, and GPU-to-GPU interconnect topology. PCIe cards are flexible and broadly deployable; SXM/HGX and OAM-style systems are optimized for high-bandwidth multi-GPU configurations.

### Intro paragraph before figure

GPU form factor is not just packaging. It determines power envelope, cooling approach, physical density, and the available communication fabric. This is why the same GPU family may behave differently in PCIe and SXM/HGX-class systems.

### Explanation paragraph after figure

PCIe systems can be excellent for development, smaller inference, single-GPU serving, or cost-sensitive workloads. SXM/HGX-class and OAM-style systems are usually preferred when the workload requires high-bandwidth GPU-to-GPU communication, high power delivery, and dense multi-GPU topology.

### Key takeaway box

> **Key Takeaway:** Form factor is an architecture choice. It changes power, cooling, topology, and the communication budget available to the model.

### Web-readiness status

**Not ready.** New figure needed.

### Print-readiness status

**Not ready.** Needs SVG and print export.

### Required fixes before production

- Create side-by-side visual:
  - PCIe card: CPU ↔ PCIe switch ↔ GPU/NIC
  - SXM/HGX or OAM baseboard: multiple GPUs + high-bandwidth fabric
- Avoid saying one form factor is universally better.
- Add alt text.
- Validate all bandwidth/power claims separately.
- Cross-reference Chapter 5 for power/cooling and Chapter 14 for networking.

---

## Table 3A.5 — Accelerator Selection Matrix

**Type:** New table  
**Existing source file:** Concept exists, but needs neutral production framing  
**Status:** Must be created  
**Exact section placement:** In the accelerator comparison section, after NVIDIA/AMD/TPU/Gaudi concepts are introduced.

### Caption

**Table 3A.5 — Accelerator Selection Matrix.**  
Accelerator choice should be driven by workload shape, memory requirements, interconnect needs, software maturity, operational constraints, and total cost — not peak TFLOPS alone.

### Proposed table content

| Accelerator Family | Common Strengths | Watchouts | Best-Fit Questions |
|---|---|---|---|
| NVIDIA GPU | Mature CUDA ecosystem, Tensor Cores, NVLink/NVSwitch, broad framework support | Cost, supply, power, vendor lock-in | Do we need strongest CUDA/software support and NVLink topology? |
| AMD CDNA / MI300X | Large HBM capacity, strong BF16/HBM specs, ROCm/HIP ecosystem, OAM/Infinity Fabric | Software maturity varies by workload and framework | Does memory capacity/bandwidth matter more than ecosystem friction? |
| TPU | Cloud-integrated systolic-array model, large-scale training/inference focus | Cloud/provider-specific stack and programming model | Are we aligned with the TPU software and cloud ecosystem? |
| Intel Gaudi-style accelerator | Ethernet-oriented accelerator fabric, cost-sensitive training/inference positioning | Framework/kernel maturity and ecosystem coverage | Does the workload fit the stack and network approach? |

### Intro paragraph before table

The goal of accelerator comparison is not to declare a universal winner. The correct accelerator depends on the workload and the organization. A GPU that is ideal for one model or deployment may be a poor fit for another.

### Explanation paragraph after table

A principal architect evaluates accelerator choice across multiple dimensions: model fit in memory, arithmetic intensity, network needs, software stack, team expertise, cost, power, supply, and risk. Peak TFLOPS is only one input.

### Key takeaway box

> **Key Takeaway:** Choose accelerators by workload and system constraints. Peak TFLOPS alone is not an architecture decision.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Medium risk.** Keep concise to avoid wide table overflow.

### Required fixes before production

- Use neutral wording.
- Validate all hardware specs in Chapter 3A technical validation.
- Avoid vendor ranking unless tied to a specific workload and measurement.
- Cross-reference Ch03B for roadmap and Appendix A for hardware tables.

---

## Table 3A.6 — GPU Spec Sheet Decision Framework

**Type:** New synthesis table / checklist  
**Existing source file:** Concept exists in chapter; should be polished  
**Status:** Must be created  
**Exact section placement:** Near the end of the chapter before key takeaways and review questions.

### Caption

**Table 3A.6 — How to Read a GPU Spec Sheet Like a Performance Architect.**  
A spec sheet becomes useful only when interpreted through workload bottlenecks: compute, memory capacity, memory bandwidth, interconnect, software support, power, and cost.

### Proposed table content

| Step | Question | Why It Matters |
|---:|---|---|
| 1 | Does the model and KV cache fit in HBM? | Capacity determines whether sharding, quantization, or paging is required |
| 2 | Is the workload compute-bound or memory-bound? | Determines whether TFLOPS or bandwidth matters more |
| 3 | What precision will the workload use? | Must map to supported fast Tensor Core paths |
| 4 | Does the workload require tensor parallelism? | Determines need for high-bandwidth GPU-to-GPU links |
| 5 | What is the local topology? | NVLink/NVSwitch/PCIe affect communication bottlenecks |
| 6 | What is the node-to-node fabric? | Distributed training and large-scale serving depend on network design |
| 7 | Does the software stack support the workload well? | Kernels, compilers, serving frameworks, and drivers matter |
| 8 | What is the power/cooling/TCO impact? | Sustained performance depends on operating conditions and cost |
| 9 | What telemetry will prove success? | Choose metrics before deployment |
| 10 | What is the fallback plan? | Hardware choice affects operational risk |

### Intro paragraph before table

The chapter should end by teaching readers how to make hardware decisions. A spec sheet is not a shopping list; it is a set of constraints and ceilings that must be mapped to workload behavior.

### Explanation paragraph after table

The framework turns hardware comparison into an engineering process. A memory-bound decode workload may care more about HBM capacity and bandwidth than peak dense TFLOPS. A tensor-parallel training workload may care more about NVLink topology and NCCL performance. A production service may care most about cost per token and failure behavior.

### Key takeaway box

> **Key Takeaway:** A GPU spec sheet only becomes meaningful after you know the workload regime and the system bottleneck.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Medium risk.** Table is long but can be split if needed.

### Required fixes before production

- Keep questions short.
- Add cross-references to Ch01, Ch04, Ch05, Ch10, Ch14, and Ch17.
- Use this as the final synthesis table for the chapter.
- Consider turning into a printable one-page checklist later.

---

# 3. Final Figure Numbering Recommendation

Use the following final figure numbering for Chapter 3A:

| Number | Asset |
|---|---|
| Fig 3A.1 | GPU vs CPU Architecture Comparison |
| Fig 3A.2 | GPU Memory Hierarchy |
| Fig 3A.3 | H100 SM Internal Block Diagram |
| Fig 3A.4 | SIMT Warp Execution: Divergence and Occupancy |
| Fig 3A.5 | CUDA Thread Hierarchy |
| Fig 3A.6 | Tensor Core / WGMMA Concept |
| Fig 3A.7 | NVLink / NVSwitch Domain |
| Fig 3A.8 | PCIe vs SXM / OAM Form Factor Comparison |

Optional figure:

| Number | Asset |
|---|---|
| Fig 3A.9 | AMD MI300X Die Stack / OAM Architecture Cross-Reference |

Recommendation: use the MI300X die stack as a cross-reference in the accelerator comparison section, but do not overload Chapter 3A if Ch03B will cover hardware roadmap and accelerator families more deeply.

---

# 4. Final Table Numbering Recommendation

Use the following table numbering:

| Number | Table |
|---|---|
| Table 3A.1 | GPU Memory Hierarchy Summary |
| Table 3A.2 | Streaming Multiprocessor Resource Summary |
| Table 3A.3 | Tensor Core Datatype Support by GPU Generation |
| Table 3A.4 | GPU Interconnect Comparison |
| Table 3A.5 | Accelerator Selection Matrix |
| Table 3A.6 | GPU Spec Sheet Decision Framework |

---

# 5. Required Updates to `publishing/figure_inventory.md`

Add or update these rows:

```markdown
| Figure | Title | Current Asset | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|---|
| Fig 3A.1 | GPU vs CPU Architecture Comparison | diagrams/diagrams_batch3.html#d25 | Ch03A | Yes | No | Export 300-DPI/vector and caption |
| Fig 3A.2 | GPU Memory Hierarchy | diagrams/diagram_01_memory_hierarchy.html / diagrams_batch1.html#d2 | Ch03A | Yes | No | Export and validate values |
| Fig 3A.3 | H100 SM Internal Block Diagram | diagrams/diagrams_batch1.html#d3 | Ch03A | Yes | No | Export and validate H100-specific labels |
| Fig 3A.4 | SIMT Warp Execution | diagrams/diagrams_batch3.html#d26 | Ch03A | Yes | No | Export and caption |
| Fig 3A.5 | CUDA Thread Hierarchy | diagrams/diagrams_batch3.html#d27 | Ch03A | Yes | No | Export and caption |
| Fig 3A.6 | Tensor Core / WGMMA Concept | TBD | Ch03A | No | No | Create SVG + print export |
| Fig 3A.7 | NVLink / NVSwitch Domain | diagrams/diagrams_batch2.html#d14 | Ch03A | Yes | No | Export and validate bandwidth directionality |
| Fig 3A.8 | PCIe vs SXM / OAM Form Factor Comparison | TBD | Ch03A | No | No | Create SVG + print export |
```

Add table tracking if desired:

```markdown
| Table | Title | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|
| Table 3A.1 | GPU Memory Hierarchy Summary | Ch03A | Yes | Needs check | Create concise table |
| Table 3A.2 | SM Resource Summary | Ch03A | Yes | Needs check | Create concept table |
| Table 3A.3 | Tensor Core Format Support by Generation | Ch03A | Yes | Needs validation | Validate against official sources |
| Table 3A.4 | Interconnect Comparison | Ch03A | Yes | High risk | Validate directionality and split if needed |
| Table 3A.5 | Accelerator Selection Matrix | Ch03A | Yes | Needs check | Neutral wording |
| Table 3A.6 | GPU Spec Sheet Decision Framework | Ch03A | Yes | Needs check | Use as synthesis checklist |
```

---

# 6. Chapter 3A Visual Production Checklist

## Web Checklist

- [ ] Embed Fig 3A.1 near the opening.
- [ ] Embed Fig 3A.2 in the memory hierarchy section.
- [ ] Embed Fig 3A.3 in the SM section.
- [ ] Embed Fig 3A.4 and Fig 3A.5 in SIMT/CUDA hierarchy section.
- [ ] Create and embed Fig 3A.6 Tensor Core concept.
- [ ] Embed Fig 3A.7 NVLink/NVSwitch domain.
- [ ] Create and embed Fig 3A.8 PCIe vs SXM/OAM form factor comparison.
- [ ] Add all six tables with responsive horizontal scrolling.
- [ ] Add alt text for every figure.
- [ ] Add anchors for figures and tables.
- [ ] Add cross-links to Ch01, Ch04, Ch05, Ch07, Ch10, Ch14, and Ch17.
- [ ] Verify mobile readability.

## Print Checklist

- [ ] Export all existing HTML/SVG diagrams as 300-DPI PNG or vector PDF.
- [ ] Validate all labels at 7×10 trim.
- [ ] Keep figures with captions.
- [ ] Prevent figure/caption separation across pages.
- [ ] Split wide interconnect and accelerator tables if needed.
- [ ] Avoid long unbreakable code/formula lines.
- [ ] Verify grayscale readability.
- [ ] Add “current as of” notes for hardware-specific values.

## Technical Validation Checklist

- [ ] Validate H100 SXM5 dense BF16 peak and sparse peak distinction.
- [ ] Validate H100 SM count and SKU-specific values.
- [ ] Validate H100 shared memory/L1, L2, registers, and Tensor Core details.
- [ ] Validate H100/H200 HBM capacity and bandwidth.
- [ ] Validate MI300X HBM capacity, bandwidth, dense BF16, and Infinity Fabric values.
- [ ] Validate B200/GB200 values and label shipped vs announced.
- [ ] Validate NVLink/NVSwitch bandwidth directionality.
- [ ] Validate PCIe 5 x16 theoretical bandwidth and directionality.
- [ ] Validate InfiniBand NDR/800G values and distinguish line rate from usable bandwidth.
- [ ] Soften absolute topology guidance.
- [ ] Label all workload-dependent rules `[REPRESENTATIVE]` or `[ENV-SPECIFIC]`.

---

# 7. Recommended Next Commit

After saving this file as:

```text
publishing/figure_plans/ch03a_figure_integration_plan.md
```

Run:

```powershell
git add publishing\\figure_plans\\ch03a_figure_integration_plan.md
git commit -m "Add Chapter 3A figure integration plan"
git push origin production-v1.0
```

Then update the master figure inventory:

```powershell
git add publishing\\figure_inventory.md
git commit -m "Update figure inventory for Chapter 3A"
git push origin production-v1.0
```

---

# 8. Next Production Step After This Plan

The next task should be:

```text
Chapter 3A Technical Validation Plan
```

Recommended file:

```text
publishing/validation/ch03a_technical_validation.md
```

The validation should cover:

1. H100 SXM5 dense and sparse BF16 values
2. H100 HBM capacity and bandwidth
3. H100 SM count and SM resource values
4. H100 L2 and shared memory/L1 details
5. H100 NVLink bandwidth and directionality
6. PCIe 5 x16 bandwidth and directionality
7. InfiniBand NDR / 800G bandwidth wording
8. H200 HBM capacity and bandwidth
9. MI300X HBM, BF16, Infinity Fabric, and TDP
10. B200/GB200 shipped vs announced values
11. Tensor Core datatype support by generation
12. Tensor Core alignment and fallback wording
13. SXM vs PCIe vs OAM form factor wording
14. Accelerator selection claims
15. Confidence labels
