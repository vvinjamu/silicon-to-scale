# Chapter 3B — GPU Architecture Roadmap: NVIDIA and AMD Generations

> “A roadmap is not a shopping list. It is a signal about which resource the industry expects to become the next bottleneck.”

---

## Chapter Overview

Chapter 3A explained the GPU as a performance system:

```text
SMs + warps + Tensor Cores + HBM + interconnect + software stack = accelerator behavior
```

Chapter 3B extends that mental model across generations.

A performance architect should not evaluate GPU roadmaps by asking only:

```text
Which GPU has the biggest TFLOPS number?
```

That is too shallow.

A better question is:

```text
Which resource changed, and does that resource limit my workload?
```

GPU generations evolve along several dimensions:

- Compute formats and peak throughput
- HBM capacity
- HBM bandwidth
- Scale-up interconnect
- Rack-scale system design
- Power and cooling
- Software maturity
- Compiler/kernel support
- Framework readiness
- Total cost of ownership

This chapter teaches you how to read GPU roadmaps safely, using NVIDIA and AMD accelerator generations as the main examples.

> **Current as of 2026 edition:** Hardware roadmap details change quickly. Treat all roadmap and future-generation claims as time-sensitive. Always verify final procurement decisions against current vendor product pages, datasheets, software support matrices, benchmark disclosures, and cloud availability.

---

## 3B.0 Roadmap Reading Rules

Before comparing A100, H100, H200, B200, GB200, MI300X, MI325X, MI350X, Rubin, or MI400, use these rules.

```text
Rule 1: Compare dense to dense and sparse to sparse.
Rule 2: Compare GPU-level numbers to GPU-level numbers.
Rule 3: Compare system-level numbers to system-level numbers.
Rule 4: Compare rack-level numbers to rack-level numbers.
Rule 5: Separate shipping specs from announced roadmap.
Rule 6: Treat benchmark wins as workload-specific, not universal.
Rule 7: Do not choose hardware from TFLOPS alone.
```

The biggest mistake in roadmap analysis is mixing units:

```text
B200 GPU != DGX B200 system != GB200 Superchip != GB200 NVL72 rack
```

Another common mistake is mixing dense and sparse peak numbers:

```text
H100 dense BF16 peak ≈ 989.4 TFLOPS
H100 sparse BF16 peak ≈ 1,978.9 TFLOPS
```

If a dense workload is measured against sparse peak, model utilization appears artificially low.

---

## Table 3B.1 — Roadmap Confidence Labels

| Label | Use in Ch03B | Example |
|---|---|---|
| `[SHIPPED]` | Vendor-published shipping product spec | H200 141 GB HBM3e, 4.8 TB/s |
| `[ANNOUNCED]` | Vendor-announced roadmap or future platform | Rubin or MI400 roadmap direction |
| `[DERIVED FROM SHIPPED]` | Calculation from official shipping spec | H100 dense BF16 derived from sparse peak |
| `[ESTIMATED]` | Simplified engineering estimate | Memory needed for 70B BF16 weights |
| `[REPRESENTATIVE]` | Workload-dependent architecture guidance | H200 often helps memory-bound inference |
| `[ENV-SPECIFIC]` | Benchmark or measured deployment behavior | MLPerf result, vendor benchmark, internal cluster throughput |

Use this table as the contract for the chapter. A shipping GPU specification is not the same thing as a roadmap announcement, and a benchmark result is not the same thing as a production guarantee.

> **Key Takeaway:** Roadmap claims are not all equal. Before using a hardware number, identify whether it is shipped, announced, derived, estimated, representative, or environment-specific.

---

## 3B.1 The Four-Lens GPU Generation Evaluation Framework

A GPU generation matters only if the generation changes a resource that limits your workload.

Use four lenses:

1. Compute density
2. Memory capacity and bandwidth
3. Interconnect and scale-up domain
4. Software and deployment maturity

---

## Figure Placeholder — Fig 3B.1

```markdown
![Fig 3B.1 — Four-Lens GPU Generation Evaluation Framework](../assets/diagrams/svg/ch03b_fig_3b_1_four_lens_gpu_generation_framework.svg)

**Fig 3B.1 — Four-Lens GPU Generation Evaluation Framework.** A performance architect evaluates each accelerator generation through compute density, memory architecture, interconnect domain, and software/deployment maturity.
```

**Figure intro:**  
GPU generations should not be evaluated by one headline number. A new generation may improve peak FLOPs, memory capacity, memory bandwidth, precision support, scale-up topology, or software-visible capability. The architect’s job is to identify which dimension actually changes the workload bottleneck.

**Figure explanation:**  
The four-lens framework keeps roadmap discussion disciplined. H200 is mainly a memory-capacity and memory-bandwidth story. Blackwell is partly a precision and rack-scale interconnect story. MI300X is strongly a memory-capacity story. Future Rubin or MI400 claims should be treated as roadmap signals until product-level specifications are verified.

> **Key Takeaway:** A GPU generation matters only when its changed resource maps to your workload bottleneck.

---

## 3B.2 Lens 1 — Compute Density

Compute density asks:

```text
How much math can this generation perform per GPU, per watt, per dollar, and per rack?
```

For AI workloads, compute density is shaped by:

- Tensor Core or matrix-engine generation
- Supported precision formats
- Dense vs sparse peak
- FP16, BF16, FP8, FP4, MXFP formats
- Real kernel availability
- Compiler/runtime support
- Sustained efficiency, not just peak numbers

Do not compare:

```text
H100 sparse BF16 peak
```

against:

```text
MI300X dense BF16 peak
```

That is not a valid comparison.

Compare dense-to-dense or sparse-to-sparse.

---

## 3B.3 Lens 2 — Memory Capacity and Bandwidth

Memory capacity asks:

```text
Does the model, KV cache, activations, and runtime overhead fit?
```

Memory bandwidth asks:

```text
Can the workload stream weights, activations, and KV cache quickly enough?
```

For LLM inference, memory often decides system economics before peak compute does.

A simplified 70B model memory estimate:

[ESTIMATED]

```text
BF16 weights ≈ 70B × 2 bytes = 140 GB
FP8 weights  ≈ 70B × 1 byte  = 70 GB
INT4 weights ≈ 70B × 0.5 byte = 35 GB
```

This estimate excludes:

- KV cache
- Runtime buffers
- Activations
- Fragmentation
- Speculative decoding buffers
- Quantization metadata
- Framework overhead

> **Key Takeaway:** A GPU with enough compute but insufficient memory capacity can still be the wrong choice.

---

## 3B.4 Lens 3 — Interconnect and Scale-Up Domain

Interconnect asks:

```text
How many accelerators can communicate efficiently as one local domain?
```

Important terms:

- PCIe
- NVLink
- NVSwitch
- Infinity Fabric
- InfiniBand
- Ethernet
- Rack-scale NVLink domains
- GPU-to-GPU bandwidth
- Line rate vs effective bandwidth

The key question is not only “what is the bandwidth?” but:

```text
What does that bandwidth connect?
```

A per-link number is not the same as a per-GPU aggregate number. A per-GPU number is not the same as a rack-level number. A network line rate is not the same as effective collective bandwidth.

---

## 3B.5 Lens 4 — Software and Deployment Maturity

Software maturity asks:

```text
Can production frameworks actually use the hardware paths?
```

This includes:

- CUDA / ROCm / XLA / vendor stack maturity
- PyTorch support
- Kernel libraries
- Compiler support
- Serving frameworks
- Distributed collectives
- Quantization support
- Debugging/profiling tools
- Driver stability
- Cloud availability
- Team expertise

[REPRESENTATIVE] The best accelerator on paper may not be the best accelerator for a team if the software path is immature for the workload.

---

# 3B.6 NVIDIA Roadmap: Ampere → Hopper → Blackwell → Rubin

The NVIDIA roadmap should be read as a sequence of system changes.

Ampere made A100 a major training platform. Hopper made FP8, HBM3, and Transformer Engine central to the LLM era. H200 expanded Hopper’s memory capacity and bandwidth. Blackwell pushes the system boundary toward newer low-precision formats and larger NVLink domains. Rubin / Vera Rubin should be treated as roadmap unless official product-level specifications are available.

---

## Figure Placeholder — Fig 3B.2

```markdown
![Fig 3B.2 — NVIDIA GPU Roadmap Timeline](../assets/diagrams/svg/ch03b_fig_3b_2_nvidia_roadmap_timeline.svg)

**Fig 3B.2 — NVIDIA GPU Roadmap Timeline.** NVIDIA accelerator generations should be read as a sequence of compute, memory, precision, interconnect, and system-level changes: Ampere, Hopper, Blackwell, and Rubin.
```

**Figure intro:**  
The NVIDIA roadmap is not simply “new GPU faster than old GPU.” Each generation changes a different part of the AI infrastructure stack. Ampere established A100 as a major training platform. Hopper made FP8 and HBM3 central to LLM infrastructure. H200 refreshed Hopper with larger and faster HBM. Blackwell shifts more emphasis toward rack-scale NVLink domains and newer precision formats. Rubin should be treated as roadmap until specific product specs are available.

**Figure explanation:**  
The timeline helps the reader avoid comparing the wrong product level. H100 and H200 are GPU-level products. DGX B200 and GB200 NVL72 are system/rack-level products. Rubin is a future roadmap family unless official product-level specifications are available.

> **Key Takeaway:** Read NVIDIA generations by product level and confidence label: GPU, system, rack, or roadmap.

---

## 3B.7 NVIDIA A100: Ampere Baseline

A100 is still important because it is the baseline many engineers compare against.

[SHIPPED] NVIDIA A100 80GB uses HBM2e memory. The 80GB PCIe SKU is commonly listed at 1,935 GB/s memory bandwidth, while the 80GB SXM SKU is listed at 2,039 GB/s.

[SHIPPED] A100 80GB lists BF16 and FP16 Tensor Core peak performance around 312 TFLOPS dense and 624 TFLOPS with structured sparsity.

Use A100 as a reference point for:

- Ampere-era training clusters
- BF16/FP16 dense vs sparse comparisons
- NVLink 3 class systems
- Migration to Hopper/H100

A100 remains a useful baseline, but not because it is the newest. It is useful because many production clusters and benchmarks were built around it.

---

## 3B.8 NVIDIA H100: Hopper Becomes the LLM Baseline

H100 became the defining accelerator of the early LLM infrastructure era.

Important changes included:

- Hopper Tensor Cores
- Transformer Engine
- FP8 support
- HBM3
- NVLink 4
- Strong CUDA/software ecosystem support
- Broad hyperscale adoption

[SHIPPED] H100 SXM5 80 GB provides 80 GB of HBM3 and 3.35 TB/s peak memory bandwidth.

[DERIVED FROM SHIPPED] H100 SXM5 dense/non-sparse BF16 Tensor Core peak is approximately 989.4 TFLOPS.

[SHIPPED] The commonly listed approximately 1,978.9 TFLOPS BF16 value is the sparse Tensor Core peak and should not be used as the denominator for dense MFU.

[SHIPPED] H100 SXM5 systems list up to 900 GB/s aggregate NVLink bandwidth per GPU using fourth-generation NVLink.

### What Changed with H100?

| Dimension | H100 Roadmap Meaning |
|---|---|
| Compute | Hopper Tensor Cores, FP8 Transformer Engine |
| Memory | 80 GB HBM3, 3.35 TB/s |
| Interconnect | NVLink 4, high-bandwidth intra-node fabric |
| Software | CUDA ecosystem and framework support matured around LLM workloads |
| System impact | Became a common reference platform for training and inference |

> **Key Takeaway:** H100 mattered because compute, memory, interconnect, and software maturity aligned for transformer workloads.

---

## 3B.9 NVIDIA H200: Memory Refresh with Large Inference Impact

H200 is best understood as a Hopper-generation memory refresh.

[SHIPPED] NVIDIA H200 provides 141 GB of HBM3e and 4.8 TB/s peak memory bandwidth.

[REPRESENTATIVE] H200 should be treated primarily as a Hopper-generation memory refresh: the major practical change versus H100 SXM5 is larger and faster HBM3e.

This matters for:

- Larger models
- Larger KV cache
- Longer context
- Higher concurrency
- Memory-bound decode
- Reduced sharding pressure

H200 is not important because it changes every part of the architecture. It is important because memory was a bottleneck.

### H100 vs H200 Mental Model

| Question | H100 | H200 |
|---|---|---|
| Generation | Hopper | Hopper-generation memory refresh |
| HBM capacity | 80 GB | 141 GB |
| HBM bandwidth | 3.35 TB/s | 4.8 TB/s |
| Main impact | LLM baseline | Better memory capacity and bandwidth for inference/HPC |
| Production lesson | Balanced Hopper platform | Memory can justify a refresh even without new architecture |

> **Key Takeaway:** A roadmap generation can matter because memory changed, even if the compute architecture is similar.

---

## 3B.10 NVIDIA Blackwell: Product-Level Discipline Required

Blackwell is where product-level confusion becomes dangerous.

Do not say:

```text
Blackwell has 1,440 GB memory.
```

That is a DGX B200 system-level number, not a single GPU number.

Do not say:

```text
GB200 NVL72 is a GPU.
```

It is a rack-scale platform.

Use the product level every time:

```text
B200 GPU
DGX B200 system
GB200 Superchip
GB200 NVL72 rack-scale platform
```

---

## Figure Placeholder — Fig 3B.7

```markdown
![Fig 3B.7 — GPU vs System vs Rack Product-Level Guardrail](../assets/diagrams/svg/ch03b_fig_3b_7_product_level_guardrail.svg)

**Fig 3B.7 — GPU vs System vs Rack Product-Level Guardrail.** Roadmap comparisons must not mix per-GPU, per-system, per-rack, and cluster-level numbers. Product level must be declared before any hardware value is compared.
```

**Figure intro:**  
Blackwell-era systems make product-level clarity mandatory. A B200 GPU, a DGX B200 system, a GB200 Superchip, and a GB200 NVL72 rack are not the same comparison unit.

**Figure explanation:**  
This figure should prevent one of the most common roadmap mistakes: comparing a rack-level HBM capacity or NVLink-domain number to a single-GPU number from a previous generation. Always declare whether a number is per GPU, per module, per node, per rack, or per cluster.

> **Key Takeaway:** Before comparing roadmap numbers, ask: “What is the unit — GPU, module, system, rack, or cluster?”

---

## Table 3B.6 — Product-Level Numbering Guardrail

| Level | Example | Valid Comparison | Common Mistake |
|---|---|---|---|
| GPU | H100, H200, B200, MI300X | GPU-to-GPU specs | Comparing one GPU to a full rack |
| Module / Superchip | GB200 Superchip, accelerator module | Module-to-module | Treating module value as per-GPU |
| System / Node | DGX B200, HGX system, 8-GPU server | System-to-system | Comparing system total HBM to per-GPU HBM |
| Rack | GB200 NVL72, rack-scale platforms | Rack-to-rack | Comparing rack NVLink domain to node fabric |
| Cluster | Multi-rack training system | Cluster-to-cluster | Confusing network bisection with local fabric |

> **Key Takeaway:** Most roadmap comparison errors happen because the comparison unit changed without the reader noticing.

---

## 3B.11 DGX B200, GB200 Superchip, and GB200 NVL72

[SHIPPED] DGX B200 is a system-level product with 8 NVIDIA Blackwell GPUs, 1,440 GB total GPU memory, 64 TB/s total HBM3e bandwidth, and 14.4 TB/s aggregate NVLink bandwidth.

These are system-level values.

They should not be quoted as per-GPU values.

[SHIPPED] The NVIDIA GB200 Superchip combines one Grace CPU and two Blackwell GPUs.

[SHIPPED] In the GB200 context, NVIDIA describes fifth-generation NVLink as providing 1.8 TB/s GPU-to-GPU bandwidth.

[SHIPPED] GB200 NVL72 is a rack-scale platform connecting 36 Grace CPUs and 72 Blackwell GPUs in a 72-GPU NVLink domain.

GB200 NVL72 values are rack-level or platform-level values. They should not be directly compared against a single H100, H200, B200, or MI300X.

### Blackwell Roadmap Lesson

Blackwell is not only a GPU upgrade. It is a system-design boundary shift:

```text
GPU-level thinking → system-level thinking → rack-scale thinking
```

> **Key Takeaway:** Blackwell-era infrastructure makes “what product level are we discussing?” a first-class architecture question.

---

## 3B.12 Rubin / Vera Rubin

[ANNOUNCED] Rubin / Vera Rubin should be treated as NVIDIA’s future roadmap direction beyond Blackwell.

Use Rubin as a planning signal unless official product-level specifications and availability are verified for the edition date.

Safe wording:

```text
Rubin indicates NVIDIA’s future direction beyond Blackwell.
```

Unsafe wording:

```text
Rubin has final production specs of X, Y, Z.
```

unless those values come from an official product-level source available at publication time.

> **Key Takeaway:** Future roadmap names are useful signals, not procurement-ready specifications.

---

## Table 3B.2 — NVIDIA Generation Comparison

| Generation / Product | Product Level | Main Change | Memory Story | Interconnect Story | Confidence |
|---|---|---|---|---|---|
| A100 | GPU | Ampere Tensor Core platform | 80 GB HBM2e; PCIe and SXM bandwidth differ | NVLink 3 class systems | `[SHIPPED]` |
| H100 | GPU | Hopper, FP8 Transformer Engine, HBM3 | 80 GB HBM3, 3.35 TB/s for SXM5 | NVLink 4, up to 900 GB/s aggregate per GPU | `[SHIPPED]` |
| H200 | GPU | Hopper-generation memory refresh | 141 GB HBM3e, 4.8 TB/s | Hopper platform continuity | `[SHIPPED]` |
| B200 | GPU / module depending context | Blackwell compute and precision evolution | Product-specific HBM3e | NVLink 5 class systems | `[SHIPPED]` or `[ANNOUNCED]` depending source |
| DGX B200 | System | 8-GPU Blackwell system | 1,440 GB total GPU memory, 64 TB/s total HBM3e bandwidth | 14.4 TB/s aggregate NVLink bandwidth | `[SHIPPED]` |
| GB200 Superchip | Module / superchip | Grace CPU + two Blackwell GPUs | Module-level values | Fifth-generation NVLink in GB200 context | `[SHIPPED]` |
| GB200 NVL72 | Rack-scale platform | 72-GPU rack-scale NVLink domain | Rack-level HBM values | 72-GPU NVLink domain | `[SHIPPED]` |
| Rubin / Vera Rubin | Roadmap | Future generation beyond Blackwell | Roadmap-specific | Roadmap-specific | `[ANNOUNCED]` |

This table is a roadmap reading tool. It shows that H200 should not be described as a completely new architecture in the same way Blackwell is. It also shows that GB200 NVL72 should not be compared directly to a single GPU number.

> **Key Takeaway:** Every NVIDIA roadmap number must answer: per GPU, per system, per rack, or roadmap?

---

# 3B.13 AMD Roadmap: CDNA 3 → CDNA 3+ → CDNA 4 → MI400

AMD’s roadmap tells a different story from NVIDIA’s roadmap.

AMD’s accelerator strategy is strongly associated with:

- Large HBM capacity
- High HBM bandwidth
- Chiplet packaging
- CDNA architecture evolution
- ROCm/HIP software stack
- OAM-style accelerator platforms
- Future rack-scale system direction

The point is not to declare AMD or NVIDIA the universal winner. The point is to evaluate each by workload fit.

---

## Figure Placeholder — Fig 3B.3

```markdown
![Fig 3B.3 — AMD Instinct Roadmap Timeline](../assets/diagrams/svg/ch03b_fig_3b_3_amd_roadmap_timeline.svg)

**Fig 3B.3 — AMD Instinct Roadmap Timeline.** AMD’s AI accelerator roadmap should be read through memory capacity, HBM bandwidth, chiplet architecture, precision support, and rack-scale platform direction.
```

**Figure intro:**  
The AMD roadmap tells a different but important story. MI300X emphasized very large HBM capacity and bandwidth. MI325X extended the memory story. MI350/MI355 moved the platform toward CDNA 4, larger HBM3e capacity, higher bandwidth, and new low-precision formats. MI400 should be treated as roadmap unless final product specifications are available.

**Figure explanation:**  
The timeline should show AMD’s memory-first relevance for LLM serving and long-context workloads. It should also avoid implying software maturity, benchmark parity, or deployment availability without evidence. The roadmap should be treated as a sequence of architectural signals, not a universal NVIDIA-vs-AMD conclusion.

> **Key Takeaway:** AMD roadmap evaluation should focus on memory fit, bandwidth, interconnect, precision support, ROCm/software readiness, and platform availability.

---

## 3B.14 AMD MI300X: Large-HBM CDNA 3 Accelerator

[SHIPPED] AMD Instinct MI300X provides 192 GB HBM3 and approximately 5.3 TB/s peak memory bandwidth.

[SHIPPED] AMD lists peak dense BF16 at approximately 1,307.4 TFLOPS and sparse BF16 at approximately 2,614.9 TFLOPS.

[REPRESENTATIVE] MI300X’s large HBM footprint is especially relevant for memory-heavy inference and KV-cache-heavy workloads, provided the software stack fits the deployment.

Do not say:

```text
MI300X always wins 70B serving.
```

Safer:

```text
MI300X’s 192 GB HBM can be attractive for memory-heavy 70B-class inference because it may reduce sharding pressure and leave more KV-cache headroom, but production results depend on model, precision, serving stack, kernels, batching, and software maturity.
```

---

## Figure Placeholder — Fig 3B.6

```markdown
![Fig 3B.6 — AMD MI300X Die Stack: Chiplets and HBM](../assets/diagrams/png_300dpi/ch03b_fig_3b_6_mi300x_die_stack.png)

**Fig 3B.6 — AMD MI300X Die Stack: Chiplets and HBM.** MI300X uses a chiplet-based accelerator architecture with multiple compute dies and large HBM capacity, making it especially relevant for memory-heavy AI workloads.
```

**Figure intro:**  
MI300X is important not only because of its compute number, but because of its packaging and memory story. Large HBM capacity changes model-fit and KV-cache economics for LLM inference.

**Figure explanation:**  
The die-stack figure should be used to explain why MI300X is often discussed in the context of large-memory serving. The architectural lesson is that memory capacity, bandwidth, packaging, and software support together determine production usefulness.

> **Key Takeaway:** MI300X is a roadmap milestone because it made large-HBM accelerator design central to AI infrastructure discussions.

---

## 3B.15 AMD MI325X: Memory Refresh

[SHIPPED/ANNOUNCED] AMD Instinct MI325X is a memory-focused refresh with 256 GB HBM3e and approximately 6 TB/s peak theoretical memory bandwidth.

Use `[SHIPPED]` only when referencing a final product page or datasheet. Use `[ANNOUNCED]` if the source language is forward-looking.

Why MI325X matters:

- Larger HBM than MI300X
- Higher memory bandwidth
- Improved model-fit and KV-cache headroom
- Same general large-memory accelerator theme

[REPRESENTATIVE] MI325X is best understood as extending the MI300X memory story rather than replacing the need for workload validation.

---

## 3B.16 AMD MI350X / MI355X: CDNA 4 and Larger Memory

[SHIPPED] AMD Instinct MI350 Series product materials list 288 GB HBM3e and 8 TB/s peak memory bandwidth, with CDNA 4 architecture and next-generation MXFP datatype support.

Use product-specific wording for MI350X vs MI355X.

The roadmap signal:

- Larger HBM capacity
- Higher HBM bandwidth
- CDNA 4 architecture
- MXFP4 / MXFP6 class low-precision support
- Continued focus on large-memory AI training and inference

[REPRESENTATIVE] MI350-class accelerators should be evaluated by memory capacity, memory bandwidth, low-precision kernel support, ROCm/framework maturity, platform availability, and workload fit.

---

## 3B.17 AMD MI400 / Helios

[ANNOUNCED] AMD Helios / MI400 should be discussed as a future rack-scale roadmap direction, not a shipping product specification, unless official product-level datasheets exist at publication time.

Safe wording:

```text
MI400 / Helios signals AMD’s direction toward future rack-scale AI infrastructure.
```

Unsafe wording:

```text
MI400 has final spec X and should beat product Y.
```

unless final official product specifications and benchmark context exist.

> **Key Takeaway:** MI400 and Helios are planning signals until final product-level specifications and availability are verified.

---

## Table 3B.3 — AMD Instinct Generation Comparison

| Product | Generation / Family | Main Change | Memory Story | Compute / Precision Story | Confidence |
|---|---|---|---|---|---|
| MI300X | CDNA 3 | Large-HBM chiplet accelerator | 192 GB HBM3, ≈5.3 TB/s | Dense BF16 ≈1,307.4 TFLOPS | `[SHIPPED]` |
| MI325X | CDNA 3+ / refresh | Memory capacity/bandwidth refresh | 256 GB HBM3e, ≈6 TB/s | Same general large-memory theme; verify exact compute values | `[SHIPPED]` or `[ANNOUNCED]` depending source |
| MI350X / MI355X | CDNA 4 / MI350 Series | Larger HBM and low-precision support | 288 GB HBM3e, 8 TB/s class bandwidth | MXFP4/MXFP6 class support; SKU-specific | `[SHIPPED]` if official product page |
| MI400 / Helios | Roadmap / next generation | Future rack-scale/platform direction | TBD until official specs | TBD until official specs | `[ANNOUNCED]` |

The table emphasizes AMD’s large-memory story without making universal performance claims. A memory-rich accelerator can be attractive for model-fit and KV-cache-heavy workloads, but production choice also depends on kernels, framework maturity, interconnect, support model, and operational risk.

> **Key Takeaway:** AMD’s roadmap should be evaluated through large HBM capacity, bandwidth, chiplet design, software maturity, and workload fit — not peak numbers alone.

---

# 3B.18 HBM Evolution: HBM2e → HBM3 → HBM3e → HBM4

The AI accelerator roadmap is also a memory roadmap.

Many LLM workloads are constrained by:

- Weight memory
- KV-cache memory
- Activation memory
- Runtime buffers
- HBM bandwidth
- Memory fragmentation
- Context length
- Concurrency

That is why HBM matters so much.

---

## Figure Placeholder — Fig 3B.4

```markdown
![Fig 3B.4 — HBM Evolution and AI Accelerator Roadmaps](../assets/diagrams/svg/ch03b_fig_3b_4_hbm_evolution.svg)

**Fig 3B.4 — HBM Evolution and AI Accelerator Roadmaps.** AI accelerator generations increasingly depend on HBM capacity and bandwidth. HBM evolution changes model-fit, KV-cache capacity, long-context serving, and training memory economics.
```

**Figure intro:**  
The AI accelerator roadmap is also a memory roadmap. Many LLM workloads are constrained not by peak compute, but by whether the model and KV cache fit in HBM and whether decode can stream data fast enough.

**Figure explanation:**  
HBM2e, HBM3, HBM3e, and HBM4 should be discussed as technology generations, while product-specific values must remain tied to a GPU or accelerator SKU. H100, H200, B200/GB200, MI300X, MI325X, and MI350 Series use memory differently at GPU, system, and rack levels.

> **Key Takeaway:** For LLM infrastructure, memory generation can matter as much as compute generation.

---

## Table 3B.4 — Memory Evolution in AI Accelerators

| Memory Generation | Used In / Associated With | What It Changes | Production Caution |
|---|---|---|---|
| HBM2e | A100-era accelerators | Large memory bandwidth relative to prior GPU generations | Product values vary by SKU |
| HBM3 | H100, MI300X class | Higher bandwidth and capacity for LLM training/inference | Compare product-specific values |
| HBM3e | H200, B200/GB200, MI325X/MI350 class | Larger capacity and higher bandwidth | Do not mix GPU, system, and rack totals |
| HBM4 | Future roadmap | Expected future capacity/bandwidth direction | Use `[ANNOUNCED]` or `[ESTIMATED]` unless final product specs exist |

[REPRESENTATIVE] AI accelerator roadmaps increasingly depend on HBM capacity and bandwidth. A100-era systems used HBM2e; H100/MI300X-class accelerators use HBM3; H200, B200/GB200, MI325X, and MI350-class products use HBM3e. HBM4 should be treated as roadmap/future until tied to official product specifications.

> **Key Takeaway:** HBM generation is a useful signal, but product-specific HBM capacity and bandwidth are what matter for architecture decisions.

---

# 3B.19 Interconnect Evolution: NVLink, NVSwitch, Infinity Fabric

Large AI systems are communication machines.

Interconnect evolution is about expanding the efficient communication domain:

```text
GPU → Node → Rack → Cluster
```

[REPRESENTATIVE] Interconnect roadmaps are about expanding the efficient communication domain. NVLink/NVSwitch and AMD Infinity Fabric should be evaluated by scope: per link, per GPU, per node, per rack, aggregate, bidirectional, or effective application bandwidth.

---

## Figure Placeholder — Fig 3B.5

```markdown
![Fig 3B.5 — Scale-Up Fabric Evolution: NVLink, NVSwitch, and Infinity Fabric](../assets/diagrams/svg/ch03b_fig_3b_5_interconnect_evolution.svg)

**Fig 3B.5 — Scale-Up Fabric Evolution: NVLink, NVSwitch, and Infinity Fabric.** AI accelerator roadmaps increasingly expand the scale-up communication domain, moving from GPU-to-GPU links to system-level and rack-level fabrics.
```

**Figure intro:**  
Large AI systems are not just collections of fast GPUs. They are communication machines. As models grow, the boundary of high-bandwidth communication shifts from one GPU to one node and, increasingly, toward rack-scale domains.

**Figure explanation:**  
NVLink/NVSwitch and AMD Infinity Fabric should be discussed by scope and directionality. Is the number per link, per GPU, per system, per rack, per direction, bidirectional aggregate, or effective application bandwidth? Without that clarity, interconnect comparisons become misleading.

> **Key Takeaway:** Interconnect evolution is about expanding the efficient communication domain, not just increasing a bandwidth number.

---

## Table 3B.5 — Interconnect Evolution and Directionality Caution

| Fabric / Generation | Associated Products | Scope | What Changed | Directionality Caution |
|---|---|---|---|---|
| NVLink 3 | A100 systems | Intra-node GPU fabric | Ampere scale-up fabric | State per GPU/aggregate |
| NVLink 4 | H100/H200 systems | Intra-node GPU fabric | Hopper scale-up bandwidth | H100 SXM often listed as 900 GB/s aggregate per GPU |
| NVLink 5 | Blackwell systems | Node/rack-scale fabric depending platform | Higher bandwidth, larger NVLink domains | Product-specific |
| NVSwitch | DGX/HGX/NVL systems | Switched local/rack fabric | Many-GPU communication domain | System/rack specific |
| AMD Infinity Fabric | MI300/MI350 systems | Accelerator fabric / package/platform | GPU-to-GPU and package/platform communication | Product/platform specific |
| InfiniBand/Ethernet | Multi-node fabric | Cluster | Scale-out training/serving | Line rate ≠ effective NCCL bandwidth |

The right comparison starts with scope. A tensor-parallel group cares about the local scale-up fabric. Data parallelism cares about scale-out bandwidth and collective efficiency. Expert parallelism may care about all-to-all patterns and topology.

> **Key Takeaway:** Interconnect numbers are meaningful only after you know the communication pattern and the scope of the bandwidth number.

---

# 3B.20 NVIDIA vs AMD: Neutral Architecture Comparison

Avoid vendor-ranking language.

Do not write:

```text
NVIDIA wins.
AMD wins.
```

without a specific workload, software stack, metric, and measurement method.

Better:

```text
[REPRESENTATIVE] NVIDIA emphasizes vertically integrated GPU, networking, software, and rack-scale systems. AMD emphasizes high-memory accelerators, ROCm/HIP software momentum, chiplet packaging, and rack-scale roadmap systems. Both must be evaluated through workload fit and operational maturity.
```

A fair comparison includes:

- Model size
- Precision
- Sequence length
- Batch size
- Training vs inference
- Prefill vs decode
- Tensor parallelism
- KV cache
- Kernel stack
- Framework version
- Driver/runtime version
- Interconnect
- Power/cooling
- Cost
- Availability
- Team expertise

[ENV-SPECIFIC] NVIDIA-vs-AMD performance comparisons are benchmark-specific. A valid comparison must state model, precision, sequence length, batch size, framework, kernel stack, compiler/runtime versions, topology, power settings, and metric.

> **Key Takeaway:** Architecture comparison is useful. Universal vendor ranking is not.

---

# 3B.21 Hardware Selection Decision Tree

The purpose of roadmap analysis is not to memorize product names. The purpose is to choose the right system for a workload.

---

## Figure Placeholder — Fig 3B.8

```markdown
![Fig 3B.8 — Hardware Selection Decision Tree](../assets/diagrams/svg/ch03b_fig_3b_8_hardware_selection_decision_tree.svg)

**Fig 3B.8 — Hardware Selection Decision Tree.** A performance architect selects hardware by matching workload bottlenecks to hardware resources: memory capacity, memory bandwidth, compute density, interconnect, precision support, software stack, power, cost, and operational risk.
```

**Figure intro:**  
The purpose of roadmap analysis is not to memorize product names. The purpose is to choose the right system for a workload. This decision tree turns roadmap knowledge into architecture judgment.

**Figure explanation:**  
A 70B inference service may prioritize HBM capacity and KV-cache headroom. A large training cluster may prioritize local and scale-out interconnect. A cost-sensitive deployment may prioritize software maturity, availability, and cost per token. Future platforms such as Rubin or MI400 should be treated as planning signals until production details are verified.

> **Key Takeaway:** Hardware selection starts with the workload bottleneck, not the newest product name.

---

## Table 3B.7 — Workload-to-Hardware Selection Matrix

| Workload Scenario | Most Important Hardware Attribute | Roadmap Implication |
|---|---|---|
| 70B inference with long context | HBM capacity, HBM bandwidth, KV-cache efficiency | H200, MI300X/MI325X/MI350-class memory-rich systems may be attractive |
| Multi-GPU tensor-parallel serving | Local scale-up fabric | NVLink/NVSwitch or equivalent high-bandwidth local fabric matters |
| Frontier training | Compute, HBM, scale-up + scale-out fabric, reliability | Hopper/Blackwell-class and rack-scale systems matter |
| Cost-sensitive inference | Software stack, availability, power, quantization support | Peak TFLOPS may matter less than cost per token |
| Long-context workloads | HBM capacity, bandwidth, attention kernels | HBM3e/HBM4 and FlashAttention-style kernels matter |
| MoE training/inference | All-to-all communication and routing balance | Interconnect and scheduler design matter |
| Enterprise deployment | Software maturity, support, observability | Ecosystem and operational maturity matter |

[REPRESENTATIVE] The table should be used as architecture guidance, not a benchmark claim. Real decisions still require profiling, cost modeling, framework testing, power/cooling analysis, and support review.

> **Key Takeaway:** The same hardware can be excellent for one workload and inefficient for another.

---

## 3B.22 Worked Example: 70B Inference

[ESTIMATED]

```text
70B BF16 weights ≈ 140 GB
70B FP8 weights  ≈ 70 GB
70B INT4 weights ≈ 35 GB
```

For 70B serving:

- H100 80 GB may fit FP8 weights but has limited KV-cache headroom.
- H200 141 GB gives more memory headroom.
- MI300X 192 GB gives even more memory capacity.
- MI325X and MI350-class accelerators continue the large-memory story.
- B200/GB200-class systems may add newer precision support and larger system/rack-scale fabrics.

But none of this answers the full production question.

You still need to evaluate:

- KV cache size
- Context length
- Concurrency
- Precision
- Quantization quality
- Prefill/decode mix
- Batching strategy
- Tensor parallelism
- Kernel availability
- Framework support
- Power and cost

> **Key Takeaway:** 70B inference is often a memory-capacity and KV-cache economics problem before it is a peak TFLOPS problem.

---

## 3B.23 Worked Example: Training Cluster Selection

A large training cluster cares about:

- Dense or low-precision Tensor Core throughput
- HBM capacity
- HBM bandwidth
- Activation memory
- Optimizer state memory
- NVLink/NVSwitch domain
- InfiniBand/Ethernet scale-out fabric
- NCCL/RCCL performance
- Failure recovery
- Checkpointing
- Power and cooling
- Software maturity

A single GPU number is not enough.

[REPRESENTATIVE] Frontier training is a system-level problem: compute, memory, local fabric, scale-out network, storage, scheduler, observability, and reliability all matter.

---

## 3B.24 How to Discuss GPU Roadmaps in a Principal Interview

Do not say:

```text
Blackwell is faster, so it is better.
```

A stronger answer:

> I do not evaluate accelerator roadmaps by peak TFLOPS alone. I look at four lenses: compute density, memory capacity and bandwidth, interconnect domain, and software maturity. H100 became the LLM baseline because Hopper Tensor Cores, HBM3, FP8, NVLink 4, and the CUDA ecosystem aligned. H200 was mainly a memory refresh that improved inference capacity. Blackwell moves the system boundary toward newer precision paths and rack-scale NVLink domains. AMD’s MI300X and MI350 line emphasize very large HBM capacity and bandwidth, which can be compelling for memory-bound inference and long-context workloads if the software stack fits. For future Rubin or MI400 claims, I separate vendor-announced roadmap from shipping specs.

### Scenario 1 — Why Did H100 Matter?

Weak answer:

```text
H100 is faster than A100.
```

Better answer:

```text
H100 mattered because Hopper combined Tensor Core improvements, FP8 Transformer Engine, HBM3, NVLink 4, and mature software support. It became a balanced platform for transformer training and inference.
```

### Scenario 2 — Why Did H200 Matter If Compute Was Similar?

Weak answer:

```text
H200 has more memory.
```

Better answer:

```text
H200 mattered because many LLM inference workloads are memory-capacity and memory-bandwidth constrained. Moving to 141 GB HBM3e and 4.8 TB/s improves model fit, KV-cache headroom, and memory-bound serving economics.
```

### Scenario 3 — Why Is Blackwell More Than Faster H100?

Weak answer:

```text
Blackwell has more FLOPS.
```

Better answer:

```text
Blackwell should be evaluated as a system-level transition: newer precision formats, higher memory bandwidth, NVLink 5, and rack-scale platforms such as GB200 NVL72. The major architectural story is not just per-GPU speed but the larger scale-up domain.
```

### Scenario 4 — Why Is MI300X Important?

Weak answer:

```text
MI300X has 192 GB memory.
```

Better answer:

```text
MI300X is important because 192 GB HBM3 changes model-fit and KV-cache economics for memory-heavy inference. It is especially relevant when memory capacity and bandwidth are the bottleneck, provided the ROCm/framework/kernel stack fits the workload.
```

### Scenario 5 — How Do You Evaluate Rubin or MI400?

Weak answer:

```text
They will be much faster.
```

Better answer:

```text
I treat Rubin and MI400 as roadmap signals unless final product specs exist. I separate announced roadmap direction from shipping product specifications, then watch which resource is changing: memory, precision, interconnect, rack-scale design, or software ecosystem.
```

### Scenario 6 — What Is the Biggest Roadmap Comparison Mistake?

Answer:

```text
Mixing product levels. A single GPU, an 8-GPU system, a superchip, and a 72-GPU rack are different comparison units. I always identify the unit before comparing memory, bandwidth, or performance.
```

---

## 3B.25 Chapter Cheat Sheet

### Roadmap Rule

```text
A roadmap is a signal, not a guarantee.
```

### Four Lenses

```text
1. Compute density
2. Memory capacity and bandwidth
3. Interconnect / scale-up domain
4. Software and deployment maturity
```

### Dense vs Sparse

```text
Compare dense to dense.
Compare sparse to sparse.
Do not mix them.
```

### Product Level

```text
GPU != module != system != rack != cluster
```

### H100 Reference

```text
[SHIPPED] H100 SXM5: 80 GB HBM3, 3.35 TB/s
[DERIVED FROM SHIPPED] Dense BF16 ≈ 989.4 TFLOPS
[SHIPPED] Sparse BF16 ≈ 1,978.9 TFLOPS
```

### H200 Reference

```text
[SHIPPED] H200: 141 GB HBM3e, 4.8 TB/s
[REPRESENTATIVE] Hopper-generation memory refresh
```

### MI300X Reference

```text
[SHIPPED] MI300X: 192 GB HBM3, ≈5.3 TB/s
[SHIPPED] Dense BF16 ≈1,307.4 TFLOPS
```

### Blackwell Guardrail

```text
B200 GPU
DGX B200 system
GB200 Superchip
GB200 NVL72 rack
```

### Future Roadmap Guardrail

```text
Rubin and MI400 are [ANNOUNCED] unless final product specs exist.
```

---

## 3B.26 Key Takeaways

1. GPU roadmaps should be treated as architecture signals, not marketing claims.
2. Always distinguish shipped product specs from announced roadmap claims.
3. Always distinguish dense and sparse peak performance.
4. Always identify the product level: GPU, module, system, rack, or cluster.
5. A100 remains useful as an Ampere baseline.
6. H100 became the LLM-era baseline because compute, memory, interconnect, and software maturity aligned.
7. H200 matters primarily as a Hopper-generation memory refresh.
8. Blackwell shifts roadmap discussion toward newer precision formats and larger scale-up domains.
9. DGX B200 values are system-level values, not per-GPU values.
10. GB200 NVL72 values are rack-scale values, not single-GPU values.
11. Rubin / Vera Rubin should be treated as roadmap unless final product specs are verified.
12. MI300X made large-HBM accelerator design central to AI infrastructure discussions.
13. MI325X and MI350-class products continue the large-memory, high-bandwidth roadmap.
14. MI400 / Helios should be treated as AMD roadmap direction until final product specs exist.
15. HBM evolution is central to LLM inference and long-context serving.
16. Interconnect evolution is about expanding the efficient communication domain.
17. NVIDIA-vs-AMD comparisons must be workload-specific and evidence-based.
18. Hardware selection starts with the workload bottleneck, not the newest product name.

---

## 3B.27 Review Questions

### Conceptual

1. Why should GPU roadmaps be treated as architecture signals rather than marketing claims?
2. What are the four lenses for evaluating a GPU generation?
3. Why is H200 important even if it is not a fundamentally new architecture?
4. Why is Blackwell more than a faster GPU?
5. Why is MI300X important for memory-heavy inference?
6. Why should Rubin and MI400 be labeled as roadmap unless final product specs exist?
7. Why is HBM evolution central to LLM infrastructure?
8. Why is interconnect scope as important as interconnect bandwidth?
9. Why is software maturity part of hardware evaluation?
10. Why is universal NVIDIA-vs-AMD ranking misleading?

### Calculation

1. Estimate BF16 weight memory for a 70B model.
2. Estimate FP8 weight memory for a 70B model.
3. Estimate INT4 weight memory for a 70B model.
4. If H100 sparse BF16 peak is approximately 1,978.9 TFLOPS, what is the approximate dense BF16 peak?
5. If a system-level product lists 1,440 GB total GPU memory across 8 GPUs, what is the simple per-GPU average? What label should this calculation use?
6. If a rack-level product has 72 GPUs, why should its values not be directly compared to one GPU?

### Principal-Level Interview Practice

1. Explain the difference between GPU-level, system-level, and rack-level roadmap claims.
2. Explain why dense vs sparse peak matters for MFU.
3. Explain H100 vs H200 for LLM inference.
4. Explain Blackwell as a system-level transition.
5. Explain why GB200 NVL72 changes the scale-up domain.
6. Explain MI300X’s relevance to 70B-class inference.
7. Explain how you would evaluate MI350X vs B200 for a long-context workload.
8. Explain how you would discuss Rubin or MI400 without overclaiming.
9. Explain why benchmark comparisons must be environment-specific.
10. Explain how you would select hardware for a new LLM serving platform.

---

## 3B.28 Claims Requiring Annual Refresh

## Table 3B.8 — Roadmap Claims Requiring Annual Refresh

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

This table is a maintenance asset. It helps future editions stay accurate and prevents roadmap chapters from becoming stale.

> **Key Takeaway:** Roadmap content must be refreshed. The method is durable; the product details change.

---

## 3B.29 Production Notes for This Chapter

### Figure Assets Needed

| Figure | Status |
|---|---|
| Fig 3B.1 — Four-Lens GPU Generation Evaluation Framework | Must be created |
| Fig 3B.2 — NVIDIA Roadmap Timeline | Must be created |
| Fig 3B.3 — AMD Roadmap Timeline | Must be created |
| Fig 3B.4 — HBM Evolution | Existing partial; roadmap visual must be created |
| Fig 3B.5 — NVLink / Infinity Fabric Evolution | Must be created |
| Fig 3B.6 — AMD MI300X Die Stack | Existing; needs print export |
| Fig 3B.7 — Product-Level Numbering Guardrail | Must be created |
| Fig 3B.8 — Hardware Selection Decision Tree | Must be created |

### Table Assets Included

| Table | Status |
|---|---|
| Table 3B.1 — Roadmap Confidence Labels | Included |
| Table 3B.2 — NVIDIA Generation Comparison | Included |
| Table 3B.3 — AMD Instinct Generation Comparison | Included |
| Table 3B.4 — Memory Evolution | Included |
| Table 3B.5 — Interconnect Evolution | Included |
| Table 3B.6 — Product-Level Numbering Guardrail | Included |
| Table 3B.7 — Workload-to-Hardware Selection Matrix | Included |
| Table 3B.8 — Claims Requiring Annual Refresh | Included |

### Source Notes to Add in Final Book

Use official or primary sources for:

- NVIDIA A100 product page / datasheet
- NVIDIA H100 product page / datasheet
- NVIDIA H200 product page / datasheet
- NVIDIA DGX B200 product page / datasheet
- NVIDIA DGX GB200 / GB200 NVL72 product pages
- NVIDIA Rubin / Vera Rubin official roadmap pages
- AMD MI300X product page / datasheet
- AMD MI325X product page / datasheet
- AMD MI350 Series / MI350X / MI355X product pages
- AMD CDNA 4 architecture whitepaper
- AMD Helios / MI400 official roadmap announcement
- JEDEC or vendor HBM roadmap sources
- MLPerf results for benchmark comparisons
- CUDA, ROCm, NCCL, RCCL, PyTorch, and serving-framework documentation

---

## 3B.30 Bridge to Chapter 4

Chapter 3A taught the accelerator architecture fundamentals.

Chapter 3B taught how accelerator generations evolve and how to interpret roadmap claims safely.

Chapter 4 moves deeper into one resource that keeps appearing in every roadmap:

```text
memory
```

The next question is:

> Why do HBM capacity, HBM bandwidth, cache hierarchy, and data movement dominate so many AI performance problems?

That is the focus of Chapter 4.
