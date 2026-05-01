# Chapter 3A Technical Validation Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Ch03A — *GPU and Accelerator Architecture Fundamentals*  
**Target file:** `publishing/validation/ch03a_technical_validation.md`  
**Production status:** Draft validation plan for `production-v1.0`  
**Last reviewed:** 2026-04-30  
**Purpose:** Validate GPU/accelerator architecture numbers, topology guidance, form-factor claims, and confidence labels before producing the Chapter 3A production Markdown source.

---

## 0. Executive Summary

Chapter 3A is hardware-heavy. That makes validation especially important because GPU specifications, accelerator roadmaps, network speeds, and vendor product names change quickly.

The biggest production corrections are:

1. **H100 BF16 must distinguish dense/non-sparse from sparse.**
   - Dense/non-sparse BF16 Tensor Core peak should be presented as approximately **989.4 TFLOPS**.
   - Sparse BF16 Tensor Core peak should be presented as approximately **1,978.9 TFLOPS**.
   - Do not write “H100 BF16 = 1,979 TFLOPS” without saying **with sparsity**.

2. **Bandwidth claims must state directionality.**
   - PCIe 5.0 x16 is approximately **64 GB/s per direction**, or about **128 GB/s aggregate bidirectional theoretical bandwidth**.
   - H100 SXM NVLink is commonly listed as **900 GB/s aggregate bandwidth per GPU**. State whether the number is aggregate and vendor-published.

3. **Topology guidance must be workload-aware.**
   - Avoid absolute wording such as “never tensor-parallel across InfiniBand.”
   - Use: “Avoid latency-sensitive tensor-parallel collectives across the inter-node fabric unless the communication schedule, model partitioning, and network topology are designed for it.”

4. **B200 / GB200 must be labeled carefully.**
   - Treat broad Blackwell product claims as `[SHIPPED]` only when tied to a shipping SKU or official product page.
   - Treat roadmap or vendor-announced rack-scale claims as `[ANNOUNCED]` unless shipment status is verified for your publication date.

5. **Accelerator comparisons must remain neutral.**
   - Do not rank NVIDIA, AMD, TPU, or Gaudi universally.
   - Compare workload fit, memory capacity, interconnect, software stack, operational maturity, cost, and risk.

---

# 1. Confidence Label Policy for Chapter 3A

| Label | Use in Ch03A |
|---|---|
| `[SHIPPED]` | Vendor-published shipping hardware or software specification |
| `[ANNOUNCED]` | Vendor-disclosed or roadmap hardware not verified as broadly shipping |
| `[DERIVED FROM SHIPPED]` | Arithmetic derived from official shipping specs |
| `[ESTIMATED]` | First-principles calculation or simplified performance model |
| `[REPRESENTATIVE]` | Practical rule of thumb that varies by workload |
| `[ENV-SPECIFIC]` | Measured performance, sustained throughput, kernel behavior, cluster behavior |

Production rule:

```text
Attach the label to the claim, not to the whole paragraph.
```

Good:

```text
[DERIVED FROM SHIPPED] H100 SXM5 dense/non-sparse BF16 Tensor Core peak is approximately 989.4 TFLOPS.
```

Avoid:

```text
H100 BF16 is 1,979 TFLOPS.
```

---

# 2. Validation Table

---

## 2.1 H100 SXM5 Dense BF16 Peak TFLOPS

| Field | Validation |
|---|---|
| Claim | H100 SXM5 dense/non-sparse BF16 Tensor Core peak is approximately 989.4 TFLOPS. |
| Current value or formula | `1,978.9 TFLOPS sparse BF16 / 2 ≈ 989.4 TFLOPS dense BF16` |
| Validation status | **Valid when explicitly labeled as dense/non-sparse.** |
| Corrected value or safer wording | Use `≈989.4 TFLOPS dense/non-sparse BF16 Tensor Core peak`. |
| Confidence label | `[DERIVED FROM SHIPPED]` or `[SHIPPED]` if sourced from a table that explicitly lists non-sparse value |
| Source type needed | NVIDIA H100 datasheet / official product specification |
| Recommended final wording | `[DERIVED FROM SHIPPED] H100 SXM5 dense/non-sparse BF16 Tensor Core peak is approximately 989.4 TFLOPS. Do not confuse this with the sparse BF16 peak, which is approximately 1,978.9 TFLOPS.` |
| Priority | **P0** |

### Production note

This is the most important correction in Chapter 3A. Keep the same value already validated for Chapter 1.

---

## 2.2 H100 SXM5 Sparse BF16 Peak TFLOPS

| Field | Validation |
|---|---|
| Claim | H100 SXM5 sparse BF16 Tensor Core peak is approximately 1,978.9 TFLOPS. |
| Current value or formula | Vendor tables often list BF16 Tensor Core as about `1,979 TFLOPS` with sparsity. |
| Validation status | **Valid if explicitly stated as sparse.** |
| Corrected value or safer wording | Add “with sparsity” every time this value appears. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 datasheet / product page |
| Recommended final wording | `[SHIPPED] H100 SXM5 sparse BF16 Tensor Core peak is approximately 1,978.9 TFLOPS. Sparse Tensor Core values assume NVIDIA structured sparsity and should not be used as dense model peak throughput.` |
| Priority | **P0** |

---

## 2.3 H100 SXM5 HBM Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | H100 SXM5 has 80 GB HBM3 and 3.35 TB/s memory bandwidth. |
| Current value or formula | `80 GB HBM3`, `3.35 TB/s` |
| Validation status | **Valid for H100 SXM5 80 GB configuration.** |
| Corrected value or safer wording | Specify SXM5 and 80 GB SKU. Avoid applying this to all H100 variants. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 product spec / datasheet |
| Recommended final wording | `[SHIPPED] H100 SXM5 80 GB provides 80 GB of HBM3 with 3.35 TB/s peak memory bandwidth. H100 PCIe and H100 NVL variants have different memory and interconnect characteristics.` |
| Priority | **P0** |

---

## 2.4 H100 SXM5 SM Count

| Field | Validation |
|---|---|
| Claim | H100 SXM5 has 132 SMs. |
| Current value or formula | `132 Streaming Multiprocessors` |
| Validation status | **Valid for common H100 SXM5 80 GB configuration.** |
| Corrected value or safer wording | State SKU-specific. Do not imply every Hopper product has 132 SMs. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 architecture whitepaper / datasheet / official SKU table |
| Recommended final wording | `[SHIPPED] The H100 SXM5 80 GB configuration exposes 132 SMs. Treat SM count as SKU-specific when comparing PCIe, NVL, SXM, or future derivatives.` |
| Priority | **P0** |

---

## 2.5 H100 Shared Memory / L1 Details

| Field | Validation |
|---|---|
| Claim | H100 has up to 228 KB shared memory per SM and a configurable L1/shared-memory subsystem. |
| Current value or formula | `Up to 228 KB shared memory per SM`; L1/shared memory behavior is configuration-dependent. |
| Validation status | **Valid with careful wording.** |
| Corrected value or safer wording | Avoid publishing a single fixed L1 value without context. |
| Confidence label | `[SHIPPED]` for official capacity; `[ENV-SPECIFIC]` for performance behavior |
| Source type needed | NVIDIA Hopper architecture whitepaper / CUDA programming guide |
| Recommended final wording | `[SHIPPED] H100 provides a large per-SM shared-memory/L1 subsystem, with up to 228 KB of shared memory per SM depending on configuration. The effective behavior seen by a kernel depends on shared-memory allocation, cache configuration, access pattern, and compiler/kernel implementation.` |
| Priority | **P0** |

---

## 2.6 H100 L2 Cache Size

| Field | Validation |
|---|---|
| Claim | H100 SXM5 has 50 MB L2 cache. |
| Current value or formula | `50 MB L2` |
| Validation status | **Valid for H100 SXM5.** |
| Corrected value or safer wording | State as SKU-specific. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA Hopper architecture whitepaper / H100 technical specs |
| Recommended final wording | `[SHIPPED] H100 SXM5 includes 50 MB of unified L2 cache. L2 capacity should be treated as GPU/SKU-specific when comparing architectures.` |
| Priority | **P0** |

---

## 2.7 H100 Tensor Core Details and Datatype Support

| Field | Validation |
|---|---|
| Claim | H100 uses fourth-generation Tensor Cores and includes Transformer Engine support for FP8. |
| Current value or formula | H100 Tensor Core formats include FP64 Tensor Core, TF32, FP16, BF16, FP8, INT8, with structured sparsity paths where applicable. |
| Validation status | **Valid with datatype-specific and sparsity-specific wording.** |
| Corrected value or safer wording | Do not imply every datatype has the same peak throughput. Distinguish Tensor Core from non-Tensor Core paths. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 architecture whitepaper / datasheet / CUDA documentation |
| Recommended final wording | `[SHIPPED] H100 introduces fourth-generation Tensor Cores and Transformer Engine support for FP8. Supported accelerated paths include TF32, FP16, BF16, FP8, INT8, and FP64 Tensor Core modes, with structured sparsity increasing peak throughput for supported sparse workloads.` |
| Priority | **P0** |

---

## 2.8 H100 NVLink Bandwidth and Directionality

| Field | Validation |
|---|---|
| Claim | H100 SXM5 NVLink bandwidth is commonly listed as 900 GB/s per GPU. |
| Current value or formula | `NVLink: 900 GB/s` |
| Validation status | **Valid when stated as vendor-published aggregate per-GPU NVLink bandwidth.** |
| Corrected value or safer wording | Always state aggregate vs per-direction if known. Avoid mixing with PCIe or InfiniBand bandwidth. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 datasheet / HGX H100 documentation |
| Recommended final wording | `[SHIPPED] H100 SXM5 systems list up to 900 GB/s aggregate NVLink bandwidth per GPU. When comparing to PCIe or InfiniBand, always state whether the number is per-direction, bidirectional aggregate, or effective application bandwidth.` |
| Priority | **P0** |

---

## 2.9 PCIe 5.0 x16 Bandwidth and Directionality

| Field | Validation |
|---|---|
| Claim | PCIe 5.0 x16 provides about 128 GB/s theoretical aggregate bidirectional bandwidth. |
| Current value or formula | `32 GT/s per lane × 16 lanes`; approximately `64 GB/s per direction`, `128 GB/s aggregate bidirectional` after encoding assumptions. |
| Validation status | **Valid if directionality is clear.** |
| Corrected value or safer wording | Use per-direction and aggregate values separately. |
| Confidence label | `[DERIVED FROM SHIPPED]` or `[SHIPPED]` for standard speed |
| Source type needed | PCI-SIG specification / NVIDIA GPU datasheet |
| Recommended final wording | `[DERIVED FROM SHIPPED] PCIe 5.0 x16 is commonly treated as about 64 GB/s per direction, or about 128 GB/s aggregate bidirectional theoretical bandwidth. Sustained application bandwidth is lower and depends on platform, DMA path, NUMA placement, payload size, and driver/runtime behavior.` |
| Priority | **P0** |

---

## 2.10 InfiniBand NDR / 400G / 800G Bandwidth Wording

| Field | Validation |
|---|---|
| Claim | NDR InfiniBand provides 400 Gb/s links; newer Quantum-X800 / XDR-class products target 800 Gb/s. |
| Current value or formula | `400 Gb/s = 50 GB/s line rate`; `800 Gb/s = 100 GB/s line rate` before protocol overhead. |
| Validation status | **Valid when described as line rate and not application bandwidth.** |
| Corrected value or safer wording | Do not equate line rate with NCCL `busbw` or effective application bandwidth. |
| Confidence label | `[SHIPPED]` for NDR 400G; `[SHIPPED]` or `[ANNOUNCED]` for 800G depending on product status and publication date |
| Source type needed | NVIDIA/Mellanox InfiniBand product pages; switch/NIC datasheets |
| Recommended final wording | `[SHIPPED] NDR InfiniBand is commonly discussed as 400 Gb/s, or about 50 GB/s line rate per direction before protocol overhead. `[SHIPPED/ANNOUNCED]` 800 Gb/s InfiniBand/Ethernet products correspond to about 100 GB/s line rate per direction. Effective NCCL bandwidth is workload-, topology-, message-size-, and congestion-dependent.` |
| Priority | **P0** |

---

## 2.11 H200 HBM Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | H200 provides 141 GB HBM3e and 4.8 TB/s memory bandwidth. |
| Current value or formula | `141 GB HBM3e`, `4.8 TB/s` |
| Validation status | **Valid for NVIDIA H200 product spec.** |
| Corrected value or safer wording | Specify H200 and avoid mixing with H100 compute values unless noting same Hopper generation. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H200 product page / datasheet |
| Recommended final wording | `[SHIPPED] NVIDIA H200 provides 141 GB of HBM3e memory with 4.8 TB/s peak memory bandwidth. Compared with H100 SXM5 80 GB, the main production impact is larger and faster memory rather than a new compute architecture.` |
| Priority | **P0** |

---

## 2.12 MI300X Dense BF16 Peak TFLOPS

| Field | Validation |
|---|---|
| Claim | AMD Instinct MI300X dense BF16 peak is approximately 1,307.4 TFLOPS. |
| Current value or formula | AMD-published peak theoretical BF16 = `1307.4 TFLOPS`; with sparsity = `2614.9 TFLOPS`. |
| Validation status | **Valid if dense vs sparsity is distinguished.** |
| Corrected value or safer wording | Use “peak theoretical dense BF16” and avoid comparing to H100 sparse values. |
| Confidence label | `[SHIPPED]` |
| Source type needed | AMD MI300X product page / datasheet |
| Recommended final wording | `[SHIPPED] AMD Instinct MI300X peak theoretical dense BF16 performance is approximately 1,307.4 TFLOPS. With structured sparsity, AMD lists approximately 2,614.9 TFLOPS. Compare dense-to-dense and sparse-to-sparse values only.` |
| Priority | **P0** |

---

## 2.13 MI300X HBM Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | MI300X provides 192 GB HBM3 and approximately 5.3 TB/s peak memory bandwidth. |
| Current value or formula | `192 GB HBM3`; `5.325 TB/s` peak theoretical bandwidth |
| Validation status | **Valid.** |
| Corrected value or safer wording | Use `5.3 TB/s` in prose and `5.325 TB/s` in detailed tables. |
| Confidence label | `[SHIPPED]` |
| Source type needed | AMD MI300X product page / datasheet |
| Recommended final wording | `[SHIPPED] AMD Instinct MI300X provides 192 GB of HBM3 and approximately 5.3 TB/s peak memory bandwidth. Use 5.325 TB/s in detailed spec tables if precision is useful.` |
| Priority | **P0** |

---

## 2.14 MI300X Infinity Fabric Bandwidth Wording

| Field | Validation |
|---|---|
| Claim | MI300X supports 8 Infinity Fabric links, with 128 GB/s peak link bandwidth. |
| Current value or formula | `8 Infinity Fabric links`; `128 GB/s peak per link`; aggregate depends on topology and directionality. |
| Validation status | **Valid with careful wording.** |
| Corrected value or safer wording | Avoid converting to a single aggregate number unless source and directionality are explicit. |
| Confidence label | `[SHIPPED]` |
| Source type needed | AMD MI300X datasheet / platform datasheet |
| Recommended final wording | `[SHIPPED] MI300X exposes 8 Infinity Fabric links, with AMD listing 128 GB/s peak Infinity Fabric link bandwidth. When discussing aggregate GPU-to-GPU bandwidth, specify whether the value is per link, per direction, bidirectional, or full-platform aggregate.` |
| Priority | **P0** |

---

## 2.15 MI300X TDP / Power Claims

| Field | Validation |
|---|---|
| Claim | MI300X OAM accelerator has 750 W peak board power / TBP. |
| Current value or formula | `750 W peak` |
| Validation status | **Valid when described as board power/TBP for OAM module.** |
| Corrected value or safer wording | Use “typical board power / peak board power as published by AMD,” not “always consumes 750 W.” |
| Confidence label | `[SHIPPED]` |
| Source type needed | AMD MI300X product page / datasheet |
| Recommended final wording | `[SHIPPED] AMD lists MI300X OAM typical/peak board power around 750 W. Actual sustained power depends on workload, platform limits, cooling, firmware, and power-management settings.` |
| Priority | **P1** |

---

## 2.16 B200 / GB200 Values and Shipped vs Announced Status

| Field | Validation |
|---|---|
| Claim | B200 / GB200 Blackwell systems provide much larger memory, FP4/FP8 capability, NVLink 5, and rack-scale NVLink domains. |
| Current value or formula | Examples: DGX B200 lists 8 Blackwell GPUs, 1,440 GB total HBM3e, 64 TB/s HBM3e bandwidth, 14.4 TB/s aggregate NVLink bandwidth. DGX GB200 lists 72 Blackwell GPUs, up to 13.4 TB HBM3e, 576 TB/s GPU memory bandwidth, and rack-scale NVLink domain. |
| Validation status | **Valid if tied to specific official product page and publication date.** |
| Corrected value or safer wording | Do not mix B100, B200, GB200, GB300, and NVL72 values. Label status carefully. |
| Confidence label | `[SHIPPED]` if sourced from a shipping official product page; `[ANNOUNCED]` if roadmap/press/disclosed but not broadly shipping |
| Source type needed | NVIDIA B200/DGX B200/DGX GB200/GB200 NVL72 official product pages or datasheets |
| Recommended final wording | `[SHIPPED/ANNOUNCED] Blackwell values should be stated by product: B200 GPU, DGX B200 8-GPU system, GB200 Superchip, or GB200 NVL72 rack. Do not quote a single “B200 number” without saying which product and whether the number is per GPU, per system, or per rack.` |
| Priority | **P0** |

---

## 2.17 Tensor Core Datatype Support by Generation

| Field | Validation |
|---|---|
| Claim | Tensor Core datatype support evolves by generation: Volta → Ampere → Hopper → Blackwell. |
| Current value or formula | Volta introduced FP16 Tensor Cores; Ampere added TF32 and BF16 paths; Hopper added FP8 Transformer Engine; Blackwell adds second-generation Transformer Engine and FP4-related paths. |
| Validation status | **Directionally valid; every row needs official source validation.** |
| Corrected value or safer wording | Present as “representative accelerated formats,” not exhaustive ISA documentation. |
| Confidence label | `[SHIPPED]` for released generations; `[ANNOUNCED]` if future/roadmap |
| Source type needed | NVIDIA architecture whitepapers and CUDA documentation |
| Recommended final wording | `[SHIPPED] Tensor Core datatype support has expanded across generations: Volta introduced FP16 Tensor Cores; Ampere added TF32 and BF16-oriented AI paths; Hopper added FP8 Transformer Engine support; Blackwell adds newer Transformer Engine and FP4-related AI paths. Exact throughput and supported modes are architecture- and SKU-specific.` |
| Priority | **P0** |

### Production table recommendation

| Generation | Representative GPUs | Safe Production Wording |
|---|---|---|
| Volta | V100 | `[SHIPPED] Introduced Tensor Cores for FP16 mixed-precision matrix math.` |
| Ampere | A100 | `[SHIPPED] Added TF32 and BF16-oriented Tensor Core acceleration, plus sparsity paths.` |
| Hopper | H100/H200 | `[SHIPPED] Fourth-generation Tensor Cores and Transformer Engine with FP8 support.` |
| Blackwell | B100/B200/GB200 | `[SHIPPED/ANNOUNCED] Fifth-generation Tensor Cores / second-generation Transformer Engine with FP4-related AI paths, depending on product status.` |

---

## 2.18 Tensor Core Alignment and Fallback Wording

| Field | Validation |
|---|---|
| Claim | Misaligned shapes can reduce Tensor Core efficiency or cause fallback/padding behavior. |
| Current value or formula | Tensor Core kernels prefer supported datatypes, tile sizes, memory layouts, and aligned dimensions. |
| Validation status | **Valid as a practical rule, but implementation-dependent.** |
| Corrected value or safer wording | Do not say misalignment always falls back to CUDA cores. |
| Confidence label | `[REPRESENTATIVE]` or `[ENV-SPECIFIC]` |
| Source type needed | CUDA programming guide, cuBLAS/cuBLASLt docs, kernel profiler evidence |
| Recommended final wording | `[REPRESENTATIVE] Tensor Core efficiency depends on datatype, matrix dimensions, tile shape, memory layout, and kernel selection. Poorly aligned dimensions may trigger padding, a less efficient kernel path, or reduced Tensor Core utilization depending on framework, library, and hardware generation.` |
| Priority | **P0** |

---

## 2.19 SXM vs PCIe vs OAM Form Factor Claims

| Field | Validation |
|---|---|
| Claim | SXM/HGX and OAM platforms are usually preferred for dense multi-GPU training/serving because they support higher power/cooling envelopes and high-bandwidth GPU-to-GPU fabrics. PCIe is more flexible and often lower cost but usually has lower GPU-to-GPU bandwidth. |
| Current value or formula | SXM/HGX = high-bandwidth NVLink/NVSwitch systems; PCIe = broader deployment/flexible attachment; OAM = accelerator module form factor used by AMD platforms. |
| Validation status | **Valid if framed as workload-dependent.** |
| Corrected value or safer wording | Avoid “SXM is mandatory.” |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | Vendor platform specs; HGX/DGX docs; AMD OAM platform datasheets |
| Recommended final wording | `[REPRESENTATIVE] SXM/HGX-class and OAM-style platforms are usually preferred when the workload requires dense multi-GPU topology, high power delivery, high cooling capacity, and high-bandwidth GPU-to-GPU communication. PCIe systems remain valuable for development, single-GPU inference, smaller models, cost-sensitive deployments, and workloads that do not depend on tight collectives.` |
| Priority | **P0** |

---

## 2.20 “Never TP Across InfiniBand” and Other Topology Guidance

| Field | Validation |
|---|---|
| Claim | Tensor parallelism should not be extended across InfiniBand. |
| Current value or formula | Current wording may be too absolute. |
| Validation status | **Directionally useful but too absolute for production.** |
| Corrected value or safer wording | Replace “never” with topology- and workload-aware guidance. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | Megatron/NCCL scaling docs, distributed training papers, internal/profiler results if available |
| Recommended final wording | `[REPRESENTATIVE] Keep latency-sensitive tensor-parallel groups within the highest-bandwidth local GPU fabric whenever possible. Avoid extending TP across InfiniBand unless model partitioning, collective scheduling, overlap strategy, and network topology are explicitly designed and validated for it. Use pipeline/data/context parallelism across nodes more commonly, but always verify with measured communication traces.` |
| Priority | **P0** |

### Related safer topology wording

| Avoid | Use |
|---|---|
| “Never TP across IB.” | “Avoid latency-sensitive TP across inter-node fabric unless validated.” |
| “SXM is mandatory.” | “SXM/HGX is usually preferred for high-performance multi-GPU training/serving.” |
| “PCIe is too slow for AI.” | “PCIe can be excellent when the workload is not dominated by tight GPU-to-GPU collectives.” |

---

## 2.21 Accelerator Selection Claims: NVIDIA vs AMD vs TPU vs Gaudi

| Field | Validation |
|---|---|
| Claim | Accelerator families differ by compute capability, memory capacity, interconnect, software stack, deployment model, and operational maturity. |
| Current value or formula | NVIDIA = CUDA/NVLink ecosystem; AMD = large HBM/ROCm/CDNA; TPU = cloud-integrated systolic-array approach; Gaudi = Ethernet-oriented AI accelerator approach. |
| Validation status | **Valid if neutral and workload-aware.** |
| Corrected value or safer wording | Do not declare a universal winner. Do not make unsourced software-maturity claims too strongly. |
| Confidence label | `[REPRESENTATIVE]`; specific product values `[SHIPPED]` or `[ANNOUNCED]` |
| Source type needed | Vendor architecture docs, model benchmark results, framework support matrices |
| Recommended final wording | `[REPRESENTATIVE] Accelerator choice should be workload-driven. NVIDIA GPUs often provide the broadest CUDA/NVLink software ecosystem. AMD Instinct systems can offer strong memory capacity and bandwidth, with ROCm/HIP as the software path. TPUs are tightly integrated with cloud/provider software stacks. Gaudi-style systems emphasize Ethernet-oriented accelerator fabrics. The correct choice depends on model fit, precision support, interconnect, kernel maturity, framework support, team expertise, power/cooling, cost, and supply risk.` |
| Priority | **P1** |

---

## 2.22 Any Claim That Needs Confidence Labels

| Claim Type | Required Label | Example |
|---|---|---|
| H100/H200/MI300X official specs | `[SHIPPED]` | H200 has 141 GB HBM3e and 4.8 TB/s bandwidth |
| Dense values derived from sparse specs | `[DERIVED FROM SHIPPED]` | H100 dense BF16 ≈ 1,978.9 / 2 |
| Blackwell roadmap or non-verified deployment claims | `[ANNOUNCED]` | Future rack-scale configuration claims |
| Sustained throughput | `[ENV-SPECIFIC]` | “600–700 TFLOPS sustained” |
| Workload bottleneck guidance | `[REPRESENTATIVE]` | “Decode is often memory-bandwidth sensitive” |
| Topology rules | `[REPRESENTATIVE]` | “Keep TP within NVLink domain when possible” |
| Kernel behavior | `[ENV-SPECIFIC]` | “This shape falls back to a slower kernel” |
| Spec-sheet comparison | `[SHIPPED]` + `[REPRESENTATIVE]` | Official values plus workload interpretation |

---

# 3. Corrected / Safer Wording Blocks

## 3.1 H100 BF16 Peak

```markdown
[DERIVED FROM SHIPPED] For dense/non-sparse BF16 Tensor Core math, H100 SXM5 peak is approximately 989.4 TFLOPS. Vendor tables often list approximately 1,978.9 TFLOPS for BF16 with structured sparsity. Always distinguish dense from sparse when using H100 peak numbers.
```

## 3.2 H100 Memory

```markdown
[SHIPPED] H100 SXM5 80 GB provides 80 GB of HBM3 with 3.35 TB/s peak memory bandwidth. This bandwidth is often more important than peak TFLOPS for memory-bound decode, normalization, KV-cache reads, and low-arithmetic-intensity kernels.
```

## 3.3 H200 Memory

```markdown
[SHIPPED] H200 keeps the Hopper compute generation but increases the memory footprint to 141 GB HBM3e and 4.8 TB/s peak memory bandwidth. For LLM inference, this primarily changes memory-bound capacity and KV-cache economics.
```

## 3.4 MI300X Memory and Compute

```markdown
[SHIPPED] AMD Instinct MI300X provides 192 GB HBM3 and approximately 5.3 TB/s peak memory bandwidth. AMD lists peak theoretical dense BF16 at approximately 1,307.4 TFLOPS and sparse BF16 at approximately 2,614.9 TFLOPS.
```

## 3.5 PCIe Directionality

```markdown
[DERIVED FROM SHIPPED] PCIe 5.0 x16 is commonly treated as approximately 64 GB/s per direction, or approximately 128 GB/s aggregate bidirectional theoretical bandwidth. Sustained bandwidth depends on platform topology, NUMA placement, payload size, DMA path, and software stack.
```

## 3.6 InfiniBand Directionality

```markdown
[SHIPPED] NDR InfiniBand is commonly described as 400 Gb/s, or about 50 GB/s line rate per direction before protocol overhead. Effective NCCL throughput is lower and depends on message size, topology, congestion, routing, and overlap.
```

## 3.7 Topology Guidance

```markdown
[REPRESENTATIVE] Keep latency-sensitive tensor-parallel groups within the highest-bandwidth local GPU fabric whenever possible. Avoid extending TP across inter-node InfiniBand unless the model partitioning, collective schedule, overlap strategy, and network topology are explicitly designed and measured for it.
```

## 3.8 Form Factor

```markdown
[REPRESENTATIVE] SXM/HGX-class and OAM-style systems are usually preferred when dense multi-GPU communication, high power delivery, and high cooling capacity are required. PCIe systems remain valuable for development, single-GPU inference, smaller models, and workloads that do not depend on tight GPU-to-GPU collectives.
```

## 3.9 Accelerator Selection

```markdown
[REPRESENTATIVE] Accelerator choice is workload-specific. Compare model fit in memory, supported precision, Tensor Core or matrix-engine paths, HBM bandwidth, interconnect, framework/kernel maturity, power/cooling, cost, supply, and operational risk. Peak TFLOPS alone is not a system architecture decision.
```

---

# 4. P0 / P1 / P2 Validation Action List

## P0 — Must Fix Before Chapter 3A Production Source

| Task | Action |
|---|---|
| Correct H100 BF16 dense vs sparse wording | Use 989.4 dense/non-sparse and 1,978.9 sparse |
| Validate H100 HBM capacity/bandwidth | Use 80 GB, 3.35 TB/s for H100 SXM5 |
| Validate H100 SM count, L2, shared memory/L1 details | Add SKU-specific notes |
| Validate H100 Tensor Core datatype support | Distinguish TC vs non-TC and dense vs sparse |
| Validate H100 NVLink directionality | State aggregate/per-GPU wording |
| Clarify PCIe 5.0 x16 directionality | 64 GB/s per direction, 128 GB/s aggregate bidirectional theoretical |
| Clarify InfiniBand line-rate vs effective bandwidth | NDR 400G = 50 GB/s line-rate per direction |
| Validate H200 HBM capacity/bandwidth | Use 141 GB, 4.8 TB/s |
| Validate MI300X dense/sparse BF16 and memory | Use AMD-published values |
| Avoid universal Blackwell numbers | Tie B200/GB200 values to exact product |
| Replace absolute topology wording | Use workload-aware wording |
| Add confidence labels to every numeric hardware claim | Required before production Markdown |

## P1 — Strongly Recommended

| Task | Action |
|---|---|
| Add Tensor Core generation table | Use official NVIDIA architecture docs |
| Add accelerator selection matrix | Use neutral workload-based criteria |
| Add form-factor decision framework | PCIe vs SXM/HGX vs OAM |
| Add “current as of” note | Hardware tables age quickly |
| Add source-note table | Helps annual refresh |
| Cross-reference Appendix A | Keep hardware specs centralized |
| Cross-reference Ch04/Ch05/Ch10/Ch14 | Memory, power, distributed systems, networking |

## P2 — Nice to Have

| Task | Action |
|---|---|
| Add appendix for detailed H100 Tensor Core derivation | Avoid confusing the main chapter |
| Add GPU spec sheet worksheet | Useful for interviews |
| Add yearly refresh checklist | Future edition maintenance |
| Add model-to-accelerator examples | H100 vs H200 vs MI300X for 70B serving |
| Add LinkedIn visual from spec-sheet decision tree | Marketing asset |

---

# 5. Source Categories for Final Book

Use these source categories when generating the final production chapter:

| Source Category | Use |
|---|---|
| NVIDIA H100 datasheet / product page | H100 memory, bandwidth, Tensor Core sparse values, NVLink |
| NVIDIA Hopper architecture whitepaper | SM details, Tensor Core generation, shared memory/L1, L2, execution model |
| NVIDIA H200 product page / datasheet | H200 HBM3e capacity and bandwidth |
| NVIDIA Blackwell / DGX B200 / DGX GB200 pages | B200/GB200 values and shipped/announced status |
| AMD MI300X product page / datasheet | MI300X BF16, HBM, Infinity Fabric, power |
| PCI-SIG PCIe 5.0 specification | PCIe speed and generation |
| NVIDIA Quantum / ConnectX datasheets | InfiniBand 400G/800G wording |
| CUDA Programming Guide | SIMT, warp, block, shared memory, Tensor Core alignment |
| cuBLAS / cuBLASLt documentation | Tensor Core kernel selection and layout/alignment behavior |
| NCCL documentation | Topology-aware collectives and bus bandwidth interpretation |
| Vendor framework support matrices | Accelerator selection and software maturity |

---

# 6. Chapter 3A Production Rules

1. Do not publish hardware numbers without confidence labels.
2. Do not mix dense and sparse peak TFLOPS.
3. Do not mix per-GPU, per-system, per-rack, and aggregate values.
4. Do not mix line rate and sustained application bandwidth.
5. Do not say “always” or “never” for topology choices unless constrained.
6. Do not rank accelerators without workload context.
7. Do not let spec tables replace the engineering lesson.
8. Keep official values in Appendix A and use Chapter 3A for mental models.
9. Include “current as of 2026” near hardware tables.
10. Plan for annual hardware table refresh.

---

# 7. Commit Instructions

Save this file as:

```text
publishing/validation/ch03a_technical_validation.md
```

Then run:

```powershell
git add publishing\validation\ch03a_technical_validation.md
git commit -m "Add Chapter 3A technical validation plan"
git push origin production-v1.0
```

---

# 8. Next Production Step

After committing this validation file, the next production task is:

```text
Create source/chapters/ch03a_gpu_architecture.md
```

That source chapter should incorporate:

1. Correct dense vs sparse H100 BF16 values.
2. H100/H200/MI300X memory and bandwidth values with confidence labels.
3. Figure placeholders from the Chapter 3A figure integration plan.
4. Tables 3A.1 through 3A.6.
5. Safer topology wording.
6. Neutral accelerator selection framework.
7. Principal interview explanation section.
8. Key takeaways and review questions.
