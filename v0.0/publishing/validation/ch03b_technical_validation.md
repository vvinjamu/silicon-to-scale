# Chapter 3B Technical Validation Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Ch03B — *GPU Architecture Roadmap: NVIDIA and AMD Generations*  
**Target file:** `publishing/validation/ch03b_technical_validation.md`  
**Production status:** Draft validation plan for `production-v1.0`  
**Last reviewed:** 2026-04-30  
**Purpose:** Validate GPU roadmap claims, shipped-vs-announced labels, product-level distinctions, memory/interconnect values, and safe wording before producing the Chapter 3B production Markdown source.

---

## 0. Executive Summary

Chapter 3B is a high-value but high-risk roadmap chapter. The core risk is not only numeric error. The larger risk is **mixing product levels and confidence levels**:

```text
GPU-level value
system-level value
rack-level value
line-rate bandwidth
effective workload bandwidth
shipping product spec
vendor-announced roadmap
benchmark-specific result
engineering estimate
```

The production version of Ch03B must enforce the following rules:

1. **Never compare dense peak to sparse peak.**
2. **Never compare a per-GPU number to a system-level or rack-level aggregate number.**
3. **Never compare line rate to effective application throughput.**
4. **Never treat an announced roadmap as a shipping product spec.**
5. **Never choose hardware from TFLOPS alone.**
6. **Always label NVIDIA/AMD benchmark claims as workload- and environment-specific unless they are pure official specs.**

The most important chapter-level corrections:

- A100 80GB SXM memory bandwidth is approximately **2,039 GB/s**, while A100 80GB PCIe is approximately **1,935 GB/s**. Use SKU-specific wording.
- A100 BF16/FP16 Tensor Core values are commonly listed as **312 TFLOPS dense** and **624 TFLOPS with sparsity**.
- H100 SXM5 dense/non-sparse BF16 should remain approximately **989.4 TFLOPS**, while H100 sparse BF16 is approximately **1,978.9 TFLOPS**.
- H100 SXM5 memory should remain **80 GB HBM3, 3.35 TB/s**.
- H200 should be presented as a **Hopper-generation memory refresh** with **141 GB HBM3e and 4.8 TB/s**, not as a brand-new architecture.
- DGX B200 is a **system-level product**: official pages list **8 Blackwell GPUs, 1,440 GB total GPU memory, 64 TB/s HBM3e bandwidth, and 14.4 TB/s aggregate NVLink bandwidth**.
- GB200 NVL72 is a **rack-scale platform**: official pages describe **36 Grace CPUs and 72 Blackwell GPUs in a 72-GPU NVLink domain**.
- GB200 Superchip is a **module/superchip level**: official pages describe **one Grace CPU plus two Blackwell GPUs** connected through fifth-generation NVLink.
- MI300X should remain **192 GB HBM3, 5.3 TB/s, dense BF16 ≈1,307.4 TFLOPS**.
- MI325X should be treated carefully because AMD wording includes “will have” and “actual results may vary” in some official material. Use `[ANNOUNCED]` or `[SHIPPED]` based on the source status used at publication time.
- MI350 Series / MI350X / MI355X should be product-specific; AMD official pages list **288 GB HBM3e and 8 TB/s** for MI350 series/MI350X-class pages.
- MI400 / Helios should remain **roadmap / announced**, not final product spec.

---

# 1. Confidence Label Policy for Ch03B

| Label | Use in Ch03B |
|---|---|
| `[SHIPPED]` | Official shipping product specification or product page value |
| `[ANNOUNCED]` | Vendor-announced roadmap, future platform, preview, or “will have” claim |
| `[DERIVED FROM SHIPPED]` | Arithmetic derived from official shipping specs |
| `[ESTIMATED]` | Simplified engineering estimate or planning calculation |
| `[REPRESENTATIVE]` | Directional architecture guidance that depends on workload |
| `[ENV-SPECIFIC]` | Benchmark result, measured deployment behavior, MLPerf result, cloud result, or vendor performance multiplier |

Production rule:

```text
Attach the label to the claim, not the paragraph.
```

Good:

```text
[SHIPPED] DGX B200 is an 8-GPU system with 1,440 GB total GPU memory and 64 TB/s total HBM3e bandwidth.
```

Avoid:

```text
B200 has 1,440 GB memory.
```

---

# 2. Validation Table

---

## 2.1 A100 80GB HBM2e Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | NVIDIA A100 80GB uses HBM2e and provides around 2 TB/s memory bandwidth. |
| Current value or formula | A100 80GB PCIe: 80 GB HBM2e, 1,935 GB/s. A100 80GB SXM: 80 GB HBM2e, 2,039 GB/s. |
| Validation status | **Valid when SKU-specific.** |
| Corrected value or safer wording | Do not write one A100 bandwidth value without identifying PCIe vs SXM. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA A100 product page / datasheet / Ampere architecture whitepaper |
| Recommended final wording for the book | `[SHIPPED] NVIDIA A100 80GB uses HBM2e memory. The 80GB PCIe SKU is commonly listed at 1,935 GB/s memory bandwidth, while the 80GB SXM SKU is listed at 2,039 GB/s. Use SKU-specific values when comparing A100 to H100/H200/Blackwell or AMD Instinct accelerators.` |
| Priority | **P0** |

---

## 2.2 A100 Dense/Sparse BF16 or FP16 Tensor Core Values If Referenced

| Field | Validation |
|---|---|
| Claim | A100 BF16/FP16 Tensor Core peak is approximately 312 TFLOPS dense and 624 TFLOPS with sparsity. |
| Current value or formula | A100 BF16 Tensor Core: 312 TFLOPS / 624 TFLOPS with sparsity; FP16 Tensor Core: 312 TFLOPS / 624 TFLOPS with sparsity. |
| Validation status | **Valid if dense vs sparsity is clearly distinguished.** |
| Corrected value or safer wording | Use “with sparsity” for 624 TFLOPS. Do not compare A100 sparse peak to H100 dense peak. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA A100 product page / Ampere architecture whitepaper |
| Recommended final wording for the book | `[SHIPPED] A100 80GB lists BF16 and FP16 Tensor Core peak performance around 312 TFLOPS dense and 624 TFLOPS with structured sparsity. Compare dense-to-dense and sparse-to-sparse values only.` |
| Priority | **P0** |

---

## 2.3 H100 SXM5 HBM Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | H100 SXM5 has 80 GB HBM3 and 3.35 TB/s memory bandwidth. |
| Current value or formula | 80 GB HBM3, 3.35 TB/s. |
| Validation status | **Valid for H100 SXM5 80 GB.** |
| Corrected value or safer wording | Specify H100 SXM5 80 GB; do not apply to H100 NVL or PCIe variants. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 product page / H100 datasheet |
| Recommended final wording for the book | `[SHIPPED] H100 SXM5 80 GB provides 80 GB of HBM3 and 3.35 TB/s peak memory bandwidth. Other H100 variants have different memory and interconnect characteristics.` |
| Priority | **P0** |

---

## 2.4 H100 Dense BF16 vs Sparse BF16 Peak Values

| Field | Validation |
|---|---|
| Claim | H100 SXM dense BF16 ≈989.4 TFLOPS; sparse BF16 ≈1,978.9 TFLOPS. |
| Current value or formula | Dense BF16 ≈ sparse BF16 / 2 ≈ 1,978.9 / 2 = 989.45 TFLOPS. |
| Validation status | **Valid when dense and sparse are distinguished.** |
| Corrected value or safer wording | Always state “dense/non-sparse” or “with structured sparsity.” |
| Confidence label | `[DERIVED FROM SHIPPED]` for dense if derived from sparse listing; `[SHIPPED]` for sparse vendor listing |
| Source type needed | NVIDIA H100 product page / datasheet |
| Recommended final wording for the book | `[DERIVED FROM SHIPPED] H100 SXM5 dense/non-sparse BF16 Tensor Core peak is approximately 989.4 TFLOPS. `[SHIPPED]` The commonly listed approximately 1,978.9 TFLOPS BF16 value is the sparse Tensor Core peak and should not be used as the denominator for dense MFU.` |
| Priority | **P0** |

---

## 2.5 H100 NVLink 4 Bandwidth and Directionality

| Field | Validation |
|---|---|
| Claim | H100 SXM5 uses fourth-generation NVLink and is commonly listed at up to 900 GB/s aggregate NVLink bandwidth per GPU. |
| Current value or formula | Up to 900 GB/s aggregate per GPU. |
| Validation status | **Valid with aggregate/per-GPU wording.** |
| Corrected value or safer wording | Avoid saying “900 GB/s per direction” unless sourced. Use “aggregate per GPU.” |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 product page / H100 datasheet / HGX H100 docs |
| Recommended final wording for the book | `[SHIPPED] H100 SXM5 systems list up to 900 GB/s aggregate NVLink bandwidth per GPU using fourth-generation NVLink. When comparing to PCIe or InfiniBand, state whether the value is per direction, bidirectional aggregate, per GPU, per system, or effective workload bandwidth.` |
| Priority | **P0** |

---

## 2.6 H200 HBM3e Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | H200 provides 141 GB HBM3e and 4.8 TB/s memory bandwidth. |
| Current value or formula | 141 GB HBM3e, 4.8 TB/s. |
| Validation status | **Valid.** |
| Corrected value or safer wording | Specify H200 product; avoid mixing H200 with H100 memory values. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H200 product page / datasheet |
| Recommended final wording for the book | `[SHIPPED] NVIDIA H200 provides 141 GB of HBM3e and 4.8 TB/s peak memory bandwidth. In production reasoning, H200 is most important as a larger/faster-memory Hopper-generation part for memory-bound inference and larger KV-cache headroom.` |
| Priority | **P0** |

---

## 2.7 H200 as Memory Refresh vs New Architecture Wording

| Field | Validation |
|---|---|
| Claim | H200 is mainly a memory refresh rather than a fundamentally new architecture. |
| Current value or formula | Hopper-generation accelerator with HBM3e memory expansion. |
| Validation status | **Directionally valid.** |
| Corrected value or safer wording | Say “Hopper-generation memory refresh” rather than “same GPU” in an absolute sense. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | NVIDIA H200 product materials and H100/H200 architecture comparison |
| Recommended final wording for the book | `[REPRESENTATIVE] H200 should be treated primarily as a Hopper-generation memory refresh: the major practical change versus H100 SXM5 is larger and faster HBM3e, which improves memory-bound inference and KV-cache capacity economics.` |
| Priority | **P1** |

---

## 2.8 B200 Per-GPU Values and Shipped/Announced Status

| Field | Validation |
|---|---|
| Claim | B200 per-GPU values include Blackwell generation, HBM3e, FP4/FP8 Tensor Core capability, and NVLink 5 class connectivity. |
| Current value or formula | Common derived/third-party values include around 180–192 GB HBM3e and 8 TB/s per GPU, but official product pages often present DGX/HGX/system-level numbers. |
| Validation status | **Must be product-source-specific.** |
| Corrected value or safer wording | Avoid publishing per-GPU B200 numbers unless sourced from an official B200/HGX B200 datasheet. If derived from DGX B200 total, label as derived and explain. |
| Confidence label | `[SHIPPED]` if official product table; `[DERIVED FROM SHIPPED]` if derived from DGX B200 total; `[ANNOUNCED]` for roadmap/preview claims |
| Source type needed | NVIDIA Blackwell/B200/HGX B200/DGX B200 official product pages or datasheets |
| Recommended final wording for the book | `[SHIPPED/DERIVED FROM SHIPPED] Treat B200 values as product-specific. DGX B200 official system specs list 8 Blackwell GPUs, 1,440 GB total GPU memory, and 64 TB/s total HBM3e bandwidth. If deriving a per-GPU estimate from those system totals, state the derivation and do not mix it with rack-level GB200 NVL72 values.` |
| Priority | **P0** |

---

## 2.9 DGX B200 System-Level Values

| Field | Validation |
|---|---|
| Claim | DGX B200 is an 8-GPU Blackwell system with 1,440 GB total GPU memory, 64 TB/s HBM3e bandwidth, 144 PFLOPS FP4, 72 PFLOPS FP8, and 14.4 TB/s aggregate NVLink bandwidth. |
| Current value or formula | Official DGX B200 system specs list 8 NVIDIA Blackwell GPUs, 1,440 GB total memory, 64 TB/s HBM3e bandwidth, FP4 Tensor Core 144 PFLOPS / 72 PFLOPS with footnote, FP8 Tensor Core 72 PFLOPS, 14.4 TB/s aggregate NVLink bandwidth. |
| Validation status | **Valid as system-level DGX B200 values.** |
| Corrected value or safer wording | Must say “DGX B200 system-level.” |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA DGX B200 product page / datasheet |
| Recommended final wording for the book | `[SHIPPED] DGX B200 is a system-level product with 8 NVIDIA Blackwell GPUs, 1,440 GB total GPU memory, 64 TB/s total HBM3e bandwidth, and 14.4 TB/s aggregate NVLink bandwidth. These are system-level values and should not be quoted as per-GPU values.` |
| Priority | **P0** |

---

## 2.10 GB200 Superchip Values

| Field | Validation |
|---|---|
| Claim | GB200 Superchip integrates one Grace CPU and two Blackwell GPUs connected with fifth-generation NVLink. |
| Current value or formula | One Grace CPU + two Blackwell GPUs; fifth-generation NVLink; 1.8 TB/s GPU-to-GPU bandwidth commonly listed for DGX GB200 / GB200 Superchip context. |
| Validation status | **Valid with product-level wording.** |
| Corrected value or safer wording | Say “GB200 Superchip” rather than B200 GPU. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA DGX GB200 / GB200 Superchip product page |
| Recommended final wording for the book | `[SHIPPED] The NVIDIA GB200 Superchip combines one Grace CPU and two Blackwell GPUs. NVIDIA describes the GB200 Superchips as connected through fifth-generation NVLink, with 1.8 TB/s GPU-to-GPU bandwidth in the GB200 context. Treat this as a superchip/module-level value, not a standalone GPU value.` |
| Priority | **P0** |

---

## 2.11 GB200 NVL72 Rack-Level Values and Product-Level Wording

| Field | Validation |
|---|---|
| Claim | GB200 NVL72 connects 36 Grace CPUs and 72 Blackwell GPUs in a 72-GPU NVLink domain. |
| Current value or formula | 36 Grace CPUs, 72 Blackwell GPUs, 72-GPU NVLink domain, liquid-cooled rack-scale design. |
| Validation status | **Valid as rack-scale product wording.** |
| Corrected value or safer wording | Must say rack-level/rack-scale. Avoid comparing to single-GPU H100/H200 values. |
| Confidence label | `[SHIPPED]` if official product page is the source |
| Source type needed | NVIDIA GB200 NVL72 official product page / DGX GB200 product page |
| Recommended final wording for the book | `[SHIPPED] GB200 NVL72 is a rack-scale platform connecting 36 Grace CPUs and 72 Blackwell GPUs in a 72-GPU NVLink domain. Its memory, bandwidth, and performance claims are rack-level or platform-level and must not be compared directly to single-GPU values.` |
| Priority | **P0** |

---

## 2.12 NVLink 5 Bandwidth and Directionality

| Field | Validation |
|---|---|
| Claim | NVLink 5 / fifth-generation NVLink provides 1.8 TB/s GPU-to-GPU bandwidth in GB200/DGX GB200 contexts. |
| Current value or formula | 1.8 TB/s GPU-to-GPU bandwidth. |
| Validation status | **Valid when tied to GB200 context and source.** |
| Corrected value or safer wording | State product context and scope. Avoid applying the same number to all Blackwell products. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA DGX GB200 / GB200 product page |
| Recommended final wording for the book | `[SHIPPED] In the GB200 context, NVIDIA describes fifth-generation NVLink as providing 1.8 TB/s GPU-to-GPU bandwidth. For roadmap comparisons, always state product context, scope, and whether the number is per GPU, GPU-to-GPU, per superchip, or aggregate.` |
| Priority | **P0** |

---

## 2.13 Rubin / Vera Rubin Roadmap Wording

| Field | Validation |
|---|---|
| Claim | NVIDIA Rubin / Vera Rubin is a future-generation roadmap platform following Blackwell. |
| Current value or formula | Rubin / Vera Rubin platform claims include future CPU/GPU/NVLink generation and roadmap performance improvements. |
| Validation status | **Roadmap/announced; not final production spec unless official product page lists a shipping SKU.** |
| Corrected value or safer wording | Treat as `[ANNOUNCED]`; avoid final procurement-ready specs. |
| Confidence label | `[ANNOUNCED]` |
| Source type needed | NVIDIA official roadmap, GTC/CES announcements, HGX Rubin page if official |
| Recommended final wording for the book | `[ANNOUNCED] Rubin / Vera Rubin should be treated as NVIDIA’s future roadmap direction beyond Blackwell. Discuss it as a planning signal — not as a procurement-ready spec — unless official product-level specifications and availability are verified for the edition date.` |
| Priority | **P0** |

---

## 2.14 MI300X HBM Capacity, Bandwidth, Dense BF16, Sparse BF16, and Chiplet Details

| Field | Validation |
|---|---|
| Claim | AMD Instinct MI300X provides 192 GB HBM3, 5.3 TB/s memory bandwidth, dense BF16 ≈1,307.4 TFLOPS, sparse BF16 ≈2,614.9 TFLOPS, and chiplet-based architecture. |
| Current value or formula | 192 GB HBM3, 5.3 TB/s, dense BF16 1,307.4 TFLOPS, sparse BF16 2,614.9 TFLOPS. Die stack often described as 6 XCDs plus 4 I/O/cache/memory dies depending source. |
| Validation status | **Valid for memory and BF16 values from AMD specs. Chiplet/die-count wording must follow AMD source terminology.** |
| Corrected value or safer wording | Avoid overclaiming “single-GPU 70B BF16 serving” unless accounting for KV/runtime overhead. |
| Confidence label | `[SHIPPED]` for official specs; `[ESTIMATED]` for model-fit examples |
| Source type needed | AMD MI300X product page / MI300X datasheet / AMD architecture materials |
| Recommended final wording for the book | `[SHIPPED] AMD Instinct MI300X provides 192 GB HBM3 and approximately 5.3 TB/s peak memory bandwidth. AMD lists peak dense BF16 at approximately 1,307.4 TFLOPS and sparse BF16 at approximately 2,614.9 TFLOPS. `[REPRESENTATIVE]` Its large HBM footprint is especially relevant for memory-heavy inference and KV-cache-heavy workloads, provided the software stack fits the deployment.` |
| Priority | **P0** |

---

## 2.15 MI325X HBM3e Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | AMD Instinct MI325X provides 256 GB HBM3e and 6 TB/s peak memory bandwidth. |
| Current value or formula | 256 GB HBM3e, 6 TB/s. |
| Validation status | **Official AMD pages describe these values, but some wording says “will have” and “actual results may vary.” Confirm edition-time status.** |
| Corrected value or safer wording | Label `[SHIPPED]` only if using final product/spec page; otherwise `[ANNOUNCED]`. |
| Confidence label | `[SHIPPED]` or `[ANNOUNCED]` depending source status at publication |
| Source type needed | AMD MI325X product page / datasheet |
| Recommended final wording for the book | `[SHIPPED/ANNOUNCED] AMD Instinct MI325X is a memory-focused refresh with 256 GB HBM3e and approximately 6 TB/s peak theoretical memory bandwidth. Use `[SHIPPED]` only when referencing a final product page or datasheet; use `[ANNOUNCED]` if the source language is forward-looking.` |
| Priority | **P0** |

---

## 2.16 MI350X / MI355X HBM Capacity, Bandwidth, CDNA 4, and Precision-Format Claims

| Field | Validation |
|---|---|
| Claim | AMD Instinct MI350 Series / MI350X / MI355X uses CDNA 4, 288 GB HBM3e, 8 TB/s peak bandwidth, and next-generation low-precision formats such as MXFP4/MXFP6. |
| Current value or formula | AMD MI350 Series official pages list 288 GB HBM3e and 8 TB/s; CDNA 4 whitepaper and MI350 Series pages discuss MXFP6/MXFP4 datatype support. |
| Validation status | **Valid with product-specific wording.** |
| Corrected value or safer wording | Do not use MI350 Series values as if they apply identically to every SKU unless source confirms. |
| Confidence label | `[SHIPPED]` if official product page; `[ANNOUNCED]` for launch/roadmap-only claims |
| Source type needed | AMD MI350 Series product page, MI350X/MI355X product pages, AMD CDNA 4 architecture whitepaper |
| Recommended final wording for the book | `[SHIPPED] AMD Instinct MI350 Series product materials list 288 GB HBM3e and 8 TB/s peak memory bandwidth, with CDNA 4 architecture and next-generation MXFP datatype support. Use product-specific wording for MI350X vs MI355X and verify each SKU before final publication.` |
| Priority | **P0** |

---

## 2.17 MI400 / Helios Roadmap Wording

| Field | Validation |
|---|---|
| Claim | AMD MI400 / Helios is a future rack-scale platform direction. |
| Current value or formula | AMD has previewed Helios as a rack-scale platform based on future MI400-series GPUs, EPYC Venice CPUs, and Pensando Vulcano networking; some materials mention 2026 and performance projections. |
| Validation status | **Roadmap / announced. Not final product spec.** |
| Corrected value or safer wording | Treat as roadmap signal and use `[ANNOUNCED]`. Avoid final TFLOPS/HBM specs unless official product page exists. |
| Confidence label | `[ANNOUNCED]` |
| Source type needed | AMD official roadmap/blog/press materials |
| Recommended final wording for the book | `[ANNOUNCED] AMD Helios / MI400 should be discussed as a future rack-scale roadmap direction, not a shipping product specification, unless official product-level datasheets exist at publication time. Use it to explain AMD’s direction toward rack-scale AI infrastructure, not to make final procurement comparisons.` |
| Priority | **P0** |

---

## 2.18 HBM2e → HBM3 → HBM3e → HBM4 Evolution Claims

| Field | Validation |
|---|---|
| Claim | AI accelerator roadmaps increasingly move from HBM2e to HBM3, HBM3e, and HBM4 to improve memory capacity and bandwidth. |
| Current value or formula | A100: HBM2e; H100/MI300X: HBM3; H200/B200/MI325X/MI350 Series: HBM3e; HBM4 appears in future roadmap discussions. |
| Validation status | **Valid as a directional roadmap, but exact values are product-specific.** |
| Corrected value or safer wording | Do not assign one fixed bandwidth to an HBM generation; tie bandwidth to product/SKU. |
| Confidence label | `[REPRESENTATIVE]` for trend; `[SHIPPED]` for product-specific values; `[ANNOUNCED]` for HBM4 roadmap |
| Source type needed | Vendor product pages, JEDEC/HBM standards, roadmap materials |
| Recommended final wording for the book | `[REPRESENTATIVE] AI accelerator roadmaps increasingly depend on HBM capacity and bandwidth. A100-era systems used HBM2e; H100/MI300X-class accelerators use HBM3; H200, B200/GB200, MI325X, and MI350-class products use HBM3e. HBM4 should be treated as roadmap/future until tied to official product specifications.` |
| Priority | **P1** |

---

## 2.19 NVLink / NVSwitch / Infinity Fabric Evolution Claims

| Field | Validation |
|---|---|
| Claim | Interconnect evolution expands the efficient scale-up communication domain from node-level to rack-scale systems. |
| Current value or formula | NVLink 4 on H100/H200, NVLink 5 in Blackwell/GB200 context, NVSwitch systems, AMD Infinity Fabric for MI300/MI350 platforms. |
| Validation status | **Valid directionally; numeric values need product and directionality context.** |
| Corrected value or safer wording | Avoid direct number comparison unless scope/directionality matches. |
| Confidence label | `[REPRESENTATIVE]` for trend; `[SHIPPED]` for official product values |
| Source type needed | NVIDIA NVLink/HGX/DGX docs, AMD Infinity Fabric platform docs |
| Recommended final wording for the book | `[REPRESENTATIVE] Interconnect roadmaps are about expanding the efficient communication domain. NVLink/NVSwitch and AMD Infinity Fabric should be evaluated by scope: per link, per GPU, per node, per rack, aggregate, bidirectional, or effective application bandwidth.` |
| Priority | **P0** |

---

## 2.20 NVIDIA vs AMD Benchmark Comparison Claims

| Field | Validation |
|---|---|
| Claim | NVIDIA and AMD products can be compared by training/inference benchmark results. |
| Current value or formula | Vendor claims, MLPerf results, third-party benchmark studies, internal tests. |
| Validation status | **Valid only when benchmark, stack, precision, model, batch size, topology, and software versions are stated.** |
| Corrected value or safer wording | Do not make universal benchmark claims in the main chapter. Use examples as environment-specific. |
| Confidence label | `[ENV-SPECIFIC]` |
| Source type needed | MLPerf submissions, official benchmark reports, reproducible third-party tests, internal profiling with method disclosure |
| Recommended final wording for the book | `[ENV-SPECIFIC] NVIDIA-vs-AMD performance comparisons are benchmark-specific. A valid comparison must state model, precision, sequence length, batch size, framework, kernel stack, compiler/runtime versions, topology, power settings, and metric. Use benchmark claims as evidence for a workload, not as universal architecture truth.` |
| Priority | **P0** |

---

## 2.21 Hardware Selection Claims: 70B Inference, Long-Context Serving, Training Clusters

| Field | Validation |
|---|---|
| Claim | Hardware selection differs for 70B inference, long-context serving, and large training clusters. |
| Current value or formula | 70B BF16 weights ≈140 GB; FP8 weights ≈70 GB; INT4 weights ≈35 GB. Long-context serving stresses KV cache. Training clusters stress compute, memory, interconnect, reliability, and power. |
| Validation status | **Valid as representative architecture guidance.** |
| Corrected value or safer wording | Use `[ESTIMATED]` for memory math and `[REPRESENTATIVE]` for selection guidance. |
| Confidence label | `[ESTIMATED]` / `[REPRESENTATIVE]` |
| Source type needed | FLOP/memory derivations, model configs, serving benchmarks, Ch11 KV-cache math |
| Recommended final wording for the book | `[ESTIMATED] A 70B dense model requires about 140 GB for BF16 weights, about 70 GB for FP8 weights, and about 35 GB for INT4 weights, before KV cache and runtime overhead. `[REPRESENTATIVE]` Long-context serving often rewards HBM capacity/bandwidth and KV-cache efficiency, while large training clusters reward compute, HBM, high-bandwidth local fabric, scale-out network, reliability, and software maturity.` |
| Priority | **P0** |

---

## 2.22 Any Claim That Needs Confidence Labels

| Claim Type | Required Label | Example |
|---|---|---|
| Shipping product page specs | `[SHIPPED]` | H200 141 GB HBM3e, 4.8 TB/s |
| Derived per-GPU value from system total | `[DERIVED FROM SHIPPED]` | DGX B200 total memory divided by 8 GPUs |
| Roadmap/future platform | `[ANNOUNCED]` | Rubin, MI400/Helios |
| Hardware-selection rule | `[REPRESENTATIVE]` | H200 often helps memory-bound inference |
| Model memory math | `[ESTIMATED]` | 70B BF16 weights ≈140 GB |
| Benchmark comparison | `[ENV-SPECIFIC]` | MI355X vs B200 benchmark ratio |
| Vendor marketing multiplier | `[ENV-SPECIFIC]` | “30x faster” for a named workload |
| HBM generation trend | `[REPRESENTATIVE]` | HBM3e improves memory capacity/bandwidth over HBM3 in product roadmaps |

### Production note

Confidence labels are especially important in Ch03B because this chapter will age fastest.

---

# 3. Corrected / Safer Wording Blocks

## 3.1 Roadmap Reading Rule

```markdown
Roadmap Reading Rules:
1. Compare dense to dense and sparse to sparse.
2. Compare GPU-level values to GPU-level values.
3. Compare system-level values to system-level values.
4. Compare rack-level values to rack-level values.
5. Separate shipping specs from announced roadmap.
6. Treat benchmark wins as workload-specific, not universal.
7. Do not choose hardware from TFLOPS alone.
```

## 3.2 NVIDIA Hopper and H200

```markdown
[SHIPPED] H100 SXM5 80 GB provides 80 GB HBM3 and 3.35 TB/s memory bandwidth. [DERIVED FROM SHIPPED] Dense/non-sparse BF16 peak is approximately 989.4 TFLOPS, while the approximately 1,978.9 TFLOPS BF16 value is the sparse Tensor Core peak.

[SHIPPED] H200 provides 141 GB HBM3e and 4.8 TB/s memory bandwidth. [REPRESENTATIVE] H200 is best understood as a Hopper-generation memory refresh that improves memory-bound inference and KV-cache capacity economics.
```

## 3.3 NVIDIA Blackwell Product-Level Guardrail

```markdown
[SHIPPED] DGX B200 is a system-level product with 8 NVIDIA Blackwell GPUs, 1,440 GB total GPU memory, 64 TB/s total HBM3e bandwidth, and 14.4 TB/s aggregate NVLink bandwidth. These are not per-GPU values.

[SHIPPED] GB200 NVL72 is a rack-scale product connecting 36 Grace CPUs and 72 Blackwell GPUs in a 72-GPU NVLink domain. Its values must be compared against rack-scale products, not a single GPU.
```

## 3.4 Rubin

```markdown
[ANNOUNCED] Rubin / Vera Rubin should be treated as NVIDIA’s future roadmap direction beyond Blackwell. Use it as a planning signal unless official product-level specifications and availability are verified for the edition date.
```

## 3.5 AMD MI300X and MI350

```markdown
[SHIPPED] AMD Instinct MI300X provides 192 GB HBM3, approximately 5.3 TB/s peak memory bandwidth, and approximately 1,307.4 TFLOPS dense BF16 peak theoretical performance.

[SHIPPED] AMD Instinct MI350 Series product materials list 288 GB HBM3e and 8 TB/s peak memory bandwidth, with CDNA 4 architecture and next-generation MXFP datatype support. Use product-specific wording for MI350X vs MI355X.
```

## 3.6 MI400 / Helios

```markdown
[ANNOUNCED] AMD Helios / MI400 should be discussed as a future rack-scale roadmap direction, not a shipping product specification, unless official product-level datasheets exist at publication time.
```

## 3.7 Benchmark Comparisons

```markdown
[ENV-SPECIFIC] Accelerator benchmark comparisons must state model, precision, batch size, sequence length, framework, kernel stack, compiler/runtime versions, topology, power settings, and metric. Do not generalize one benchmark result into a universal hardware ranking.
```

---

# 4. P0 / P1 / P2 Validation Action List

## P0 — Must Fix Before Chapter 3B Production Source

| Task | Action |
|---|---|
| Add roadmap reading rules | Opening callout |
| Validate A100 80GB memory and Tensor Core values | Use SKU-specific wording |
| Reuse corrected H100 dense/sparse values | Dense ≈989.4, sparse ≈1,978.9 |
| Validate H100 NVLink 4 directionality | Use aggregate per-GPU wording |
| Validate H200 memory refresh wording | H200 = Hopper-generation memory refresh |
| Separate B200, DGX B200, GB200 Superchip, GB200 NVL72 | Product-level guardrail |
| Validate DGX B200 system-level values | Do not use as per-GPU values |
| Validate GB200 NVL72 rack-level wording | Do not compare to single GPU |
| Treat Rubin as `[ANNOUNCED]` | No procurement-ready claims |
| Validate MI300X values | Memory, BF16, chiplet wording |
| Validate MI325X status | `[SHIPPED]` or `[ANNOUNCED]` depending source |
| Validate MI350/MI355 values | Product-specific, CDNA 4 and MXFP |
| Treat MI400/Helios as `[ANNOUNCED]` | Roadmap, not final spec |
| Label all benchmark comparisons `[ENV-SPECIFIC]` | Avoid universal ranking |
| Add confidence labels to all roadmap claims | Required before source chapter |

## P1 — Strongly Recommended

| Task | Action |
|---|---|
| Add NVIDIA generation comparison table | Product-level columns |
| Add AMD generation comparison table | Product-level/status columns |
| Add HBM evolution table | Product-specific caution |
| Add interconnect evolution table | Directionality caution |
| Add hardware selection matrix | Workload-driven guidance |
| Add annual-refresh checklist | Current-as-of tracking |
| Cross-reference Appendix A | Keep hardware specs centralized |
| Cross-reference Ch03A, Ch04, Ch10, Ch14 | Hardware/memory/network continuity |

## P2 — Nice to Have

| Task | Action |
|---|---|
| Add one-page roadmap cheat sheet | Reader aid |
| Add hardware comparison worksheet | Interview prep |
| Add procurement checklist | Business/architecture value |
| Add future-edition refresh template | Maintenance |
| Add optional benchmark appendix | Avoid cluttering main roadmap chapter |

---

# 5. Source Categories for Final Book

Use these source categories when generating the final production chapter:

| Source Category | Use |
|---|---|
| NVIDIA A100 product page / datasheet | A100 HBM2e, BF16/FP16 Tensor Core values |
| NVIDIA H100 product page / datasheet | H100 HBM3, sparse BF16, NVLink 4 |
| NVIDIA H200 product page / datasheet | H200 HBM3e capacity and bandwidth |
| NVIDIA DGX B200 product page / datasheet | DGX B200 system-level values |
| NVIDIA DGX GB200 / GB200 NVL72 pages | GB200 Superchip and rack-level values |
| NVIDIA Blackwell architecture pages | Product-level Blackwell features |
| NVIDIA Rubin / HGX Rubin official pages | Roadmap wording |
| AMD MI300X product page / datasheet | MI300X memory, compute, fabric |
| AMD MI325X product page / datasheet | MI325X HBM3e status and bandwidth |
| AMD MI350 Series / MI350X / MI355X pages | CDNA 4, HBM3e, MXFP claims |
| AMD CDNA 4 whitepaper | MI350 architecture details |
| AMD Helios / MI400 official announcement | Roadmap wording |
| JEDEC / HBM sources | HBM generation background |
| MLPerf results | Benchmark comparisons |
| Vendor benchmark reports | Label `[ENV-SPECIFIC]` |
| Internal lab measurements | Label `[ENV-SPECIFIC]` and disclose method |

---

# 6. Chapter 3B Production Rules

1. Put “Current as of 2026 edition” near roadmap/spec tables.
2. Do not mix per-GPU, per-system, and per-rack values.
3. Do not mix dense and sparse TFLOPS.
4. Do not mix line rate and effective workload bandwidth.
5. Treat B200, DGX B200, GB200 Superchip, and GB200 NVL72 as different product levels.
6. Treat Rubin and MI400 as announced roadmap unless final product pages exist.
7. Keep benchmark comparisons out of the main narrative unless fully qualified.
8. Use neutral NVIDIA-vs-AMD wording.
9. Use workload-fit language instead of vendor-ranking language.
10. Keep detailed specs in Appendix A and use Ch03B to teach roadmap interpretation.

---

# 7. Commit Instructions

Save this file as:

```text
publishing/validation/ch03b_technical_validation.md
```

Then run:

```powershell
git add publishing\validation\ch03b_technical_validation.md
git commit -m "Add Chapter 3B technical validation plan"
git push origin production-v1.0
```

---

# 8. Next Production Step

After committing this validation file, the next production task is:

```text
Create source/chapters/ch03b_gpu_roadmap.md
```

That source chapter should incorporate:

1. Roadmap reading rules.
2. Table 3B.1 through Table 3B.8.
3. Figure placeholders for Fig 3B.1 through Fig 3B.8.
4. Product-level guardrails for B200, DGX B200, GB200, and GB200 NVL72.
5. NVIDIA timeline with shipped vs announced labels.
6. AMD timeline with shipped vs announced labels.
7. HBM evolution section.
8. Interconnect evolution section.
9. Neutral NVIDIA-vs-AMD comparison language.
10. Hardware selection decision tree.
11. Principal interview explanation section.
12. Key takeaways and review questions.
