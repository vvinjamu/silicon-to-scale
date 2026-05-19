# Chapter 4 Technical Validation Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Ch04 — *GPU Memory Hierarchy and HBM Deep Dive*  
**Target file:** `publishing/validation/ch04_technical_validation.md`  
**Production status:** Production Planning Pack  
**Last reviewed:** 2026-04-30

---

## 0. Executive Summary

Ch04 is a memory-performance chapter. The highest-risk claims are:

1. Product-specific HBM values.
2. Dense vs sparse compute values used in ridge-point math.
3. KV-cache memory formulas.
4. FlashAttention HBM-traffic claims.
5. HBM generation claims that accidentally become product-specific.
6. PCIe/host memory fallback bandwidth claims.
7. “Decode is memory-bound” claims that should be labeled representative.
8. Any B200/GB200 values that mix GPU/system/rack levels.

Core production rule:

```text
Memory claims must declare product level, datatype, directionality, and confidence label.
```

---

# 1. Validation Table

---

## 1.1 H100 SXM5 HBM Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | H100 SXM5 provides 80 GB HBM3 and 3.35 TB/s peak HBM bandwidth. |
| Current value/formula | 80 GB HBM3, 3.35 TB/s |
| Validation status | Valid for H100 SXM5 80 GB. |
| Corrected value/safe wording | Specify SXM5 and 80 GB SKU. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 product page / datasheet |
| Recommended final wording | `[SHIPPED] H100 SXM5 80 GB provides 80 GB HBM3 with 3.35 TB/s peak memory bandwidth. Other H100 variants may differ.` |
| Priority | P0 |

---

## 1.2 H200 HBM3e Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | H200 provides 141 GB HBM3e and 4.8 TB/s bandwidth. |
| Current value/formula | 141 GB HBM3e, 4.8 TB/s |
| Validation status | Valid. |
| Corrected value/safe wording | State H200 specifically. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H200 product page / datasheet |
| Recommended final wording | `[SHIPPED] NVIDIA H200 provides 141 GB HBM3e and 4.8 TB/s peak memory bandwidth, making it especially relevant for memory-capacity and memory-bandwidth constrained workloads.` |
| Priority | P0 |

---

## 1.3 MI300X HBM Capacity and Bandwidth

| Field | Validation |
|---|---|
| Claim | MI300X provides 192 GB HBM3 and approximately 5.3 TB/s bandwidth. |
| Current value/formula | 192 GB HBM3, 5.3 TB/s |
| Validation status | Valid. |
| Corrected value/safe wording | Use “approximately 5.3 TB/s peak local bandwidth.” |
| Confidence label | `[SHIPPED]` |
| Source type needed | AMD MI300X product page / datasheet |
| Recommended final wording | `[SHIPPED] AMD Instinct MI300X provides 192 GB HBM3 with approximately 5.3 TB/s peak local memory bandwidth.` |
| Priority | P0 |

---

## 1.4 MI325X HBM3e Capacity and Bandwidth If Referenced

| Field | Validation |
|---|---|
| Claim | MI325X provides 256 GB HBM3e and approximately 6 TB/s bandwidth. |
| Current value/formula | 256 GB HBM3e, ≈6 TB/s |
| Validation status | Valid if official product page/datasheet is used at publication time. |
| Corrected value/safe wording | Use `[SHIPPED]` only if final product docs; otherwise `[ANNOUNCED]`. |
| Confidence label | `[SHIPPED]` or `[ANNOUNCED]` |
| Source type needed | AMD MI325X official product page / datasheet |
| Recommended final wording | `[SHIPPED/ANNOUNCED] AMD Instinct MI325X is a memory-focused refresh with 256 GB HBM3e and approximately 6 TB/s peak memory bandwidth. Verify product status at publication time.` |
| Priority | P1 |

---

## 1.5 MI350 / MI355 HBM3e Capacity and Bandwidth If Referenced

| Field | Validation |
|---|---|
| Claim | MI350 Series provides 288 GB HBM3e and approximately 8 TB/s bandwidth. |
| Current value/formula | 288 GB HBM3e, 8 TB/s |
| Validation status | Valid if tied to AMD MI350 Series official product material. |
| Corrected value/safe wording | Use product-specific MI350X/MI355X wording. |
| Confidence label | `[SHIPPED]` if official product page |
| Source type needed | AMD MI350 Series / MI350X / MI355X page |
| Recommended final wording | `[SHIPPED] AMD Instinct MI350 Series product materials list 288 GB HBM3e and approximately 8 TB/s peak memory bandwidth. Use product-specific wording for MI350X vs MI355X.` |
| Priority | P1 |

---

## 1.6 B200 / DGX B200 / GB200 HBM Values If Referenced

| Field | Validation |
|---|---|
| Claim | Blackwell-class systems have larger HBM3e memory and bandwidth. |
| Current value/formula | DGX B200: 1,440 GB total GPU memory, 64 TB/s total HBM3e bandwidth. |
| Validation status | Valid only as system-level DGX B200 values. |
| Corrected value/safe wording | Do not quote system-level values as per-GPU values. |
| Confidence label | `[SHIPPED]` for DGX B200 official system page; `[DERIVED FROM SHIPPED]` for calculated per-GPU average |
| Source type needed | NVIDIA DGX B200 / GB200 product pages |
| Recommended final wording | `[SHIPPED] DGX B200 lists 1,440 GB total GPU memory and 64 TB/s total HBM3e bandwidth across the system. These are system-level values and must not be presented as per-GPU values.` |
| Priority | P0 |

---

## 1.7 H100 Dense BF16 Ridge Point

| Field | Validation |
|---|---|
| Claim | H100 SXM5 ridge point using dense BF16 is approximately 295 FLOP/byte. |
| Current value/formula | 989.4 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte |
| Validation status | Valid if dense BF16 is used. |
| Corrected value/safe wording | Do not use sparse BF16 peak for dense ridge point. |
| Confidence label | `[DERIVED FROM SHIPPED]` |
| Source type needed | NVIDIA H100 spec plus arithmetic |
| Recommended final wording | `[DERIVED FROM SHIPPED] Using H100 SXM5 dense/non-sparse BF16 peak of approximately 989.4 TFLOPS and 3.35 TB/s HBM bandwidth gives a ridge point of approximately 295 FLOP/byte.` |
| Priority | P0 |

---

## 1.8 Arithmetic Intensity Formula

| Field | Validation |
|---|---|
| Claim | Arithmetic intensity equals FLOPs divided by bytes moved. |
| Current value/formula | `AI = FLOPs / bytes moved` |
| Validation status | Valid. |
| Corrected value/safe wording | Clarify bytes moved through relevant memory tier. |
| Confidence label | `[DERIVED FROM SHIPPED]` for hardware example; formula itself standard |
| Source type needed | Roofline literature / Ch01 reference |
| Recommended final wording | `Arithmetic intensity is FLOPs divided by bytes moved through the bottleneck memory tier. In Roofline analysis, compare arithmetic intensity to the hardware ridge point.` |
| Priority | P0 |

---

## 1.9 Ridge Point Formula

| Field | Validation |
|---|---|
| Claim | Ridge point equals peak compute divided by peak memory bandwidth. |
| Current value/formula | `ridge point = peak FLOPs / memory bandwidth` |
| Validation status | Valid. |
| Corrected value/safe wording | Use consistent units. |
| Confidence label | `[DERIVED FROM SHIPPED]` when using hardware values |
| Source type needed | Roofline literature / Ch01 reference |
| Recommended final wording | `The ridge point is the arithmetic intensity required to become compute-bound: peak FLOPs divided by peak memory bandwidth.` |
| Priority | P0 |

---

## 1.10 KV Cache Memory Formula

| Field | Validation |
|---|---|
| Claim | KV cache bytes scale with layers, batch/concurrency, sequence length, KV heads, head dimension, and dtype size. |
| Current value/formula | `KV bytes ≈ 2 × L × B × S × n_kv × d_head × bytes_per_element` |
| Validation status | Valid for decoder-only transformer KV cache using consistent notation. |
| Corrected value/safe wording | Explain MHA/GQA/MQA difference through `n_kv`. |
| Confidence label | `[ESTIMATED]` for sizing calculations |
| Source type needed | Transformer architecture references / Ch02 / Ch11 |
| Recommended final wording | `[ESTIMATED] KV cache bytes can be approximated as 2 × L × B × S × n_kv × d_head × bytes_per_element, where the factor 2 accounts for key and value tensors. For GQA/MQA models, use the number of KV heads, not the number of query heads.` |
| Priority | P0 |

---

## 1.11 70B Model Weight Memory

| Field | Validation |
|---|---|
| Claim | 70B model weights require ~140 GB BF16, ~70 GB FP8, ~35 GB INT4 before overhead. |
| Current value/formula | parameters × bytes per parameter |
| Validation status | Valid as simplified estimate. |
| Corrected value/safe wording | Exclude KV cache/runtime overhead explicitly. |
| Confidence label | `[ESTIMATED]` |
| Source type needed | Basic parameter memory math |
| Recommended final wording | `[ESTIMATED] A 70B dense model requires approximately 140 GB for BF16 weights, 70 GB for FP8 weights, and 35 GB for INT4 weights before KV cache, metadata, runtime buffers, and fragmentation.` |
| Priority | P0 |

---

## 1.12 Decode Is Often Memory-Bound

| Field | Validation |
|---|---|
| Claim | LLM decode is often memory-bandwidth-bound. |
| Current value/formula | Decode processes one/few tokens and repeatedly reads weights/KV cache. |
| Validation status | Directionally valid but workload-dependent. |
| Corrected value/safe wording | Use “often” and label representative. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | LLM inference performance papers / serving framework docs / profiler evidence |
| Recommended final wording | `[REPRESENTATIVE] LLM decode is often memory-bandwidth-sensitive because each generated token may require reading model weights and prior KV-cache state while exposing limited arithmetic intensity compared with large prefill GEMMs.` |
| Priority | P0 |

---

## 1.13 Prefill vs Decode Memory Behavior

| Field | Validation |
|---|---|
| Claim | Prefill is more GEMM/compute-heavy while decode is more memory/KV-cache bandwidth-sensitive. |
| Current value/formula | Prefill processes long prompt in parallel; decode generates one token at a time. |
| Validation status | Valid as representative guidance. |
| Corrected value/safe wording | Avoid universal claim; depends on batch, model, kernel, hardware. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | LLM serving references / Ch06 |
| Recommended final wording | `[REPRESENTATIVE] Prefill usually exposes more parallel GEMM work, while decode is often more sensitive to HBM bandwidth, KV-cache layout, and batching strategy. The exact bottleneck depends on model, batch, sequence length, kernel implementation, and hardware.` |
| Priority | P1 |

---

## 1.14 FlashAttention Memory Traffic Claim

| Field | Validation |
|---|---|
| Claim | FlashAttention reduces HBM traffic by tiling attention and avoiding full attention-matrix materialization in HBM. |
| Current value/formula | SRAM/shared-memory tiling + online softmax. |
| Validation status | Valid directionally. Exact speedup is environment-specific. |
| Corrected value/safe wording | Avoid fixed speedup unless benchmark context provided. |
| Confidence label | `[REPRESENTATIVE]`; benchmark speedups `[ENV-SPECIFIC]` |
| Source type needed | FlashAttention paper / implementation docs |
| Recommended final wording | `[REPRESENTATIVE] FlashAttention improves attention performance by reducing HBM reads/writes through on-chip tiling and online softmax, avoiding materialization of the full attention matrix in HBM. Exact speedup is workload-, hardware-, and implementation-specific.` |
| Priority | P0 |

---

## 1.15 Operator Fusion Reduces HBM Traffic

| Field | Validation |
|---|---|
| Claim | Operator fusion reduces intermediate reads/writes to HBM. |
| Current value/formula | Fuse elementwise/norm/activation operations. |
| Validation status | Valid. |
| Corrected value/safe wording | Fusion can increase register pressure or reduce flexibility. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | Compiler docs / kernel optimization references |
| Recommended final wording | `[REPRESENTATIVE] Operator fusion can reduce HBM traffic by avoiding intermediate tensor materialization, but it may increase register pressure, reduce occupancy, or require specialized kernels.` |
| Priority | P1 |

---

## 1.16 Quantization Reduces Memory Footprint

| Field | Validation |
|---|---|
| Claim | Quantization reduces weight and/or KV-cache memory footprint. |
| Current value/formula | BF16=2 bytes, FP8/INT8=1 byte, INT4=0.5 byte per value before metadata. |
| Validation status | Valid as simplified memory math. |
| Corrected value/safe wording | Include metadata and quality caveats. |
| Confidence label | `[ESTIMATED]` for memory math; `[REPRESENTATIVE]` for production guidance |
| Source type needed | Quantization literature / Ch08 |
| Recommended final wording | `[ESTIMATED] Reducing precision from BF16 to FP8 roughly halves raw value storage, and INT4 roughly quarters it, before metadata and packing overhead. `[REPRESENTATIVE]` Production use must account for accuracy, kernel support, calibration, and serving-stack compatibility.` |
| Priority | P1 |

---

## 1.17 CPU Offload / Host Memory Fallback

| Field | Validation |
|---|---|
| Claim | CPU offload can reduce HBM pressure but may add PCIe latency and bandwidth bottlenecks. |
| Current value/formula | PCIe 5.0 x16 ≈64 GB/s per direction theoretical. |
| Validation status | Valid with platform caveats. |
| Corrected value/safe wording | Treat as environment-specific. |
| Confidence label | `[ENV-SPECIFIC]` for performance; `[DERIVED FROM SHIPPED]` for PCIe theoretical |
| Source type needed | PCIe spec / framework docs |
| Recommended final wording | `[REPRESENTATIVE] CPU offload can reduce HBM capacity pressure, but if it enters the critical path it can become limited by PCIe bandwidth, host-memory latency, NUMA placement, DMA behavior, and software scheduling.` |
| Priority | P1 |

---

## 1.18 L2 Cache / Shared Memory Values

| Field | Validation |
|---|---|
| Claim | L2 cache and shared memory reduce HBM traffic when reuse exists. |
| Current value/formula | H100 SXM5 L2 50 MB; shared memory/L1 configuration up to 228 KB shared memory per SM if referenced. |
| Validation status | Valid for H100 if SKU/source-specific. |
| Corrected value/safe wording | Avoid overgeneralizing across GPUs. |
| Confidence label | `[SHIPPED]` for official values; `[REPRESENTATIVE]` for behavior |
| Source type needed | NVIDIA Hopper architecture whitepaper |
| Recommended final wording | `[REPRESENTATIVE] L2 and shared memory improve performance only when the workload has exploitable reuse. Official cache capacities are architecture- and SKU-specific and should be stated only when sourced.` |
| Priority | P1 |

---

## 1.19 HBM Generation Trend

| Field | Validation |
|---|---|
| Claim | HBM2e → HBM3 → HBM3e → HBM4 represents increasing capacity/bandwidth direction. |
| Current value/formula | Product-specific adoption: A100 HBM2e, H100/MI300X HBM3, H200/MI325X/MI350/B200 systems HBM3e, HBM4 future. |
| Validation status | Valid as trend; exact values product-specific. |
| Corrected value/safe wording | Do not assign generic bandwidth values to HBM generation. |
| Confidence label | `[REPRESENTATIVE]`; product values `[SHIPPED]`; future `[ANNOUNCED]` |
| Source type needed | Vendor product pages / JEDEC / HBM vendors |
| Recommended final wording | `[REPRESENTATIVE] HBM generations generally increase capacity and bandwidth, but architecture decisions should use product-specific values. Treat HBM4 as roadmap/future unless tied to a shipping product specification.` |
| Priority | P1 |

---

## 1.20 Memory Bandwidth Utilization Claims

| Field | Validation |
|---|---|
| Claim | A kernel using high HBM bandwidth but low compute utilization is likely memory-bound. |
| Current value/formula | Profiler-driven interpretation. |
| Validation status | Valid but environment/tool-specific. |
| Corrected value/safe wording | Use profiler evidence and compare to ridge point. |
| Confidence label | `[ENV-SPECIFIC]` |
| Source type needed | Nsight Compute / rocprof / profiling docs |
| Recommended final wording | `[ENV-SPECIFIC] High HBM bandwidth utilization with low arithmetic-unit utilization is a strong memory-bound signal, but confirm with arithmetic intensity, cache behavior, stall reasons, and profiler counters.` |
| Priority | P1 |

---

# 2. P0 / P1 / P2 Validation Action List

## P0 — Must Validate Before Production Source

| Task |
|---|
| Validate H100 SXM5 HBM capacity/bandwidth |
| Validate H200 HBM3e capacity/bandwidth |
| Validate MI300X HBM capacity/bandwidth |
| Validate H100 dense BF16 ridge point |
| Validate arithmetic intensity and ridge point formulas |
| Validate KV-cache memory formula and variable notation |
| Validate 70B weight memory estimates |
| Validate FlashAttention wording |
| Label decode memory-bound claims `[REPRESENTATIVE]` |
| Avoid generic HBM3e bandwidth wording |
| Avoid per-GPU/system/rack HBM mixing |

## P1 — Strongly Recommended

| Task |
|---|
| Validate MI325X and MI350/MI355 values if referenced |
| Validate B200/DGX B200/GB200 system-level values if referenced |
| Validate L2/shared memory values if included |
| Validate CPU offload / PCIe bandwidth wording |
| Validate operator fusion tradeoff wording |
| Validate quantization memory math and caveats |
| Add profiler metric references |

## P2 — Nice to Have

| Task |
|---|
| Add HBM4 roadmap note |
| Add memory worksheet formulas |
| Add advanced HBM packaging note |
| Add benchmark examples with `[ENV-SPECIFIC]` labels |

---

# 3. Corrected/Safe Wording Blocks

## HBM Product Values

```markdown
[SHIPPED] H100 SXM5 80 GB provides 80 GB HBM3 with 3.35 TB/s peak memory bandwidth.

[SHIPPED] NVIDIA H200 provides 141 GB HBM3e with 4.8 TB/s peak memory bandwidth.

[SHIPPED] AMD Instinct MI300X provides 192 GB HBM3 with approximately 5.3 TB/s peak local memory bandwidth.
```

## Ridge Point

```markdown
[DERIVED FROM SHIPPED] Using H100 SXM5 dense/non-sparse BF16 peak of approximately 989.4 TFLOPS and 3.35 TB/s HBM bandwidth gives a ridge point of approximately 295 FLOP/byte.
```

## KV Cache Formula

```markdown
[ESTIMATED] KV cache bytes can be approximated as:

KV cache bytes ≈ 2 × L × B × S × n_kv × d_head × bytes_per_element

The factor 2 accounts for key and value tensors. For GQA/MQA models, use the number of KV heads, not the number of query heads.
```

## Decode Bottleneck

```markdown
[REPRESENTATIVE] LLM decode is often memory-bandwidth-sensitive because each generated token may require reading model weights and prior KV-cache state while exposing limited arithmetic intensity compared with large prefill GEMMs.
```

## FlashAttention

```markdown
[REPRESENTATIVE] FlashAttention improves attention performance by reducing HBM reads/writes through on-chip tiling and online softmax, avoiding materialization of the full attention matrix in HBM. Exact speedup is workload-, hardware-, and implementation-specific.
```

---

# 4. Recommended Source Categories

| Source Category | Use |
|---|---|
| NVIDIA H100 product page/datasheet | H100 HBM and dense/sparse BF16 context |
| NVIDIA H200 product page/datasheet | H200 HBM3e capacity/bandwidth |
| AMD MI300X datasheet | MI300X HBM and bandwidth |
| AMD MI325X / MI350 Series pages | Future or current AMD HBM values |
| NVIDIA Hopper architecture whitepaper | H100 L2/shared memory/TMA |
| FlashAttention paper | Memory traffic and tiling claims |
| PCI-SIG PCIe spec | PCIe bandwidth directionality |
| CUDA / Nsight Compute docs | Memory bottleneck profiling |
| ROCm / rocprof docs | AMD profiling |
| Ch11 KV-cache source | KV formula alignment |

---

# 5. Recommended Commit

Save this file as:

```text
publishing/validation/ch04_technical_validation.md
```

Then run:

```powershell
git add publishing\validation\ch04_technical_validation.md
git commit -m "Add Chapter 4 technical validation plan"
git push origin production-v1.0
```
