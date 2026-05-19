# Chapter 8 Technical Validation Plan — Quantization and Precision Optimization

**Book:** AI/ML Infrastructure from Silicon to Scale  
**Chapter:** Ch08 — Quantization and Precision Optimization  
**Primary source:** `ch08_quantization.pdf`  
**Validation goal:** Convert PDF claims into safe, production-ready wording with confidence labels and source requirements.

---

## Confidence Labels Used

```text
[SHIPPED]              Verified shipping hardware/software or stable published standard/source
[ANNOUNCED]            Vendor-disclosed roadmap/product capability; confirm availability before procurement
[DERIVED FROM SHIPPED] Calculated from shipped specs or stable definitions
[ESTIMATED]            Calculated or inferred from assumptions; methodology shown
[REPRESENTATIVE]       Illustrative values for teaching; not universal
[ENV-SPECIFIC]         Depends on model, hardware, runtime, kernel, dataset, traffic, and configuration
```

---

## Validation Table

| # | Claim | Current value/formula from PDF | Validation status | Corrected value / safe wording | Confidence label | Source type needed | Recommended final wording | Priority |
|---:|---|---|---|---|---|---|---|---|
| 1 | Raw model memory scales with bytes per parameter | `model_bytes = params × bytes_per_param` | Valid | Keep; clarify excludes KV cache, activations, optimizer state, runtime buffers, metadata, fragmentation | `[DERIVED FROM SHIPPED]` | First-principles formula | `Raw weight memory = parameter count × bytes per parameter. This is only the starting point; serving also needs KV cache and runtime headroom, while training also needs gradients, activations, and optimizer state.` | P0 |
| 2 | 70B FP32 model is 280 GB | `70B × 4 B = 280 GB` | Valid derived estimate | Keep as raw weight memory | `[ESTIMATED]` | Calculation | `A 70B dense model requires roughly 280 GB for FP32 raw weights before overhead.` | P0 |
| 3 | 70B BF16 model is 140 GB | `70B × 2 B = 140 GB` | Valid derived estimate | Keep as raw weight memory | `[ESTIMATED]` | Calculation | `BF16 raw weights for a 70B dense model are roughly 140 GB before KV cache, activations, and runtime overhead.` | P0 |
| 4 | 70B FP8 model is 70 GB | `70B × 1 B = 70 GB` | Valid derived estimate | Keep; warn about scaling metadata and runtime headroom | `[ESTIMATED]` | Calculation; runtime docs | `FP8 raw weights are roughly 70 GB for a 70B dense model, before scale metadata and serving overhead.` | P0 |
| 5 | 70B INT4 model is 35 GB | `70B × 0.5 B = 35 GB` | Valid derived estimate | Keep; include scale/packing overhead | `[ESTIMATED]` | Calculation; quantization method docs | `INT4 raw weights are roughly 35 GB for a 70B dense model, plus scale and metadata overhead.` | P0 |
| 6 | BF16 has FP32-like dynamic range | BF16 has 1 sign, 8 exponent, 7 mantissa bits | Valid | Keep | `[SHIPPED]` | IEEE/bfloat16 references; framework docs | `BF16 keeps an 8-bit exponent like FP32, giving it much wider range than FP16 while using 16 bits of storage.` | P0 |
| 7 | FP16 range is much narrower than BF16 | FP16 has 5 exponent bits and range around ±65,504 | Valid | Keep; avoid saying BF16 always eliminates all loss-scaling scenarios universally | `[SHIPPED]` | IEEE FP16 docs; framework docs | `FP16 has a much smaller exponent range than BF16, which is why BF16 is often preferred for large-model training stability.` | P0 |
| 8 | FP8 has E4M3 and E5M2 variants | E4M3 for weights/activations; E5M2 for gradients | Valid with NVIDIA TE hybrid recipe | Use official TE wording: hybrid uses E4M3 forward and E5M2 backward | `[SHIPPED]` | NVIDIA Transformer Engine docs | `In NVIDIA Transformer Engine hybrid FP8 recipes, forward tensors commonly use E4M3 and backward tensors use E5M2 because E5M2 provides wider range.` | P0 |
| 9 | Transformer Engine accelerates transformer models with FP8 | TE uses FP8 on Hopper/Ada/Blackwell | Valid | Keep; tie to official docs | `[SHIPPED]` | NVIDIA Transformer Engine docs | `NVIDIA Transformer Engine provides FP8 mixed-precision support for transformer workloads on supported NVIDIA GPUs, with automatic scaling recipes and optimized kernels.` | P0 |
| 10 | Blackwell supports NVFP4 | NVFP4/FP4 on B200/Blackwell | Valid in current TE docs | Keep; avoid universal performance claims | `[SHIPPED]` | NVIDIA TE docs and Blackwell docs | `Blackwell-generation NVIDIA GPUs and Transformer Engine documentation include NVFP4 support; production benefit depends on recipe, model, kernel, and deployment stack.` | P0 |
| 11 | AMD CDNA3 supports FP8 | ROCm/MI300X supports FP8 | Valid with nuance: AMD FP8 FNUZ differs from NVIDIA H100 FP8 | `[SHIPPED]` | AMD ROCm precision docs | `AMD CDNA3 / MI300-series support FP8 formats in ROCm, but AMD FP8 FNUZ details differ from NVIDIA FP8 formats; use framework-supported kernels and validated models.` | P0 |
| 12 | AMD MI350 has 288 GB HBM3E and 8 TB/s bandwidth | PDF says MI350/CDNA4 future | Needs refresh | Update to official MI350 wording; label shipped if product page is public/current | `[SHIPPED]` | AMD MI350 official product page | `AMD Instinct MI350-series product pages list 288 GB HBM3E and up to 8 TB/s bandwidth, with MXFP6/MXFP4 datatype support. Verify exact SKU before quoting.` | P0 |
| 13 | Quantization formula | `Q(x)=clamp(round(x/scale + zero_point), q_min, q_max)` | Valid | Keep | `[SHIPPED]` | Numerical methods / quantization docs | `Quantization maps values to a smaller grid using a scale and optionally a zero point; dequantization reconstructs an approximate value.` | P0 |
| 14 | Symmetric quantization | `scale=max(abs(x))/q_max`, `zero_point=0` | Valid | Keep | `[SHIPPED]` | Quantization docs | `Symmetric quantization uses a zero point of zero and is common for hardware-friendly weight quantization.` | P1 |
| 15 | Asymmetric quantization | `scale=(max-min)/(q_max-q_min)` | Valid | Keep | `[SHIPPED]` | Quantization docs | `Asymmetric quantization can better use integer range for non-zero-centered data but may add runtime complexity.` | P1 |
| 16 | Outliers damage quantization quality | Example max value 72.1 destroys small values | Concept valid; values representative | Label illustrative | `[REPRESENTATIVE]` | Paper examples; model calibration data | `A single large outlier can force a coarse scale and cause many small values to round to zero. The exact outlier distribution is model- and layer-specific.` | P0 |
| 17 | Per-group quantization improves INT4 quality | group size 128 example | Concept valid; exact quality values representative | Keep as method guidance, not universal | `[REPRESENTATIVE]` | GPTQ/AWQ papers and implementations | `Low-bit LLM quantization usually requires finer granularity such as per-channel or per-group scales; per-tensor INT4 is usually unsafe for high-quality LLM serving.` | P0 |
| 18 | Naive INT4 is poor | PDF cites large MMLU drop | Concept valid; values representative | Keep with labels | `[REPRESENTATIVE]` | Benchmark suite; model-specific eval | `Naive INT4 PTQ can produce unacceptable quality loss on LLMs; GPTQ/AWQ-style methods are commonly used to make INT4 viable.` | P0 |
| 19 | GPTQ uses second-order error compensation | Hessian-based correction | Valid | Keep but simplify for readability | `[SHIPPED]` | GPTQ paper | `GPTQ is a one-shot PTQ method that uses approximate second-order information to quantize large transformer weights while compensating for quantization error.` | P1 |
| 20 | AWQ protects salient channels | Top ~1% salient weights/channels | Valid from AWQ paper, but exact percentage should be softened | `[SHIPPED]` | AWQ paper | `AWQ uses activation statistics to identify salient channels and applies scaling transformations that protect important weights during low-bit quantization.` | P1 |
| 21 | SmoothQuant migrates activation difficulty to weights | `Y=XW=(X diag(s)^-1)(diag(s)W)` | Valid | Keep | `[SHIPPED]` | SmoothQuant paper | `SmoothQuant smooths activation outliers by moving quantization difficulty from activations to weights using a mathematically equivalent scaling transformation.` | P1 |
| 22 | QLoRA fine-tunes large quantized models on a single 48 GB GPU | PDF claims 65B+ on 48 GB | Valid from QLoRA paper for 65B-class; do not overgeneralize to every 70B workload | `[SHIPPED]` | QLoRA paper | `QLoRA demonstrated that a frozen 4-bit base model plus LoRA adapters can drastically reduce fine-tuning memory; exact model size and sequence settings determine fit.` | P1 |
| 23 | FP8 TE quality loss is ~0.2% | PDF gives benchmark deltas | Needs caution | Treat as representative, not universal | `[REPRESENTATIVE]` | NVIDIA docs; benchmark reports; internal eval | `FP8 recipes can be near-lossless for many transformer workloads, but every production deployment should validate task-specific quality.` | P0 |
| 24 | FP8 training speedup is 1.5-1.9x | PDF claim | Environment-specific | Keep only as potential range with measured workload caveat | `[ENV-SPECIFIC]` | Benchmarks; NVIDIA docs | `FP8 can improve throughput substantially on supported hardware, but end-to-end gain depends on GEMM fraction, communication, kernels, and memory behavior.` | P0 |
| 25 | FP8 W8A8 support in vLLM | vLLM supports FP8 on H100/MI300X | Valid with version/hardware caveats | Use current docs wording | `[SHIPPED]` | vLLM docs | `vLLM documents FP8 W8A8 quantization support on supported NVIDIA and AMD GPUs; confirm version, hardware, and model support.` | P0 |
| 26 | vLLM FP8 KV cache reduces memory footprint | FP8 KV cache halves BF16/FP16 KV bytes | Valid | Keep; avoid exact performance claims | `[SHIPPED]` | vLLM docs | `vLLM documents FP8 KV-cache quantization, which reduces KV memory footprint and can improve throughput or longer-context capacity when KV memory is limiting.` | P0 |
| 27 | KV cache quantization is independent from weight quantization | PDF claim | Valid | Keep | `[DERIVED FROM SHIPPED]` | KV cache docs; serving framework docs | `Weight precision and KV-cache precision are separate deployment choices and can be combined.` | P0 |
| 28 | GQA + FP8 KV can reduce KV bytes vs MHA BF16 by 16x | Formula-based example | Valid if H_kv 64 -> 8 and BF16 -> FP8 | `[ESTIMATED]` | Formula; model architecture assumptions | `In an illustrative model where KV heads drop from 64 to 8 and KV dtype drops from BF16 to FP8, KV bytes per token can fall by 16x.` | P0 |
| 29 | Decode throughput is inversely proportional to model bytes | PDF says exact for batch=1 | Concept valid in memory-bound simplified model; not universally exact | `[ESTIMATED]` | Roofline model; benchmark validation | `In memory-bound decode, reducing bytes loaded per token can improve throughput roughly in proportion to byte reduction, until another bottleneck dominates.` | P0 |
| 30 | Tokens/sec values in PDF | H100 FP8 ~2200 tok/s, AWQ etc. | Environment-specific | Use only as representative table if assumptions shown | `[ENV-SPECIFIC]` | Benchmark methodology; hardware/runtime stack | `Representative throughput values depend on model, batch, prompt/output length, GPU, kernel, runtime, and SLA. Treat them as examples, not guarantees.` | P0 |
| 31 | Cost per 1M token estimates | Based on GPU hourly cost and utilization | Estimated | Keep formula; not fixed values | `[ESTIMATED]` | Pricing, utilization, throughput assumptions | `Cost/token should be computed from measured throughput, utilization, GPU-hour cost, and overhead assumptions.` | P1 |
| 32 | AWQ usually beats GPTQ by ~0.5% MMLU | PDF claim | Representative | Avoid universal claim | `[REPRESENTATIVE]` | Benchmark suite | `AWQ and GPTQ trade quality, calibration cost, and runtime support; compare on the target model and task.` | P0 |
| 33 | Code/math tasks can regress more than MMLU | PDF claim | Valid principle | Keep; benchmark values representative | `[REPRESENTATIVE]` | Eval methodology; benchmark data | `Aggregate benchmarks may hide task-specific regressions; code, math, and factual QA must be validated separately if they matter in production.` | P0 |
| 34 | SmoothQuant best for A100 W8A8 | PDF claim | Generally valid, but runtime-dependent | Use as recommended starting point, not rule | `[REPRESENTATIVE]` | SmoothQuant paper; deployment docs | `On hardware without native FP8, INT8 W8A8 methods such as SmoothQuant may be a strong starting point if the runtime has optimized kernels.` | P1 |
| 35 | GGUF is best for CPU/local inference | PDF claim | Generally valid within llama.cpp ecosystem | Keep with ecosystem caveat | `[REPRESENTATIVE]` | llama.cpp/GGUF docs | `GGUF is most relevant for local/CPU/edge deployment ecosystems, while GPU production serving often uses FP8, AWQ, GPTQ, or TensorRT/vLLM-supported formats.` | P1 |
| 36 | QAT gives best INT4 quality but costs training | PDF claim | Valid concept; exact cost varies | Keep | `[ENV-SPECIFIC]` | QAT methodology; internal training cost | `QAT can recover quality by exposing the model to quantization error during training, but it is much more expensive than PTQ and should be justified by production volume or quality requirements.` | P1 |
| 37 | Evaluation should include PPL, MMLU, task-specific evals | PDF claim | Valid | Keep | `[SHIPPED]` | Eval suite docs; best practices | `Quantization validation should include language-model metrics, broad capability benchmarks, and deployment-specific tasks before production rollout.` | P0 |
| 38 | MMLU-only validation is an anti-pattern | PDF claim | Valid principle | Keep | `[REPRESENTATIVE]` | Evaluation best practice | `Do not ship a quantized model based only on a single aggregate benchmark; run task-specific evals and production latency/memory tests.` | P0 |
| 39 | TF32 is transparent to many PyTorch FP32 matmul paths | PDF claim | Needs version/hardware caveat | Use safe wording | `[SHIPPED]` | PyTorch/NVIDIA docs | `On supported NVIDIA GPUs and framework settings, TF32 can be used for some FP32 tensor-core operations without changing tensor storage type.` | P1 |
| 40 | FP4/NVFP4 can be used for training/inference | PDF claim | Needs hardware/software caveat | Use TE official wording and avoid overgeneralizing quality | `[SHIPPED]` for support, `[ENV-SPECIFIC]` for results | NVIDIA TE docs; benchmark docs | `NVFP4 is supported in NVIDIA Transformer Engine on Blackwell; production suitability depends on recipe, model, eval target, and framework support.` | P0 |

---

## Current Official / Primary Source Checklist

Use these source types during the Production Source Pack step:

1. **NVIDIA Transformer Engine documentation**
   - FP8 formats, hybrid recipe, amax/scaling, Hopper/Ada/Blackwell support.
2. **NVIDIA Transformer Engine / Blackwell NVFP4 documentation**
   - NVFP4 support, MXFP8/NVFP4 distinctions, Blackwell support.
3. **vLLM quantization documentation**
   - FP8 W8A8 support and FP8 KV-cache support.
4. **AMD ROCm precision support documentation**
   - CDNA3 FP8 support, AMD FP8 FNUZ distinction.
5. **AMD MI350 official product pages**
   - 288 GB HBM3E, up to 8 TB/s bandwidth, MXFP6/MXFP4 support.
6. **Original papers**
   - GPTQ, AWQ, SmoothQuant, QLoRA.
7. **Book-local source PDFs**
   - `ch08_quantization.pdf`, `ch11_kv_cache_complete-combined.pdf`, Ch04/Ch05 approved chapter style.

---

## P0 Validation Fixes Before Source Pack

1. Recompute all 70B memory examples and label as raw weight memory.
2. Recompute KV-cache examples using the same formula style as Ch04/Ch11.
3. Replace absolute phrases like "always doubles throughput" with memory-bound, workload-qualified wording.
4. Refresh AMD MI350/CDNA4 wording from official current docs.
5. Refresh FP8/NVFP4 wording from NVIDIA Transformer Engine docs.
6. Mark all benchmark and quality values as representative unless tied to source-specific benchmark citations.
7. Make "validate on your task" a chapter-level rule.

---

## Recommended Safe Language Snippets

### Quantization Definition

```text
Quantization reduces the number of bytes used to represent model values or runtime state. It improves model fit, memory bandwidth pressure, and serving cost only when the reduced precision remains accurate enough for the target task and the runtime has efficient kernels for that format.
```

### Decode Throughput

```text
In memory-bound decode, reducing model or KV bytes can improve tokens/sec because fewer bytes must move through HBM per generated token. The gain is not automatically equal to the byte reduction once scheduling, kernels, communication, or compute become the bottleneck.
```

### FP8

```text
On supported hardware and frameworks, FP8 can reduce memory footprint and accelerate transformer GEMMs using scaling recipes such as NVIDIA Transformer Engine. FP8 is often the lowest-risk first quantization step for H100/H200/Blackwell-class deployments, but task-specific validation is still required.
```

### INT4

```text
INT4 can dramatically reduce weight memory, but naive INT4 is usually unsafe for high-quality LLM serving. Production INT4 typically relies on methods such as AWQ or GPTQ, group-wise scales, calibrated kernels, and task-specific evaluation.
```

### KV Cache Quantization

```text
KV-cache quantization is independent of weight quantization. Reducing KV precision can improve concurrent sequence capacity and long-context headroom when HBM is the limiting resource, but quality and latency must be validated for the target model and serving stack.
```

---

## Source-Pack Instruction

The final Chapter 8 source pack should use the PDF as the structural source of truth while applying this validation plan to soften, correct, and label claims. The HTML should include reader-facing content only; production notes should remain in the Markdown source or be visually separated if included.
