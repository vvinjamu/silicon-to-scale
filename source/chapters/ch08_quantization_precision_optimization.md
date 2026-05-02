# Chapter 8 — Quantization and Precision Optimization

> “Quantization is not about making your model less accurate. It is about using exactly as many bits as the task requires — no more, no less.”

---

## Chapter Overview

Chapter 7 explained how GPU kernels convert model operations into real hardware execution.

Chapter 8 changes the question from:

```text
How do we execute the operation faster?
```

to:

```text
How many bits does this operation actually need?
```

A dense 70B parameter model has a simple raw weight-memory ladder:

```text
FP32: 70B × 4 bytes   = 280 GB
BF16: 70B × 2 bytes   = 140 GB
FP8:  70B × 1 byte    = 70 GB
INT4: 70B × 0.5 byte  = 35 GB
```

That is why quantization is one of the highest-ROI optimizations in AI infrastructure. It reduces model bytes, HBM traffic, KV-cache footprint, memory pressure, and sometimes the number of GPUs needed to serve a model.

This chapter answers:

- Why did BF16 replace FP16 for training?
- Why does FP8 matter for H100/H200/Blackwell-class systems?
- Why does naive INT4 often fail?
- What do scale and zero point actually do?
- Why do per-group scales matter?
- How do GPTQ, AWQ, and SmoothQuant differ?
- Why is KV-cache quantization independent from weight quantization?
- How should a principal engineer choose FP8, INT8, INT4, GGUF, QAT, or BF16 fallback?

> **Current as of 2026 edition:** Quantization support changes quickly across hardware generations, runtimes, kernels, and serving frameworks. Treat model-quality numbers, throughput numbers, and cost/token numbers as workload- and environment-specific unless validated on your own model, prompt distribution, hardware, and serving stack.

---

## 8.0 Quantization in One Page

Quantization reduces numeric precision.

But the production reason to quantize is not merely smaller files.

The production reason is:

```text
fewer bytes moved
  → less HBM pressure
  → more model/KV headroom
  → higher throughput when memory-bound
  → lower cost per token when quality remains acceptable
```

A practical mental model:

```text
Training:
  Use enough range to avoid overflow and unstable gradients.
  BF16 and FP8 mixed precision are training tools.

Inference:
  Use enough precision to preserve task quality.
  FP8, INT8, INT4, FP4, and KV-cache quantization are serving tools.

Serving economics:
  Bytes per parameter and bytes per KV token often dominate capacity and cost.
```

Quantization is not a single technique. It is a stack:

```text
numeric format
  + scaling recipe
  + granularity
  + calibration data
  + kernel support
  + serving runtime
  + task-specific evaluation
```

> **Key Takeaway:** Quantization is a memory-bandwidth and capacity optimization that must be validated as a quality and production-runtime decision.

---

## 8.1 Floating Point Formats — The Precision Landscape

Every floating-point format divides bits across three fields:

```text
sign     → positive or negative
exponent → representable range
mantissa → precision within that range
```

The tradeoff is fundamental:

```text
more exponent bits  → wider range
more mantissa bits  → finer precision
fewer total bits    → lower storage and bandwidth, but more scaling risk
```

For AI training, range is often more important than mantissa precision because gradients can vary by many orders of magnitude. For AI inference, storage and bandwidth often dominate because the model repeatedly reads weights and KV state.

## Figure Placeholder — Fig 8.1

```markdown
![Fig 8.1 — Precision Format Spectrum](../assets/diagrams/svg/ch08_fig_8_1_precision_format_spectrum.svg)

**Fig 8.1 — Precision Format Spectrum.** Numeric formats trade exponent range, mantissa precision, storage size, and hardware/runtime support.
```

**Figure intro:**  
Before choosing FP8, INT8, INT4, or FP4, the reader must understand the bit layout that defines range and precision.

**Figure explanation:**  
The figure should show that BF16 keeps FP32-like exponent range with fewer mantissa bits, while FP8 and FP4 reduce storage and bandwidth but require scaling and hardware/runtime support.

> **Key Takeaway:** Numeric format is a hardware/software contract, not just a smaller data type.

---

## Table 8.1 — Precision Format Bit Layouts

| Format | Sign | Exponent | Mantissa | Total | Practical Use | Confidence |
|---|---:|---:|---:|---:|---|---|
| FP32 | 1 | 8 | 23 | 32 bits | Reference precision, optimizer/master state | `[SHIPPED]` |
| BF16 | 1 | 8 | 7 | 16 bits | Standard modern training/inference baseline | `[SHIPPED]` |
| FP16 | 1 | 5 | 10 | 16 bits | Older mixed precision, vision, some inference | `[SHIPPED]` |
| TF32 | 1 | 8 | 10 effective | 19-bit compute path, 32-bit storage | NVIDIA Tensor Core acceleration for FP32 inputs | `[SHIPPED]` |
| FP8 E4M3 | 1 | 4 | 3 | 8 bits | FP8 weights/activations in many recipes | `[SHIPPED]` |
| FP8 E5M2 | 1 | 5 | 2 | 8 bits | Wider-range FP8, often gradients or KV cache depending stack | `[SHIPPED]` |
| INT8 | integer | — | — | 8 bits | W8A8 inference, activation/weight quantization | `[SHIPPED]` |
| INT4 | integer | — | — | 4 bits | Weight-only low-bit inference, GGUF/AWQ/GPTQ families | `[SHIPPED]` |
| NVFP4 / E2M1-style FP4 | 1 | 2 | 1 | 4 bits plus scaling | Blackwell-class low-precision path | `[SHIPPED]` for supported hardware/software, performance `[ENV-SPECIFIC]` |

[SHIPPED] BF16 keeps the same exponent width as FP32, which gives it FP32-like dynamic range while cutting storage in half. That is why BF16 became a practical default for large-scale model training.

[SHIPPED] NVIDIA Transformer Engine documents FP8 formats such as E4M3 and E5M2, with hybrid recipes that can use E4M3 in the forward path and E5M2 in the backward path depending on scaling strategy and implementation.

[SHIPPED] TF32 is a compute path: FP32 tensors can be consumed by Tensor Cores with reduced mantissa precision while keeping FP32-like exponent range and FP32-style programming behavior.

> **Key Takeaway:** The exponent protects range; the mantissa protects precision; the total bit count controls storage and bandwidth.

---

## 8.2 Model Size Mental Math

The first quantization calculation is raw weight memory:

```text
weight memory = parameter count × bytes per parameter
```

For a dense 70B parameter model:

```text
FP32 weights ≈ 70B × 4 bytes   = 280 GB
BF16 weights ≈ 70B × 2 bytes   = 140 GB
FP8 weights  ≈ 70B × 1 byte    = 70 GB
INT4 weights ≈ 70B × 0.5 byte  = 35 GB
```

[DERIVED FROM SHIPPED] These are raw parameter-storage estimates. Production serving also needs KV cache, activation/workspace buffers, quantization metadata, allocator headroom, runtime overhead, communication buffers, and safety margin.

Do not say:

```text
FP8 always fits a 70B model on one 80 GB GPU.
```

Say:

```text
Raw FP8 weights for a 70B dense model are about 70 GB.
Whether production serving fits on one 80 GB GPU depends on runtime overhead, KV-cache budget, context length, batch/concurrency, tensor parallelism, and serving stack.
```

---

## Table 8.2 — 70B Model Size by Precision

| Format | Bytes / Parameter | Raw 70B Weight Size | What It Usually Means | Confidence |
|---|---:|---:|---|---|
| FP32 | 4.0 | 280 GB | Too large for single-GPU serving; used for master/optimizer states | `[DERIVED FROM SHIPPED]` |
| BF16 | 2.0 | 140 GB | Common training/inference baseline; needs sharding for 80 GB GPUs | `[DERIVED FROM SHIPPED]` |
| FP8 | 1.0 | 70 GB | Can reduce model memory by about 2× vs BF16 | `[DERIVED FROM SHIPPED]` |
| INT8 | 1.0 | 70 GB | Similar raw storage to FP8, different arithmetic and scaling | `[DERIVED FROM SHIPPED]` |
| INT4 | 0.5 | 35 GB | About 4× smaller than BF16 before metadata | `[DERIVED FROM SHIPPED]` |
| FP4/NVFP4-style | ~0.5 plus scales | ~35–40 GB representative | Hardware/runtime-specific; scale overhead matters | `[REPRESENTATIVE]` |

> **Key Takeaway:** Weight memory math is simple. Production fit is not.

---

## 8.3 Quantization Fundamentals — Scale and Zero Point

Quantization maps high-precision values onto a smaller numeric grid.

A common affine quantization form is:

```text
Quantize:
Q(x) = clamp(round(x / scale + zero_point), q_min, q_max)

Dequantize:
x_hat = scale × (Q(x) − zero_point)
```

Where:

```text
x           = original floating-point value
Q(x)        = quantized value
scale       = grid spacing
zero_point  = integer value that represents real zero
q_min/q_max = representable quantized range
x_hat       = reconstructed value
```

For symmetric quantization:

```text
zero_point = 0
scale = max(abs(x)) / q_max
Q(x) = round(x / scale)
```

For asymmetric quantization:

```text
scale = (max(x) − min(x)) / (q_max − q_min)
zero_point ≈ q_min − round(min(x) / scale)
```

[REPRESENTATIVE] Symmetric quantization is simpler and common for weight quantization and hardware-friendly GEMM paths. Asymmetric quantization can use the integer range more fully when distributions are shifted away from zero, but it adds implementation complexity.

Maximum rounding error for a value that is inside range is roughly:

```text
max rounding error ≈ scale / 2
```

That simple formula explains why outliers are dangerous. If one large value makes `scale` large, the entire grid becomes coarse.

## Figure Placeholder — Fig 8.2

```markdown
![Fig 8.2 — Quantization Formula and Dequantization Flow](../assets/diagrams/svg/ch08_fig_8_2_quantization_formula_flow.svg)

**Fig 8.2 — Quantization Formula and Dequantization Flow.** Quantization maps floating-point values to a smaller numeric grid using scale and sometimes zero point.
```

**Figure intro:**  
Quantization is easiest to understand as a mapping problem: continuous values are rounded onto a smaller grid.

**Figure explanation:**  
The figure should show input values, min/max range, computed scale, rounded quantized values, and reconstructed values. It should include symmetric and asymmetric variants but keep symmetric as the primary mental model.

> **Key Takeaway:** Quantization error is controlled by scale, granularity, clipping, and outliers.

> **Key Takeaway:** Scale decides the grid spacing. The grid spacing decides how much information survives.

---

## 8.4 Granularity — The Most Important Quality Lever

A quantized tensor needs scale information.

The question is:

```text
How many values share one scale?
```

The answer is granularity.

Three common choices:

```text
Per-tensor:   one scale for the whole tensor
Per-channel:  one scale per output/input channel or row/column
Per-group:    one scale per small group of weights, such as 64 or 128 values
```

The same INT4 format can be unusable or production-acceptable depending on this choice.

---

## Table 8.3 — Quantization Granularity Comparison

| Granularity | Scale Count | Metadata Overhead | INT8 Quality | INT4 Quality | Typical Use |
|---|---:|---:|---|---|---|
| Per-tensor | 1 per tensor | Lowest | Sometimes acceptable | Often poor | Legacy/simple demos |
| Per-channel | 1 per channel/row | Low | Good | Mixed | INT8 weights/activations, some PTQ |
| Per-group | 1 per group, e.g. 64/128 weights | Moderate | Very good | Often acceptable | GPTQ, AWQ, GGUF-like low-bit formats |
| Per-token KV | 1 scale per token position or block/token policy | Runtime-dependent | Useful for KV cache | Stack-specific | KV-cache quantization |

[REPRESENTATIVE] Group size 128 is common in many low-bit LLM examples, but the best group size is model-, kernel-, runtime-, and quality-budget-specific.

## Figure Placeholder — Fig 8.4

```markdown
![Fig 8.4 — Per-Tensor vs Per-Channel vs Per-Group Quantization](../assets/diagrams/svg/ch08_fig_8_4_granularity_comparison.svg)

**Fig 8.4 — Per-Tensor vs Per-Channel vs Per-Group Quantization.** Finer granularity uses more scale metadata but protects local value distributions and improves low-bit quality.
```

**Figure intro:**  
The same bit width can perform badly or acceptably depending on scale granularity.

**Figure explanation:**  
The figure should show one matrix with one global scale, row/channel scales, and small group scales. The visual should emphasize quality versus metadata overhead.

> **Key Takeaway:** For INT4, granularity is often the difference between unusable and production-acceptable quality.

> **Key Takeaway:** Low-bit quality is often won or lost by scale granularity, not only by the number of bits.

---

## 8.5 The Outlier Problem

Naive quantization fails when one extreme value controls the scale for many normal values.

Suppose an activation tensor has an outlier:

```text
max(|x|) = 72.1
```

For symmetric INT8:

```text
scale = 72.1 / 127 ≈ 0.568
```

A small value such as `0.02` becomes:

```text
Q(0.02) = round(0.02 / 0.568)
        = round(0.035)
        = 0

x_hat = 0 × 0.568 = 0
```

The small value is destroyed.

[REPRESENTATIVE] Transformer outliers can appear in attention, residual streams, normalization-adjacent channels, and activation distributions. The exact distribution is model- and layer-specific, but the failure pattern is stable: a few large values can force a coarse scale that damages many smaller values.

## Figure Placeholder — Fig 8.3

```markdown
![Fig 8.3 — Outlier-Dominated Quantization Error](../assets/diagrams/svg/ch08_fig_8_3_outlier_quantization_error.svg)

**Fig 8.3 — Outlier-Dominated Quantization Error.** One extreme value can force a large scale, causing small but important values to round to zero.
```

**Figure intro:**  
The most common reason naive transformer quantization fails is not the average value; it is the outlier.

**Figure explanation:**  
The figure should compare a normal range with an outlier-expanded range and show how values near zero lose precision when the quantization step becomes too coarse.

> **Key Takeaway:** Outliers can make most values quantize poorly even when the model looks stable in BF16.

This is why the naive idea:

```text
Just convert every weight to INT4.
```

usually fails for large language models.

The better production question is:

```text
Which values need protection, which values can tolerate error, and what scale granularity gives enough quality at acceptable runtime cost?
```

> **Key Takeaway:** Outliers make quantization a distribution problem, not only a bit-width problem.

---

## 8.6 Post-Training Quantization: GPTQ, AWQ, and SmoothQuant

Post-training quantization, or PTQ, tries to quantize a pretrained model without fully retraining it.

Naive PTQ is simple:

```text
1. Run calibration data.
2. Collect min/max or activation statistics.
3. Compute scales.
4. Convert weights/activations to lower precision.
5. Evaluate quality.
```

For INT8, naive or simple calibration may be acceptable for many models.

For INT4 LLMs, naive PTQ is usually not enough.

Three major methods address different failure modes.

### GPTQ

GPTQ is weight-error compensation.

The core idea:

```text
When quantizing one weight column causes error,
adjust later weights so the output error is reduced.
```

GPTQ uses calibration activations and a second-order approximation to estimate which weight errors matter most. It is stronger than naive INT4 because it does not treat every rounding error independently.

[REPRESENTATIVE] GPTQ can deliver production-usable INT4 weight-only models, but calibration time, group size, kernel support, and task-specific quality vary by model and runtime.

### AWQ

AWQ is activation-aware weight quantization.

The core idea:

```text
A small fraction of channels are especially important because their activations are large.
Protect those salient channels before quantizing the weights.
```

AWQ identifies activation-salient channels from calibration data, rescales the corresponding weights and activations so the mathematical output is preserved, then quantizes the weights more safely.

A simplified identity:

```text
Y = X × W
  = (X / s) × (W × s)
```

The output stays the same before quantization, but the scaled weights can use the low-bit range more effectively.

[REPRESENTATIVE] AWQ often calibrates faster than GPTQ and can work well for production INT4 serving, especially where activation outliers dominate. It still requires task-specific evaluation.

### SmoothQuant

SmoothQuant targets W8A8 INT8 deployment.

The core idea:

```text
Activations are hard to quantize because of outliers.
Weights are easier to quantize.
Move some quantization difficulty from activations into weights.
```

The mathematical identity is similar:

```text
Y = X × W
  = (X × diag(s)^-1) × (diag(s) × W)
```

Activations become smoother; weights absorb the scaling.

[REPRESENTATIVE] SmoothQuant is especially useful on hardware and stacks where INT8 Tensor Core or matrix paths are mature but native FP8 is unavailable or not preferred.

---

## Table 8.4 — GPTQ vs AWQ vs SmoothQuant

| Method | Main Target | Core Idea | Calibration Signal | Typical Output | Best Fit | Watchouts |
|---|---|---|---|---|---|---|
| GPTQ | Weight-only low-bit quantization | Compensate weight quantization error using calibration activations | Hessian/second-order approximation | INT4 W4A16-style weights | INT4 serving where GPTQ kernels are strong | Slower calibration; quality varies by task |
| AWQ | Weight-only low-bit quantization | Protect activation-salient weight channels | Activation magnitude / salient channels | INT4 W4A16-style weights | Production INT4 serving, outlier-heavy models | Needs compatible kernels and evals |
| SmoothQuant | W8A8 INT8 | Migrate activation outliers into weights | Activation/weight scale balance | INT8 W8A8 | A100/INT8-friendly stacks | Less relevant if FP8 path is better on target hardware |

## Figure Placeholder — Fig 8.5

```markdown
![Fig 8.5 — GPTQ vs AWQ vs SmoothQuant Method Map](../assets/diagrams/svg/ch08_fig_8_5_ptq_method_map.svg)

**Fig 8.5 — GPTQ vs AWQ vs SmoothQuant Method Map.** GPTQ compensates weight error, AWQ protects activation-salient channels, and SmoothQuant migrates activation outlier difficulty into weights.
```

**Figure intro:**  
GPTQ, AWQ, and SmoothQuant solve related but different quantization problems.

**Figure explanation:**  
The figure should show the input problem, core idea, calibration signal, and output format for each method.

> **Key Takeaway:** Choose the quantization method based on the failure mode.

> **Key Takeaway:** GPTQ fixes weight-error accumulation, AWQ protects salient channels, and SmoothQuant makes activations easier to quantize.

---

## 8.7 FP8 and Transformer Engine

FP8 is not just a smaller storage format.

FP8 is a mixed-precision execution strategy that requires scaling.

A practical FP8 recipe keeps sensitive operations in higher precision and uses FP8 where Tensor Core GEMMs can benefit.

Typical pattern:

```text
GEMM-heavy paths: FP8 inputs with higher-precision accumulation
Norms/residual-sensitive paths: BF16/FP16/FP32 depending recipe
Scaling: dynamic, delayed, current, blockwise, or framework-specific
```

[SHIPPED] NVIDIA Transformer Engine documents FP8 support and scaling recipes for Hopper and newer NVIDIA GPUs. FP8 E4M3 and E5M2 have different range/precision tradeoffs, and hybrid recipes can use different formats for forward and backward tensors.

[ENV-SPECIFIC] FP8 can improve training or inference throughput when GEMM time and memory traffic are major bottlenecks, but end-to-end speedup depends on non-GEMM operations, communication, batch shape, kernels, compiler/runtime behavior, and quality constraints.

A simplified FP8 flow:

```text
Track tensor amax
  → choose scale
  → cast BF16/FP32 tensor to FP8
  → run Tensor Core GEMM
  → accumulate in higher precision
  → descale output
  → continue mixed-precision pipeline
```

## Figure Placeholder — Fig 8.6

```markdown
![Fig 8.6 — FP8 Transformer Engine Dynamic Scaling Pipeline](../assets/diagrams/svg/ch08_fig_8_6_fp8_transformer_engine_pipeline.svg)

**Fig 8.6 — FP8 Transformer Engine Dynamic Scaling Pipeline.** Transformer Engine monitors tensor ranges, selects scales, executes FP8 Tensor Core GEMMs, and converts outputs back into the mixed-precision stack.
```

**Figure intro:**  
FP8 is not just a smaller format; it requires scaling recipes that keep tensors inside representable range.

**Figure explanation:**  
The figure should show BF16/FP32 source tensors, amax history, scale computation, FP8 casting, Tensor Core GEMM, accumulation, descale, and output path.

> **Key Takeaway:** FP8 quality depends on recipe, scaling, accumulation, and which tensors remain in higher precision.

### NVFP4 and FP4

FP4 pushes the same idea further: fewer bits, more need for scaling and hardware/runtime support.

[SHIPPED] NVIDIA Blackwell-class software documentation describes NVFP4 as an E2M1-style 4-bit floating format with block scaling support in Transformer Engine contexts.

[ENV-SPECIFIC] FP4 performance and quality depend heavily on Blackwell-class hardware support, scaling recipe, model architecture, calibration/fine-tuning path, and serving framework. Do not compare FP4 peak numbers directly to BF16 or FP8 without specifying dense/sparse mode, accumulation type, batch shape, and quality target.

> **Key Takeaway:** FP8 and FP4 are not only formats. They are format plus scaling plus kernel plus runtime plus quality validation.

---

## 8.8 KV Cache Quantization

The KV cache is independent from weight quantization.

A model can use:

```text
FP8 weights + BF16 KV
BF16 weights + FP8 KV
INT4 weights + FP8 KV
AWQ weights + INT8 KV
```

The KV-cache memory estimate should stay consistent with Chapters 4 and 11:

```text
KV cache bytes ≈ 2 × L × B × S × n_kv × d_head × bytes_per_element
```

Where:

```text
2       = K and V tensors
L       = number of transformer layers
B       = concurrent sequences / batch
S       = sequence length
n_kv    = number of KV heads, not query heads
 d_head = head dimension
bytes   = bytes per KV element
```

For one sequence, remove `B` or set `B = 1`.

Example for a LLaMA-like 70B model with GQA-8, `L = 80`, `n_kv = 8`, `d_head = 128`, `S = 4096`:

```text
BF16 KV per sequence
≈ 2 × 80 × 1 × 4096 × 8 × 128 × 2 bytes
≈ 1.34 GB

FP8/INT8 KV per sequence
≈ 2 × 80 × 1 × 4096 × 8 × 128 × 1 byte
≈ 0.67 GB
```

[DERIVED FROM SHIPPED] These are formula-derived estimates. Exact production memory depends on block size, metadata, page tables, allocator behavior, prefix sharing, fragmentation, runtime overhead, and serving stack.

## Figure Placeholder — Fig 8.7

```markdown
![Fig 8.7 — KV Cache Quantization Memory Reduction](../assets/diagrams/svg/ch08_fig_8_7_kv_cache_quantization_reduction.svg)

**Fig 8.7 — KV Cache Quantization Memory Reduction.** KV-cache precision reduction cuts memory footprint per token and can increase concurrent sequence capacity when HBM is the bottleneck.
```

**Figure intro:**  
Weight quantization and KV-cache quantization solve different memory problems and can be combined.

**Figure explanation:**  
The figure should compare how many sequence blocks fit with BF16 KV versus FP8/INT8 KV and separate KV precision from GQA/MQA architecture choices.

> **Key Takeaway:** KV-cache quantization improves concurrency and context capacity, not necessarily single-token compute speed.

---

## Table 8.5 — Weight Quantization vs KV-Cache Quantization

| Technique | Reduces | Usually Improves | Does Not Automatically Improve | Key Risk | Confidence |
|---|---|---|---|---|---|
| FP8 weights | Weight memory and weight bandwidth | Model fit, prefill/decode throughput when weight reads dominate | Task quality without validation | Runtime/kernel support | `[SHIPPED]` / `[ENV-SPECIFIC]` |
| INT4/AWQ/GPTQ weights | Weight memory strongly | Model fit, cost/token, decode bandwidth pressure | Quality-sensitive tasks | Accuracy regressions, kernel support | `[REPRESENTATIVE]` |
| FP8/INT8 KV cache | KV-cache capacity and bandwidth | Concurrency, long-context headroom | Raw model-weight size | Attention quality, scale handling | `[SHIPPED]` / `[ENV-SPECIFIC]` |
| GQA/MQA | Number of KV heads | KV memory and decode bandwidth | Weight memory | Architecture-specific; not a deployment switch | `[SHIPPED]` when model architecture uses it |
| Prefix caching | Redundant prefill compute and KV duplication | TTFT and capacity for shared-prefix workloads | Random unrelated prompts | Cache hit rate and routing | `[ENV-SPECIFIC]` |

> **Key Takeaway:** Weight quantization reduces model bytes. KV-cache quantization reduces serving-state bytes. Production systems often need both.

---

## 8.9 Quantization-Aware Training and QLoRA

Post-training quantization is not always enough.

Quantization-aware training, or QAT, exposes the model to quantization error during training or fine-tuning so weights can adapt.

A simplified fake-quantization path:

```text
forward pass:
  x_quant_like = round(x / scale) × scale

backward pass:
  use straight-through estimator so learning can continue
```

[REPRESENTATIVE] QAT can recover quality for aggressive low-bit formats, but it is much more expensive than PTQ because it requires training or fine-tuning cycles. It is usually justified only when deployment volume is high enough to repay the extra engineering and compute cost.

### QLoRA

QLoRA is a fine-tuning method, not a general-purpose serving quantization format.

The core idea:

```text
Keep the base model in 4-bit NF4.
Dequantize on the fly for compute.
Train small BF16 LoRA adapter weights.
Update only the adapter weights.
```

QLoRA is useful when full fine-tuning is too expensive, but you still need task adaptation.

[REPRESENTATIVE] QLoRA memory usage depends on model size, adapter rank, sequence length, activation checkpointing, optimizer state, and framework implementation. Treat single-GPU fine-tuning examples as workload-specific, not universal.

> **Key Takeaway:** PTQ is usually the first production step; QAT and QLoRA are used when quality recovery or fine-tuning economics justify the extra complexity.

---

## 8.10 Quantization on AMD and ROCm

Quantization is not only an NVIDIA topic.

AMD Instinct systems also support low-precision inference and training paths through ROCm, HIP, Composable Kernel, vLLM ROCm support, and framework/runtime integrations.

[SHIPPED] AMD ROCm documentation describes FP8 quantization support for MI300X inference workflows, including FP8 weight/activation quantization with hardware acceleration in supported stacks.

[ENV-SPECIFIC] Performance and feature availability on AMD depend on ROCm version, model, kernel path, vLLM/serving version, quantization format, and whether the workload uses supported optimized kernels.

A safe production framing:

```text
For MI300X/ROCm:
  evaluate FP8 first when the runtime and model path support it;
  evaluate AWQ/INT4 when memory or bandwidth pressure dominates;
  validate accuracy and kernel availability before making a procurement or deployment claim.
```

MI300X has a large-HBM advantage for memory-heavy inference. A raw 70B BF16 model is about 140 GB, and MI300X-class 192 GB HBM capacity can reduce the need for tensor parallelism in some serving configurations. But raw fit does not guarantee production fit.

[DERIVED FROM SHIPPED] Larger HBM capacity can improve model-fit and KV-cache headroom, but final capacity depends on runtime overhead, context length, concurrency, tensor parallelism, and quantization strategy.

### CDNA 4 / MI350-Class Claims

[ANNOUNCED] AMD has discussed newer low-precision formats and MI350-class roadmap/product capabilities. Use current official AMD product and ROCm documentation before making final FP4/FP6 performance or availability claims.

> **Key Takeaway:** AMD quantization support should be discussed as hardware plus ROCm plus serving runtime plus kernel maturity, not as a paper format alone.

---

## 8.11 Deployment Decision Framework

A production quantization decision starts with constraints.

Do not start with:

```text
Which quantization method is popular?
```

Start with:

```text
What must be true for this deployment to succeed?
```

## Figure Placeholder — Fig 8.8

```markdown
![Fig 8.8 — Production Quantization Decision Tree](../assets/diagrams/svg/ch08_fig_8_8_quantization_decision_tree.svg)

**Fig 8.8 — Production Quantization Decision Tree.** A production quantization decision starts with model fit and hardware support, then filters by quality, calibration time, serving stack, and task-specific evaluation.
```

**Figure intro:**  
Quantization should not be chosen by trend or format name. It should be chosen by constraints.

**Figure explanation:**  
The figure should guide the reader from hardware and memory constraints toward FP8, SmoothQuant, AWQ, GPTQ, GGUF, QAT, or BF16 fallback.

> **Key Takeaway:** A production quantization choice is a constraint-satisfaction problem.

---

## Table 8.6 — Production Quantization Deployment Framework

| Step | Question | Decision Logic | Example Outcome |
|---|---|---|---|
| 0 | Does the model fit comfortably at BF16? | If not, quantization or sharding is required. | 70B BF16 is about 140 GB, so one 80 GB GPU is not enough for raw weights. |
| 1 | What hardware is targeted? | H100/H200/Blackwell may favor FP8/FP4 paths; A100 may favor INT8/SmoothQuant; CPU may favor GGUF. | Hardware chooses the realistic format menu. |
| 2 | What is the quality budget? | Tight budget favors BF16/FP8; moderate budget may allow AWQ/GPTQ INT4. | Code/math may reject INT4 even when MMLU looks fine. |
| 3 | What is the memory target? | 2× weight reduction suggests FP8/INT8; 4× suggests INT4. | INT4 can reduce GPU count but increases quality risk. |
| 4 | What calibration budget is available? | FP8 may require little or no offline calibration; AWQ/GPTQ require calibration; QAT is expensive. | Choose method based on engineering time and ROI. |
| 5 | What serving stack supports it well? | Kernel/runtime support decides whether theoretical savings become real. | vLLM/TRT-LLM/SGLang/ROCm support must be checked. |
| 6 | Did task-specific eval pass? | Never ship on aggregate benchmarks only. | Run MMLU plus real production prompts and failure cases. |

> **Key Takeaway:** Quantization is not complete until the deployment task passes evaluation under the intended serving stack.

---

## 8.12 Production Decision Table

The following table is a starting point, not a universal rule.

---

## Table 8.7 — Format Selection by Use Case

| Use Case | Recommended Starting Point | Second Choice | Avoid / Be Careful With |
|---|---|---|---|
| H100/H200 production serving, high quality | FP8 W8A8 + FP8 KV eval | AWQ INT4 if cost pressure dominates | Naive INT4 |
| A100 production serving | SmoothQuant W8A8 or AWQ INT4 | GPTQ INT4 | FP8 if not supported by the stack |
| Blackwell-class maximum performance | FP8 or NVFP4 path with official stack | AWQ/GPTQ for compatibility | Comparing peak FP4 to BF16 without quality and mode labels |
| AMD MI300X production | FP8 ROCm/vLLM path if supported | AWQ INT4 | Assuming NVIDIA-specific kernels apply |
| Local CPU inference | GGUF Q4/Q5 family | GGUF Q6 for quality | GPU-only AWQ/GPTQ assumptions |
| Code/math quality-sensitive serving | FP8 or BF16 fallback | AWQ only after code/math eval | Shipping based on MMLU alone |
| Long-context serving | Weight quantization + FP8/INT8 KV + GQA/MQA architecture | Prefix caching and routing | Ignoring KV-cache budget |
| Fine-tuning with limited memory | QLoRA/NF4 + LoRA adapters | Smaller model full fine-tune | Treating QLoRA as the same as production serving quantization |
| Research baseline | BF16 | FP8 if validated | Low-bit formats before baseline quality is known |

[REPRESENTATIVE] This table is a decision guide. Real deployments should validate latency, throughput, quality, memory, and cost on target hardware and workload.

---

## 8.13 Quantization Evaluation Checklist

Quantization is not ready when the model loads.

It is ready when it passes a staged evaluation.

---

## Table 8.8 — Quantization Evaluation Checklist

| Layer | What to Test | Why It Matters | Pass/Fail Signal |
|---|---|---|---|
| Language modeling | Perplexity on held-out data | Detects broad degradation | PPL increase within budget |
| Broad knowledge | MMLU / HellaSwag / ARC / TruthfulQA | Catches general capability loss | Delta within quality budget |
| Task-specific | Your production prompts and labels | Aggregate benchmarks can hide failures | No unacceptable business/task regression |
| Code/math | HumanEval, MBPP, GSM8K, MATH if relevant | Low-bit quantization can hurt reasoning precision | Delta acceptable for product |
| Latency | TTFT, TPOT, P50/P95/P99 | Quantization can change scheduler and kernel behavior | SLA met under load |
| Throughput | tokens/sec, requests/sec | Cost and capacity planning | Sustained throughput improves |
| Memory | weights, KV, fragmentation, overhead | Raw model size is not enough | Headroom remains at target concurrency |
| Stability | long-run serving, mixed prompts, long context | Quantized kernels may have edge cases | No unacceptable crash/quality drift |
| Observability | per-format metrics and rollback path | Production safety | Can compare and revert quickly |

> **Key Takeaway:** A quantized model must pass quality, performance, memory, stability, and rollback tests.

---

## 8.14 Wrong Fix vs Right First Question

Quantization failures are often misdiagnosed.

A senior engineer may jump to a different format.

A principal engineer asks which constraint failed.

---

## Table 8.9 — Wrong Fix vs Right First Question for Quantization Problems

| Symptom | Wrong Fix | Right First Question |
|---|---|---|
| INT4 quality collapses | Try random lower-bit format | Is scale granularity too coarse or are outliers dominating? |
| MMLU looks fine but code quality drops | Ship anyway | Did we evaluate the production task and code/math benchmarks? |
| FP8 speedup is small | Claim FP8 does not work | Is the workload GEMM-bound, memory-bound, communication-bound, or scheduler-bound? |
| Model fits by raw weight math but OOMs | Blame framework | Did we include KV cache, runtime overhead, fragmentation, and workspace buffers? |
| KV quantization increases errors | Disable all quantization | Is the issue KV scale, dtype, long context, or attention kernel support? |
| AMD deployment underperforms NVIDIA example | Blame hardware | Are ROCm version, kernel path, model format, and serving stack comparable? |

> **Key Takeaway:** Quantization is a multi-layer system. Diagnose the failed layer before changing the format.

---

## 8.15 How to Discuss Quantization in a Principal Interview

A weak answer sounds like this:

```text
Quantization makes the model smaller and faster.
```

That is true, but too shallow.

A principal-level answer sounds like this:

> I treat quantization as a memory-bandwidth, capacity, and quality tradeoff. First I calculate raw weight memory and KV-cache memory. Then I check hardware support, kernel/runtime support, and quality budget. For H100/H200-class inference, I usually evaluate FP8 first because it offers strong quality and hardware support. If cost or memory pressure remains high, I evaluate AWQ or GPTQ INT4 with task-specific evals. I never ship based on MMLU alone; I validate the production task, latency percentiles, KV-cache behavior, and rollback path.

### Scenario 1 — Why Is Quantization a Decode Optimization?

Answer:

```text
LLM decode is often memory-bandwidth-sensitive because each output token repeatedly touches model weights and prior KV state. Reducing weight bytes from BF16 to FP8 or INT4 reduces HBM traffic. When decode is memory-bound, fewer bytes can improve tokens/sec and cost/token. If decode is not memory-bound, the speedup may be smaller.
```

### Scenario 2 — Why Did BF16 Replace FP16 for Training?

Answer:

```text
BF16 keeps FP32-like exponent range with fewer mantissa bits. FP16 has more mantissa precision but much smaller range. Large-model training needs range to avoid overflow and NaNs in gradients, so BF16 is safer for training even though it has less mantissa precision.
```

### Scenario 3 — FP8 or AWQ INT4?

Answer:

```text
I start with hardware and quality budget. On H100/H200/Blackwell, FP8 is often the first candidate because the hardware and Transformer Engine-style recipes are mature and quality can be near BF16 for many tasks. AWQ INT4 is attractive when memory or cost pressure is stronger, but I validate quality carefully, especially for code, math, and domain-specific tasks.
```

### Scenario 4 — Why Is MMLU Not Enough?

Answer:

```text
MMLU is a useful broad benchmark, but it is not the production task. A quantized model can lose little on MMLU and still regress on code completion, math, structured extraction, or safety-critical prompts. I run broad benchmarks plus the target workload's real eval set before shipping.
```

### Scenario 5 — How Would You Deploy a 70B Model on H100?

Answer:

```text
I calculate raw memory first: BF16 is about 140 GB, FP8 is about 70 GB, INT4 is about 35 GB. On an 80 GB H100, raw FP8 weights may fit, but production serving also needs KV cache and runtime headroom. I would evaluate FP8 weights, FP8 KV cache, and no tensor parallelism only if overhead and context targets fit. If not, I would use tensor parallelism or INT4 weight-only quantization, then validate TTFT, TPOT, P95/P99, and task quality.
```

---

## 8.16 Chapter Cheat Sheet

### Raw Weight Memory

```text
weight memory = parameters × bytes per parameter

70B FP32 ≈ 280 GB
70B BF16 ≈ 140 GB
70B FP8  ≈ 70 GB
70B INT4 ≈ 35 GB
```

### Quantization Formula

```text
Q(x) = clamp(round(x / scale + zero_point), q_min, q_max)
x_hat = scale × (Q(x) − zero_point)
```

### Symmetric Quantization

```text
zero_point = 0
scale = max(abs(x)) / q_max
```

### Maximum Rounding Error

```text
max error ≈ scale / 2
```

### KV Cache Formula

```text
KV cache bytes ≈ 2 × L × B × S × n_kv × d_head × bytes_per_element
```

### Production Rule

```text
Do not ship a quantized model until the production task passes evaluation.
```

---

## 8.17 Key Takeaways

1. Quantization is a bandwidth, capacity, and cost optimization.
2. FP32, BF16, FP16, TF32, FP8, INT8, INT4, and FP4 differ in range, precision, storage, and hardware support.
3. BF16 became a training default because it preserves FP32-like exponent range.
4. FP8 is a format plus scaling recipe plus hardware/runtime path.
5. Raw 70B weight memory is about 280 GB in FP32, 140 GB in BF16, 70 GB in FP8, and 35 GB in INT4.
6. Raw fit is not production fit; KV cache and runtime headroom matter.
7. Quantization maps values onto a smaller grid using scale and sometimes zero point.
8. Outliers can force large scale values and destroy small-magnitude information.
9. Granularity is one of the most important quality levers.
10. GPTQ compensates weight quantization error.
11. AWQ protects activation-salient channels.
12. SmoothQuant migrates activation outlier difficulty into weights.
13. KV-cache quantization is independent from weight quantization.
14. QAT can recover quality but costs far more than PTQ.
15. QLoRA is a fine-tuning strategy, not simply a serving quantization format.
16. AMD/ROCm quantization decisions require hardware, ROCm, kernel, and serving-stack validation.
17. MMLU is not enough; production tasks need their own evals.
18. Principal-level quantization decisions start with constraints and end with measured quality, latency, memory, and cost.

---

## 8.18 Review Questions

### Conceptual

1. Why is quantization more than model compression?
2. Why did BF16 replace FP16 for large-model training?
3. What is the difference between exponent range and mantissa precision?
4. Why is TF32 called a compute path rather than a storage format?
5. Why does FP8 require scaling recipes?
6. What is the difference between FP8 E4M3 and E5M2?
7. Why is naive INT4 risky for large transformers?
8. What does scale do in quantization?
9. What is zero point?
10. Why does per-group quantization usually outperform per-tensor quantization for INT4?
11. What is the outlier problem?
12. How do GPTQ, AWQ, and SmoothQuant differ?
13. Why is KV-cache quantization independent from weight quantization?
14. Why is GGUF not the same kind of deployment choice as TensorRT-LLM FP8?
15. Why should AMD quantization claims be tied to ROCm version and serving stack?

### Calculation

1. Estimate FP32, BF16, FP8, and INT4 raw weight memory for a 70B parameter model.
2. For symmetric INT8 with `max(abs(x)) = 72.1`, compute the scale.
3. Using that scale, quantize and dequantize `x = 0.02`. What information is lost?
4. Write the KV-cache memory formula and explain each variable.
5. Estimate BF16 KV memory for one 4096-token sequence with `L=80`, `n_kv=8`, `d_head=128`.
6. Estimate FP8 KV memory for the same sequence.
7. If KV precision changes from BF16 to FP8, what happens to raw KV bytes before metadata?
8. If weight precision changes from BF16 to INT4, what happens to raw weight bytes before metadata?

### Principal-Level Interview Practice

1. Explain quantization as a memory-bandwidth optimization for decode.
2. Explain how you would choose between FP8 and AWQ INT4 on H100.
3. Explain why a 70B FP8 model may still fail to serve on one 80 GB GPU.
4. Explain why MMLU alone is not enough for quantization validation.
5. Explain how you would debug INT4 quality collapse.
6. Explain how KV-cache quantization changes serving capacity.
7. Explain what additional checks you would run before shipping a quantized model in production.
8. Explain how you would evaluate AMD MI300X quantization support without assuming NVIDIA-specific kernels.
9. Explain when you would use QAT instead of PTQ.
10. Explain QLoRA in a way that distinguishes fine-tuning memory savings from production serving quantization.

---

## 8.19 Production Notes for This Chapter

### Figure Assets Needed

| Figure | Status |
|---|---|
| Fig 8.1 — Precision Format Spectrum | Adapt from `diagrams_batch1.html#d10` |
| Fig 8.2 — Quantization Formula Flow | Must be created |
| Fig 8.3 — Outlier-Dominated Quantization Error | Adapt from `diagrams_batch2.html#d18` |
| Fig 8.4 — Quantization Granularity Comparison | Must be created |
| Fig 8.5 — GPTQ vs AWQ vs SmoothQuant | Must be created |
| Fig 8.6 — FP8 Transformer Engine Pipeline | Must be created |
| Fig 8.7 — KV Cache Quantization Reduction | Adapt from `diagram_02_kv_cache.html` |
| Fig 8.8 — Production Quantization Decision Tree | Must be created |

### Table Assets Included

| Table | Status |
|---|---|
| Table 8.1 — Precision Format Bit Layouts | Included |
| Table 8.2 — 70B Model Size by Precision | Included |
| Table 8.3 — Quantization Granularity Comparison | Included |
| Table 8.4 — GPTQ vs AWQ vs SmoothQuant | Included |
| Table 8.5 — Weight vs KV Quantization | Included |
| Table 8.6 — Deployment Framework | Included |
| Table 8.7 — Format Selection by Use Case | Included |
| Table 8.8 — Evaluation Checklist | Included |
| Table 8.9 — Wrong Fix vs Right First Question | Included |

### Source Notes to Add in Final Book

Use official or primary sources for:

- NVIDIA Transformer Engine FP8 and FP4/NVFP4 documentation
- NVIDIA Hopper/Blackwell architecture and Tensor Core documentation
- vLLM quantized KV-cache documentation
- TensorRT-LLM quantization and KV-cache documentation
- AMD ROCm MI300X precision and inference optimization documentation
- AMD MI350/CDNA4 product or ROCm documentation before final claims
- GPTQ paper: Frantar et al., 2022
- AWQ paper: Lin et al., 2023
- SmoothQuant paper: Xiao et al., 2022
- QLoRA paper: Dettmers et al., 2023

### HTML Production Rule

The HTML version should remove this Production Notes section and end with a reader-facing bridge to Chapter 9.
