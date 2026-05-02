# Chapter 8 Figure Integration Plan — Quantization and Precision Optimization

**Book:** AI/ML Infrastructure from Silicon to Scale  
**Chapter:** Ch08 — Quantization and Precision Optimization  
**Primary source:** `ch08_quantization.pdf`  
**Purpose:** Ensure the Chapter 8 source and HTML match the PDF while using existing diagram assets where possible.

---

## Figure Plan Summary

| Asset | Count |
|---|---:|
| Planned figures | 8 |
| Existing/adaptable diagram sources | 3 |
| Figures that must be created | 5 |
| Planned tables | 7 |

---

# Figures

## Fig 8.1 — Precision Format Spectrum

| Field | Value |
|---|---|
| Number | Fig 8.1 |
| Title | Precision Format Spectrum: FP32, TF32, BF16, FP16, FP8, INT8, INT4, FP4 |
| Existing source file if available | `../diagrams/diagrams_batch1.html#d10` |
| Exists or must be created | Exists/adapt from Mixed Precision diagram |
| Exact section placement | After Section 8.1.1, before Table 8.1 |
| Caption | **Fig 8.1 — Precision Format Spectrum.** Numeric formats trade exponent range, mantissa precision, storage size, and hardware support. AI training favors range; inference often favors reduced bytes and sufficient quality. |
| Intro paragraph | Before choosing FP8, INT8, INT4, or FP4, the reader must understand the bit layout that defines range and precision. |
| Explanation paragraph | The figure should show that BF16 keeps FP32-like exponent range with fewer mantissa bits, while FP8 and FP4 reduce storage and bandwidth but require scaling and hardware/runtime support. |
| Key takeaway | Numeric format is a hardware/software contract, not just a smaller data type. |
| Web-readiness | Link to source diagram and add figure placeholder card in HTML. |
| Print-readiness | Export 300 DPI PNG or SVG with labels visible in grayscale. |
| Required production fixes | Update title from generic mixed precision to Chapter 8 terminology; avoid tiny text. |

---

## Fig 8.2 — Quantization Formula and Scale/Zero-Point Flow

| Field | Value |
|---|---|
| Number | Fig 8.2 |
| Title | Quantization Formula: Scale, Zero Point, Quantize, Dequantize |
| Existing source file if available | None |
| Exists or must be created | Must be created |
| Exact section placement | Section 8.2.1, immediately after the quantization formula |
| Caption | **Fig 8.2 — Quantization Formula and Dequantization Flow.** Quantization maps floating-point values to a smaller integer or float grid using scale and sometimes zero point; dequantization maps them back for computation. |
| Intro paragraph | Quantization is easiest to understand as a mapping problem: continuous values are rounded onto a smaller grid. |
| Explanation paragraph | The figure should show input values, min/max range, computed scale, rounded quantized values, and reconstructed values. It should include symmetric and asymmetric variants but keep symmetric as the primary mental model. |
| Key takeaway | Quantization error is controlled by scale, granularity, clipping, and outliers. |
| Web-readiness | Use a simple flow diagram; avoid dense equations in the image. |
| Print-readiness | Use high-contrast arrows and formula labels. |
| Required production fixes | Ensure equations match table/formula text exactly. |

---

## Fig 8.3 — Outlier-Dominated Quantization Error

| Field | Value |
|---|---|
| Number | Fig 8.3 |
| Title | Outlier-Dominated Quantization Error |
| Existing source file if available | `../diagrams/diagrams_batch2.html#d18` |
| Exists or must be created | Exists/adapt from Quant Error diagram |
| Exact section placement | Section 8.2.3, after the outlier example |
| Caption | **Fig 8.3 — Outlier-Dominated Quantization Error.** One extreme activation can force a large scale, causing small but important values to round to zero. |
| Intro paragraph | The most common reason naive quantization fails in transformers is not the average value; it is the outlier. |
| Explanation paragraph | The figure should compare a normal range with an outlier-expanded range and show how values near zero lose precision when the quantization step becomes too coarse. |
| Key takeaway | Outliers can make most values quantize poorly even when the model looks numerically stable in BF16. |
| Web-readiness | Link to diagram pack and include concise caption. |
| Print-readiness | Export with both values and bin boundaries visible. |
| Required production fixes | Label it as representative; do not imply every model has the same outlier distribution. |

---

## Fig 8.4 — Quantization Granularity Comparison

| Field | Value |
|---|---|
| Number | Fig 8.4 |
| Title | Per-Tensor vs Per-Channel vs Per-Group Quantization |
| Existing source file if available | None |
| Exists or must be created | Must be created |
| Exact section placement | Section 8.2.2, after Table 8.3 |
| Caption | **Fig 8.4 — Quantization Granularity Comparison.** Finer granularity uses more scale metadata but protects local value distributions and improves low-bit quality. |
| Intro paragraph | The same INT4 bit width can perform badly or acceptably depending on scale granularity. |
| Explanation paragraph | Show one matrix with one global scale, row/channel scales, and small group scales. Highlight quality vs overhead tradeoff. |
| Key takeaway | For INT4, granularity is often the difference between unusable and production-acceptable quality. |
| Web-readiness | Simple matrix visual with three columns. |
| Print-readiness | Avoid relying only on color; label each scale group. |
| Required production fixes | Use group size examples as representative, not universal. |

---

## Fig 8.5 — GPTQ vs AWQ vs SmoothQuant

| Field | Value |
|---|---|
| Number | Fig 8.5 |
| Title | Post-Training Quantization Method Map |
| Existing source file if available | None |
| Exists or must be created | Must be created |
| Exact section placement | Section 8.3, after GPTQ/AWQ/SmoothQuant explanations |
| Caption | **Fig 8.5 — GPTQ vs AWQ vs SmoothQuant.** GPTQ compensates weight error, AWQ protects activation-salient channels, and SmoothQuant migrates activation outlier difficulty into weights. |
| Intro paragraph | GPTQ, AWQ, and SmoothQuant solve related but different quantization problems. |
| Explanation paragraph | The figure should show the input problem, core idea, calibration signal, and output format for each method. |
| Key takeaway | Choose the quantization method based on the failure mode: weight error, activation salience, or activation outliers. |
| Web-readiness | Use as a comparison card or flowchart. |
| Print-readiness | Keep text blocks short; export wide enough for print. |
| Required production fixes | Cite papers in source notes; avoid claiming one method always wins. |

---

## Fig 8.6 — FP8 Transformer Engine Dynamic Scaling Pipeline

| Field | Value |
|---|---|
| Number | Fig 8.6 |
| Title | FP8 Transformer Engine Dynamic Scaling Pipeline |
| Existing source file if available | None |
| Exists or must be created | Must be created |
| Exact section placement | Section 8.4.1, after FP8 dynamic scaling explanation |
| Caption | **Fig 8.6 — FP8 Transformer Engine Dynamic Scaling Pipeline.** Transformer Engine monitors tensor ranges, selects scales, executes FP8 Tensor Core GEMMs, and converts outputs back into the mixed-precision training or inference stack. |
| Intro paragraph | FP8 is not just a smaller format; it requires scaling recipes that keep tensors inside representable range. |
| Explanation paragraph | Show BF16/FP32 source tensor, amax history, scale computation, FP8 cast, Tensor Core GEMM, accumulation, and descale/output path. |
| Key takeaway | FP8 quality depends on recipe, scaling, accumulation, and which tensors stay in higher precision. |
| Web-readiness | Useful as a flow diagram with formula callouts. |
| Print-readiness | Use readable labels; avoid code-only screenshots. |
| Required production fixes | Align wording with NVIDIA Transformer Engine official FP8 docs. |

---

## Fig 8.7 — KV Cache Quantization Memory Reduction

| Field | Value |
|---|---|
| Number | Fig 8.7 |
| Title | KV Cache Quantization: BF16 vs FP8/INT8 |
| Existing source file if available | `../diagrams/diagram_02_kv_cache.html`; optional support from `../diagrams/diagrams_batch3.html#d24` |
| Exists or must be created | Existing KV cache diagram can be adapted; new overlay needed |
| Exact section placement | Section 8.5, after KV-cache quantization formula/table |
| Caption | **Fig 8.7 — KV Cache Quantization Memory Reduction.** KV-cache precision reduction cuts memory footprint per token and can increase concurrent sequence capacity when HBM is the bottleneck. |
| Intro paragraph | Weight quantization and KV-cache quantization solve different memory problems and can be combined. |
| Explanation paragraph | Show a fixed HBM budget and compare how many sequence blocks fit with BF16 KV versus FP8/INT8 KV. Mention GQA/MQA as separate architectural multipliers. |
| Key takeaway | KV-cache quantization improves concurrency and context capacity, not necessarily single-token compute speed. |
| Web-readiness | Link to existing KV diagram and add Ch08-specific placeholder. |
| Print-readiness | Show math in caption and use simple blocks. |
| Required production fixes | Align all values with Ch04/Ch11 KV formula. |

---

## Fig 8.8 — Quantization Deployment Decision Tree

| Field | Value |
|---|---|
| Number | Fig 8.8 |
| Title | Production Quantization Decision Tree |
| Existing source file if available | None |
| Exists or must be created | Must be created |
| Exact section placement | Section 8.8, before Table 8.6 |
| Caption | **Fig 8.8 — Production Quantization Decision Tree.** A production quantization decision starts with model fit and hardware support, then filters by quality budget, calibration time, serving stack, and task-specific evaluation. |
| Intro paragraph | Quantization should not be chosen by trend or format name. It should be chosen by constraints. |
| Explanation paragraph | Decision branches should include: model fits at BF16, hardware supports FP8, quality tolerance, calibration budget, CPU/edge deployment, and KV-cache constraints. |
| Key takeaway | Production quantization is a decision workflow, not a single best format. |
| Web-readiness | Keep decision tree compact; avoid too many branches. |
| Print-readiness | Use large text and clear branching. |
| Required production fixes | Validate hardware/platform recommendations with current official docs. |

---

# Tables

## Table 8.1 — Precision Format Bit Layouts

| Field | Value |
|---|---|
| Number | Table 8.1 |
| Title | Precision Format Bit Layouts |
| Existing source file if available | `ch08_quantization.pdf` |
| Exists or must be created | Exists in PDF; convert to Markdown table |
| Exact section placement | Section 8.1 |
| Caption | **Table 8.1 — Precision Format Bit Layouts.** Bit allocation determines range, precision, storage cost, and hardware execution behavior. |
| Intro paragraph | This table establishes the vocabulary for the rest of the chapter. |
| Explanation paragraph | Explain that exponent bits control dynamic range and mantissa bits control precision. |
| Key takeaway | BF16 keeps FP32 range; FP8/FP4 reduce bytes but require recipes and hardware support. |
| Web-readiness | Responsive table; break long dynamic-range text if needed. |
| Print-readiness | Avoid overly wide columns. |
| Required production fixes | Verify FP8/FP4 ranges and mark product-specific behavior. |

---

## Table 8.2 — 70B Model Size by Precision

| Field | Value |
|---|---|
| Number | Table 8.2 |
| Title | 70B Model Size by Precision |
| Existing source file if available | `ch08_quantization.pdf` |
| Exists or must be created | Exists in PDF; convert and simplify |
| Exact section placement | Section 8.1.2 |
| Caption | **Table 8.2 — 70B Model Size by Precision.** Raw weight memory scales directly with bytes per parameter before metadata, sharding, activations, KV cache, and runtime buffers. |
| Intro paragraph | A simple 70B model memory ladder makes the value of quantization concrete. |
| Explanation paragraph | The table should clearly separate raw weight memory from deployable serving memory. |
| Key takeaway | FP8 halves BF16 raw weight memory; INT4 quarters it. |
| Web-readiness | Include labels `[DERIVED FROM SHIPPED]` or `[ESTIMATED]` for computed values. |
| Print-readiness | Keep formulas in caption/notes. |
| Required production fixes | Do not state "fits on one H100" without KV/runtime caveats. |

---

## Table 8.3 — Quantization Granularity Comparison

| Field | Value |
|---|---|
| Number | Table 8.3 |
| Title | Per-Tensor vs Per-Channel vs Per-Group Quantization |
| Existing source file if available | `ch08_quantization.pdf` |
| Exists or must be created | Exists in PDF; expand with safe wording |
| Exact section placement | Section 8.2.2 |
| Caption | **Table 8.3 — Quantization Granularity Comparison.** Smaller scale groups improve quality but increase metadata and kernel complexity. |
| Intro paragraph | Granularity is one of the highest-impact quantization decisions. |
| Explanation paragraph | Per-tensor is simple but outlier-sensitive; per-group is common for low-bit LLM quantization. |
| Key takeaway | INT4 generally needs fine granularity to be production viable. |
| Web-readiness | Keep rows concise. |
| Print-readiness | Add footnote for representative quality claims. |
| Required production fixes | Label quality rows `[REPRESENTATIVE]`. |

---

## Table 8.4 — GPTQ vs AWQ vs SmoothQuant

| Field | Value |
|---|---|
| Number | Table 8.4 |
| Title | Post-Training Quantization Method Comparison |
| Existing source file if available | `ch08_quantization.pdf`; papers for validation |
| Exists or must be created | Must be created from PDF narrative |
| Exact section placement | Section 8.3 |
| Caption | **Table 8.4 — GPTQ vs AWQ vs SmoothQuant.** Different PTQ methods target different quantization failure modes. |
| Intro paragraph | Once naive PTQ fails, the correct next method depends on the problem. |
| Explanation paragraph | GPTQ uses approximate second-order correction; AWQ uses activation-aware salient-channel scaling; SmoothQuant migrates activation outlier difficulty into weights for W8A8. |
| Key takeaway | The best method is workload-, hardware-, and quality-budget-specific. |
| Web-readiness | Use columns: method, target, calibration signal, output format, watchout. |
| Print-readiness | Keep table width manageable. |
| Required production fixes | Use paper-backed claims and avoid universal ranking. |

---

## Table 8.5 — Precision/Quantization Method Decision Comparison

| Field | Value |
|---|---|
| Number | Table 8.5 |
| Title | FP8, INT8, INT4, KV Quantization Decision Comparison |
| Existing source file if available | `ch08_quantization.pdf` |
| Exists or must be created | Must be created from PDF |
| Exact section placement | After Section 8.5 |
| Caption | **Table 8.5 — Precision/Quantization Decision Comparison.** Weight precision, activation precision, and KV-cache precision affect different resources. |
| Intro paragraph | Quantization choices should be compared by what they reduce: weights, activations, KV cache, bandwidth, or calibration cost. |
| Explanation paragraph | Distinguish W8A8, W4A16, W4A8, FP8 KV, and INT8 KV. |
| Key takeaway | Weight quantization and KV-cache quantization are complementary. |
| Web-readiness | Add confidence labels for speed/quality values. |
| Print-readiness | Avoid too many numeric columns. |
| Required production fixes | Mark tokens/sec and quality deltas as `[REPRESENTATIVE]` or `[ENV-SPECIFIC]`. |

---

## Table 8.6 — Production Deployment Decision Framework

| Field | Value |
|---|---|
| Number | Table 8.6 |
| Title | Production Deployment Decision Framework |
| Existing source file if available | `ch08_quantization.pdf` |
| Exists or must be created | Exists conceptually; convert to table |
| Exact section placement | Section 8.8 |
| Caption | **Table 8.6 — Production Deployment Decision Framework.** Choose quantization by model fit, hardware support, quality target, calibration budget, and deployment runtime. |
| Intro paragraph | Production teams need a repeatable way to choose precision, not a one-off opinion. |
| Explanation paragraph | The table should map constraints to recommended starting points and validation requirements. |
| Key takeaway | Start with the least risky precision that meets memory and cost goals. |
| Web-readiness | Use concise rows, not a giant decision matrix. |
| Print-readiness | Consider splitting into two tables if too wide. |
| Required production fixes | Validate hardware rows against current official docs. |

---

## Table 8.7 — Quantization Evaluation Checklist

| Field | Value |
|---|---|
| Number | Table 8.7 |
| Title | Quantization Evaluation Checklist |
| Existing source file if available | `ch08_quantization.pdf` |
| Exists or must be created | Must be created from Section 8.9.3 |
| Exact section placement | Section 8.9.3 |
| Caption | **Table 8.7 — Quantization Evaluation Checklist.** A quantized model should pass language-model, broad-capability, task-specific, latency, throughput, and rollback checks before production. |
| Intro paragraph | Quantization is not complete when calibration finishes; it is complete after validation. |
| Explanation paragraph | Include PPL, MMLU/HellaSwag/TruthfulQA, task-specific evals, P50/P95/P99 latency, throughput, memory headroom, and shadow traffic. |
| Key takeaway | Never ship quantization based on a single aggregate benchmark. |
| Web-readiness | Strong checklist format. |
| Print-readiness | Easy to reuse as a production checklist. |
| Required production fixes | Add environment-specific warning for all benchmark claims. |

---

## Production Diagram Asset List

```text
../assets/diagrams/svg/ch08_fig_8_1_precision_format_spectrum.svg
../assets/diagrams/svg/ch08_fig_8_2_quantization_formula_flow.svg
../assets/diagrams/svg/ch08_fig_8_3_outlier_quantization_error.svg
../assets/diagrams/svg/ch08_fig_8_4_granularity_comparison.svg
../assets/diagrams/svg/ch08_fig_8_5_ptq_method_map.svg
../assets/diagrams/svg/ch08_fig_8_6_fp8_te_pipeline.svg
../assets/diagrams/svg/ch08_fig_8_7_kv_cache_quantization.svg
../assets/diagrams/svg/ch08_fig_8_8_quantization_decision_tree.svg
```

---

## Figure Integration Warnings

1. Existing source diagrams should be linked in HTML as source references, but production figure placeholders should use Ch08-specific planned asset paths.
2. If the source-pack deadline is tight, placeholder cards are acceptable; do not block Ch08 on final SVG exports.
3. Do not embed PDF screenshots as final figures unless they are redrawn or exported cleanly.
4. Use SVG for flowcharts and PNG 300 DPI for detailed charts.
5. Make sure all figures can be understood in grayscale for print.
