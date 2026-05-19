---
title: "Chapter 6 Figure Integration Plan"
book: "AI/ML Infrastructure from Silicon to Scale"
chapter: "Ch06 — Training and Inference Workloads: From Batch Training to Real-Time Serving"
author: "Venkat Vinjam"
edition_note: "Current as of 2026 edition"
branch_target: "production-v1.0"
status: "Production Planning Pack"
output_path: "publishing/figure_plans/ch06_figure_integration_plan.md"
---

# Chapter 6 Figure Integration Plan

## 0. Figure Plan Intent

Chapter 6 is about workload behavior over time. The visual language should therefore emphasize:

- Timeline behavior: training step, prefill/decode, batching, chunked prefill.
- Resource regimes: compute-bound, memory-bound, communication-bound, I/O-bound, scheduler-bound.
- System flows: data pipeline, training pipeline, serving pipeline, disaggregated serving.
- Tail-latency behavior: throughput/latency knee and P99 cliffs.
- Principal-level tradeoffs: throughput vs latency, utilization vs SLO, HBM capacity vs concurrency, FLOPS vs cost/token.

Recommended production set:

- **9 figures** — 5 reused/adapted from existing assets, 4 must be created.
- **8 tables** — all must be created in chapter source for clarity and print/web consistency.

---

## 1. Figure and Table Inventory

### Figure inventory

| Number | Title | Existing source file if available | Exists or must be created | Exact section placement |
|---:|---|---|---|---|
| Fig 6.1 | Training vs Inference Workload Taxonomy Map | None | Must be created | §6.1 Workload Taxonomy |
| Fig 6.2 | Training Step Waterfall and Stream Overlap | `diagrams_batch3.html#d29` | Exists, adapt | §6.2 The Training Step |
| Fig 6.3 | Training vs Inference Memory Pressure | None | Must be created | §6.3 Training Memory and Serving Memory |
| Fig 6.4 | Prefill vs Decode Regime Split | `diagrams_batch1.html#d7` | Exists, adapt | §6.5 Inference Anatomy |
| Fig 6.5 | Static vs Continuous Batching Timeline | `diagrams_batch1.html#d9` | Exists, adapt | §6.7 Static vs Continuous Batching |
| Fig 6.6 | Chunked Prefill Scheduler Fairness | None or adapt from serving timeline | Must be created | §6.8 Chunked Prefill |
| Fig 6.7 | Throughput-Latency Knee and P99 Cliff | None | Must be created | §6.12 Throughput-Latency-Cost Tradeoff |
| Fig 6.8 | Online Model Serving Pipeline | None | Must be created | §6.9 Online Serving Pipeline |
| Fig 6.9 | Disaggregated Prefill/Decode Architecture | `diagrams_batch2.html#d20` | Exists, adapt | §6.10 Disaggregated Prefill/Decode |

### Table inventory

| Number | Title | Existing source file if available | Exists or must be created | Exact section placement |
|---:|---|---|---|---|
| Table 6.1 | Workload Operating Modes | None | Must be created | §6.1 Workload Taxonomy |
| Table 6.2 | Training Step Bottleneck Checklist | None | Must be created | §6.2/§6.4 Training Bottlenecks |
| Table 6.3 | Training vs Inference Memory Budget | None | Must be created | §6.3 Memory Pressure |
| Table 6.4 | Serving Metrics Glossary | None | Must be created | §6.6 Serving Metrics |
| Table 6.5 | Batching Strategy Comparison | None | Must be created | §6.7 Static vs Continuous Batching |
| Table 6.6 | Serving Framework Feature Matrix | None | Must be created | §6.9/§6.10 Serving Pipeline |
| Table 6.7 | What to Measure First | None | Must be created | §6.13 Diagnostic Playbooks |
| Table 6.8 | Principal-Level Workload Tradeoffs | None | Must be created | §6.14 Principal Interview Lens |

---

## 2. Detailed Figure Specifications

## Fig 6.1 — Training vs Inference Workload Taxonomy Map

**Number:** Fig 6.1  
**Title:** Training vs Inference Workload Taxonomy Map  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.1 Workload Taxonomy: Training, Offline Inference, Batch API, Interactive Serving

**Caption:**  
Training, offline inference, batch API serving, and interactive streaming serving are different operating modes. They use similar models and hardware, but optimize different objective functions and fail in different ways.

**Intro paragraph:**  
Before discussing formulas or frameworks, the reader needs to see that “AI workload” is not a single category. Batch training is a long-running throughput problem. Interactive serving is a tail-latency and cost/token problem. Offline inference can tolerate latency but not poor GPU-hour efficiency. Batch API serving sits in the middle.

**Explanation paragraph:**  
Use a 2D map. X-axis: latency sensitivity, low to high. Y-axis: request/batch dynamism, static to dynamic. Place batch training and offline inference on the static side, batch API serving in the middle, and interactive serving in the dynamic/high-latency-sensitivity zone. Add one metric under each zone: MFU, tokens/sec, throughput under SLA, P99 TTFT/TPOT.

**Key takeaway:**  
The first optimization is workload classification. The wrong metric leads to the wrong architecture.

**Web-readiness:**  
Create as inline SVG in a responsive figure card. Use short labels and tooltip-style notes only if necessary. Use dark background consistent with `index.html`.

**Print-readiness:**  
Ensure each quadrant label is readable in grayscale. Do not rely only on color; use icons or line patterns.

**Required production fixes:**  
Create new SVG. Add alt text. Add responsive wrapper. Confirm it prints within one page width.

---

## Fig 6.2 — Training Step Waterfall and Stream Overlap

**Number:** Fig 6.2  
**Title:** Training Step Waterfall and Stream Overlap  
**Existing source file if available:** `diagrams_batch3.html#d29` — Training Step Timeline — CUDA Stream Overlap  
**Exists or must be created:** Exists, adapt  
**Exact section placement:** §6.2 The Training Step: From Batch to Weight Update

**Caption:**  
A training step is a pipeline, not just forward and backward compute. Data loading, H2D transfer, forward, backward, gradient synchronization, optimizer update, and checkpoint I/O all compete for the step-time budget.

**Intro paragraph:**  
Training workloads operate on large batches and usually run for hours or days. The unit of performance is the training step. If the step is slow, the principal engineer must decompose it before blaming the GPU kernel.

**Explanation paragraph:**  
Adapt the existing stream-overlap diagram into a simpler waterfall. Show data prefetch and H2D transfer on one lane, compute on a second lane, NCCL communication on a third lane, and checkpoint/I/O as an occasional fourth lane. Show overlap with translucent bars. Label bottleneck classes: data-bound, compute-bound, communication-bound, optimizer/memory-bound, I/O-bound.

**Key takeaway:**  
High GPU utilization alone does not prove efficient training. Step-time breakdown and MFU are the first diagnostic tools.

**Web-readiness:**  
Keep as SVG. Add horizontal scroll wrapper for mobile. Use `figcaption` and anchor `#fig-6-2-training-step`.

**Print-readiness:**  
Reduce dense labels from the existing asset. Use minimum 8–9 pt equivalent font. Avoid small colored text on dark backgrounds.

**Required production fixes:**  
Remove old “Diagram 29” label. Replace any implementation-specific “cuDNN DataLoader” language with generic “DataLoader / CPU preprocessing / H2D.” Add confidence labels only in surrounding text, not inside the figure.

---

## Fig 6.3 — Training vs Inference Memory Pressure

**Number:** Fig 6.3  
**Title:** Training vs Inference Memory Pressure  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.3 Training Memory: Activations, Gradients, Optimizer State, and Checkpoints

**Caption:**  
Training memory is dominated by parameters, gradients, optimizer state, and activations. Inference serving memory is dominated by weights and KV cache. The memory pressure moves when the workload moves.

**Intro paragraph:**  
A common mistake is to reason about training and serving memory as if they were the same budget. They are not. Training must keep enough information to compute gradients and update weights. Serving must keep enough context to continue active user sessions.

**Explanation paragraph:**  
Create a side-by-side stacked bar. Left bar: training memory components — weights, gradients, optimizer states, activations, temporary buffers. Right bar: serving memory components — weights, KV cache, CUDA graphs/runtime buffers, request metadata, temporary activations. Use relative conceptual sizes, not exact hardware claims.

**Key takeaway:**  
Training usually fights activation and optimizer-state memory; serving fights KV-cache and concurrency memory.

**Web-readiness:**  
SVG card with legend. Use short labels and avoid exact percentages unless representative.

**Print-readiness:**  
Use patterns or labels inside bars so grayscale print remains understandable.

**Required production fixes:**  
Mark as `[REPRESENTATIVE]`. Do not imply exact memory percentages. Add note that actual budgets depend on precision, framework, parallelism, and sequence length.

---

## Fig 6.4 — Prefill vs Decode Regime Split

**Number:** Fig 6.4  
**Title:** Prefill vs Decode Regime Split  
**Existing source file if available:** `diagrams_batch1.html#d7` — Prefill vs Decode Phase — Bottleneck Comparison  
**Exists or must be created:** Exists, adapt  
**Exact section placement:** §6.5 Inference Anatomy: Prefill and Decode

**Caption:**  
LLM inference has two phases with different bottlenecks. Prefill processes the prompt in parallel and is often compute-heavy. Decode generates one token at a time and is often constrained by memory bandwidth, KV cache, and scheduler behavior.

**Intro paragraph:**  
Interactive LLM serving is not one uniform forward pass. The request first performs prefill over the prompt, then enters decode, where each iteration generates the next token. Optimizing one phase does not automatically optimize the other.

**Explanation paragraph:**  
Use a two-panel visual. Left panel: prompt tokens enter a single prefill pass with large GEMMs. Right panel: a loop generates one output token per active sequence per iteration. Add metric labels: TTFT is influenced by queue and prefill; TPOT is influenced by decode iteration time.

**Key takeaway:**  
Prefill wants large efficient compute; decode wants steady bandwidth-efficient iteration scheduling.

**Web-readiness:**  
Reuse existing SVG but update title and alt text. Keep formulas visible but not too dense.

**Print-readiness:**  
Check all token labels and formula labels. Convert tiny text to larger explanatory caption if needed.

**Required production fixes:**  
If the existing diagram says a 70B model reads exactly 140 GB every decode step, qualify it as BF16 dense-weight approximation. Label formula in text as `[ESTIMATED]` or `[DERIVED FROM SHIPPED]` depending on hardware bandwidth use.

---

## Fig 6.5 — Static vs Continuous Batching Timeline

**Number:** Fig 6.5  
**Title:** Static vs Continuous Batching Timeline  
**Existing source file if available:** `diagrams_batch1.html#d9` — Continuous Batching — Iteration-Level Scheduling  
**Exists or must be created:** Exists, adapt  
**Exact section placement:** §6.7 Static Batching vs Continuous Batching

**Caption:**  
Static batching waits for the longest request in the batch. Continuous batching admits new requests at decode-iteration boundaries, reducing slot waste for variable-length generation.

**Intro paragraph:**  
Training uses stable batches. Interactive serving receives requests with different prompt lengths, output lengths, arrival times, and priorities. Static batching wastes capacity when shorter requests finish early.

**Explanation paragraph:**  
Show two timelines with requests A, B, C, and D. In static batching, all requests begin together and the next batch waits for the longest. In continuous batching, completed slots are refilled during later decode iterations. Emphasize that implementation details differ by framework.

**Key takeaway:**  
Continuous batching is a scheduler optimization that turns variable-length request chaos into higher GPU occupancy without forcing all users to wait for the longest request.

**Web-readiness:**  
Reuse existing SVG with updated title. Put in responsive card. Add `aria-label` summary.

**Print-readiness:**  
Remove or soften any “10–30×” universal improvement text. Keep the diagram as behavior-focused rather than benchmark-focused.

**Required production fixes:**  
Replace fixed improvement claim with: “Throughput improvement is workload- and runtime-specific.” Add `[ENV-SPECIFIC]` in surrounding text.

---

## Fig 6.6 — Chunked Prefill Scheduler Fairness

**Number:** Fig 6.6  
**Title:** Chunked Prefill Scheduler Fairness  
**Existing source file if available:** None; can reuse visual language from `diagrams_batch1.html#d9`  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.8 Chunked Prefill and Scheduler Fairness

**Caption:**  
Chunked prefill breaks a long prompt into smaller work units so active decode streams are not blocked by one large prefill job.

**Intro paragraph:**  
Long-context prompts can create head-of-line blocking. A single 32K-token prompt may consume a long prefill window while active users wait for their next decode token.

**Explanation paragraph:**  
Show two timelines. Top: non-chunked long prefill blocks decode iterations for existing requests. Bottom: long prefill is split into chunks and interleaved with decode steps. Label the tradeoff: better tail fairness, more scheduler complexity, and possible prefill overhead.

**Key takeaway:**  
Chunked prefill is a latency-tail protection technique, not just a throughput feature.

**Web-readiness:**  
Create a compact SVG with only three requests and one long prompt. Include short labels: “decode protected,” “prefill chunk,” “tail risk.”

**Print-readiness:**  
Use clear timeline ticks and label each lane directly. Avoid tiny legends.

**Required production fixes:**  
Validate framework support wording against vLLM/TensorRT-LLM/SGLang docs. Use `[ENV-SPECIFIC]` for measured tail-latency gains.

---

## Fig 6.7 — Throughput-Latency Knee and P99 Cliff

**Number:** Fig 6.7  
**Title:** Throughput-Latency Knee and P99 Cliff  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.12 Throughput-Latency-Cost Tradeoff Framework

**Caption:**  
Interactive serving should not blindly maximize utilization. As arrival rate approaches service capacity, queueing delay rises nonlinearly and P99 latency can fail before average utilization looks alarming.

**Intro paragraph:**  
A training system often wants sustained high utilization. A serving system must leave enough headroom to absorb bursts, long prompts, cache misses, and uneven output lengths.

**Explanation paragraph:**  
Plot X-axis as offered load or GPU utilization and Y-axis as latency. Show a smooth P50 curve and a sharper P99 curve. Mark the “safe operating region,” “knee,” and “tail cliff.” Add a callout that cost/token improves with load until tail latency violates SLO.

**Key takeaway:**  
For real-time serving, the best operating point is usually before the utilization maximum.

**Web-readiness:**  
SVG plot with simple axes. Avoid requiring interactive chart libraries.

**Print-readiness:**  
Use dashed/solid line distinction for P50 and P99. Make axis labels large.

**Required production fixes:**  
Label all curve values `[REPRESENTATIVE]`. Do not present a universal utilization threshold.

---

## Fig 6.8 — Online Model Serving Pipeline

**Number:** Fig 6.8  
**Title:** Online Model Serving Pipeline  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.9 Online Serving Pipeline: Router, Queue, Scheduler, Runtime, GPU

**Caption:**  
A model server is a pipeline: request routing, admission control, queueing, scheduling, runtime execution, token streaming, and telemetry. GPU kernels are only one part of the serving path.

**Intro paragraph:**  
When a user says an LLM is slow, the delay may not be in the model kernel. It may be in the queue, tokenizer, scheduler, prefill, decode loop, KV allocation, network streaming, or overload policy.

**Explanation paragraph:**  
Show flow: Gateway/API → tokenizer → router → admission control → scheduler queue → prefill/decode runtime → GPU kernels/KV cache → streamer → metrics/tracing. Add side-channel telemetry arrows to Prometheus/OpenTelemetry/Grafana.

**Key takeaway:**  
Serving performance is a system pipeline problem. TTFT and TPOT must be decomposed before optimization.

**Web-readiness:**  
Create as wide but responsive SVG. Use table-like blocks with consistent styling.

**Print-readiness:**  
Must fit landscape or one full-page width. Keep labels short.

**Required production fixes:**  
Make “telemetry” visual but do not deep-dive; forward-reference Ch17.

---

## Fig 6.9 — Disaggregated Prefill/Decode Architecture

**Number:** Fig 6.9  
**Title:** Disaggregated Prefill/Decode Architecture  
**Existing source file if available:** `diagrams_batch2.html#d20` — Disaggregated Prefill-Decode Architecture  
**Exists or must be created:** Exists, adapt  
**Exact section placement:** §6.10 Disaggregated Prefill/Decode

**Caption:**  
Disaggregated serving separates prefill and decode into different GPU pools so each phase can be scheduled and scaled according to its resource profile.

**Intro paragraph:**  
Prefill and decode want different things from the system. Prefill benefits from large compute-heavy batches. Decode needs steady low-jitter iteration scheduling and memory-bandwidth efficiency.

**Explanation paragraph:**  
Show router → prefill pool → KV transfer → decode pool → streaming response. Include labels: prefill pool compute-heavy, decode pool bandwidth/KV-heavy, transfer path latency-sensitive. Mention that exact transport and maturity depend on runtime.

**Key takeaway:**  
Disaggregation is a workload-specialization strategy: it optimizes separate phases instead of forcing them to share one scheduling policy.

**Web-readiness:**  
Reuse existing SVG but remove framework-specific cost reduction claims unless validated. Link to Ch11 for KV transfer details.

**Print-readiness:**  
Check labels and arrows. If existing diagram includes small percentage claims, move them to caption or remove.

**Required production fixes:**  
Remove any universal “30–50% cost reduction” wording unless tied to a specific cited benchmark. Label benefits `[ENV-SPECIFIC]`.

---

## 3. Detailed Table Specifications

## Table 6.1 — Workload Operating Modes

**Number:** Table 6.1  
**Title:** Workload Operating Modes  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.1 Workload Taxonomy

**Caption:**  
The four major AI workload modes optimize different metrics and fail for different reasons.

**Intro paragraph:**  
Use this table as the chapter’s opening reference. It should appear immediately after Fig 6.1.

**Explanation paragraph:**  
Columns: workload mode, unit of work, objective, batch behavior, dominant bottleneck, primary metric, failure mode, confidence label.

**Key takeaway:**  
Do not discuss performance until the workload mode is named.

**Web-readiness:**  
Use responsive table wrapper. Keep cells concise.

**Print-readiness:**  
May need reduced font size. Avoid long sentences.

**Required production fixes:**  
Create from scratch. No exact hardware numbers.

---

## Table 6.2 — Training Step Bottleneck Checklist

**Number:** Table 6.2  
**Title:** Training Step Bottleneck Checklist  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.2 or §6.4 Training Bottlenecks

**Caption:**  
A training slowdown should be debugged by step-stage attribution before kernel-level profiling.

**Intro paragraph:**  
This table makes the training section practical. It tells the reader what to measure first when step time is poor.

**Explanation paragraph:**  
Rows: data loading, H2D transfer, forward compute, backward compute, activation recomputation, gradient sync, optimizer step, checkpointing, straggler/rank skew. Columns: symptom, metric, likely tool, likely fix.

**Key takeaway:**  
Training performance is a step-time attribution problem.

**Web-readiness:**  
Responsive table. Consider converting to cards on mobile.

**Print-readiness:**  
Keep each cell to one line where possible.

**Required production fixes:**  
Do not include tool commands that belong in Ch17 unless brief.

---

## Table 6.3 — Training vs Inference Memory Budget

**Number:** Table 6.3  
**Title:** Training vs Inference Memory Budget  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.3 Training Memory and Serving Memory

**Caption:**  
Training and inference stress different memory components even when they use the same model weights.

**Intro paragraph:**  
Memory pressure is the easiest way to confuse training and serving. This table should sit next to Fig 6.3.

**Explanation paragraph:**  
Columns: component, training role, inference role, scaling driver, common mitigation, confidence label. Rows: weights, activations, gradients, optimizer states, KV cache, temporary buffers, checkpoint storage.

**Key takeaway:**  
Optimizer state is a training problem; KV cache is a serving problem.

**Web-readiness:**  
Use line breaks within cells sparingly. Add glossary links for KV cache and optimizer state.

**Print-readiness:**  
This is likely wide. Use abbreviations only if defined.

**Required production fixes:**  
Use `[REPRESENTATIVE]` for conceptual scaling and `[ENV-SPECIFIC]` for actual memory values.

---

## Table 6.4 — Serving Metrics Glossary

**Number:** Table 6.4  
**Title:** Serving Metrics Glossary  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.6 Serving Metrics

**Caption:**  
Serving metrics must separate first-token latency, per-token latency, throughput, and tail behavior.

**Intro paragraph:**  
Readers often use “latency” too loosely. This glossary should define the terms used for the rest of the chapter.

**Explanation paragraph:**  
Rows: TTFT, TPOT/TBT, E2E latency, output tokens/sec, requests/sec, queue time, prefill time, decode time, P50/P95/P99, cost per 1M tokens. Columns: definition, why it matters, what worsens it, confidence label.

**Key takeaway:**  
Averages hide serving failures. P95/P99 reveal scheduler and overload behavior.

**Web-readiness:**  
Use anchor links for TTFT, TPOT, P99.

**Print-readiness:**  
Definitions should fit without footnotes.

**Required production fixes:**  
Use exact percentile wording: P99 means 99% of requests are at or below that latency and 1% are worse.

---

## Table 6.5 — Batching Strategy Comparison

**Number:** Table 6.5  
**Title:** Batching Strategy Comparison  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.7 Static Batching vs Continuous Batching

**Caption:**  
Batching strategy determines whether the GPU is filled efficiently and whether users wait behind unrelated requests.

**Intro paragraph:**  
Batching is where training and serving diverge sharply. Training batches are planned. Serving batches are assembled from live arrivals.

**Explanation paragraph:**  
Rows: static batching, dynamic batching, continuous batching, chunked prefill, disaggregated prefill/decode. Columns: how it works, best for, tradeoff, risk, confidence label.

**Key takeaway:**  
Serving batching is scheduler design, not just tensor shape selection.

**Web-readiness:**  
Responsive table with short cells.

**Print-readiness:**  
Avoid excessive framework names in the main table.

**Required production fixes:**  
No fixed throughput multipliers unless labeled `[ENV-SPECIFIC]` and sourced.

---

## Table 6.6 — Serving Framework Feature Matrix

**Number:** Table 6.6  
**Title:** Serving Framework Feature Matrix  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.9 Online Serving Pipeline or §6.10 Disaggregated Prefill/Decode

**Caption:**  
Modern serving runtimes provide different combinations of batching, KV cache, quantization, and disaggregation features. Validate exact feature support against the runtime version used.

**Intro paragraph:**  
This table should be factual and cautious. It should not rank frameworks broadly. It should show why the chapter concepts map to real systems.

**Explanation paragraph:**  
Rows: vLLM, TensorRT-LLM, SGLang, optional TGI. Columns: continuous/in-flight batching, paged KV, chunked prefill, prefix caching, speculative decoding, disaggregated P/D, notes, source type.

**Key takeaway:**  
The concepts are stable; feature maturity is version-specific.

**Web-readiness:**  
Include short source links in final “References” section, not inside every cell.

**Print-readiness:**  
Use “Yes / Version-specific / Experimental / Validate” rather than long prose.

**Required production fixes:**  
Validate against official docs at source-pack time. Mark `[SHIPPED]` only for documented stable features; mark experimental features carefully.

---

## Table 6.7 — What to Measure First

**Number:** Table 6.7  
**Title:** What to Measure First  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.13 Diagnostic Playbooks

**Caption:**  
Principal performance work starts with the first discriminating measurement.

**Intro paragraph:**  
This table turns the chapter into a practical diagnostic guide.

**Explanation paragraph:**  
Rows: low MFU, GPU idle, slow data loader, high AllReduce time, slow TTFT, slow TPOT, P99 spike, KV OOM/preemption, checkpoint slowdown. Columns: first measurement, likely cause, next tool, likely fix.

**Key takeaway:**  
Measure at the workload boundary before descending into kernel counters.

**Web-readiness:**  
Can be a compact table or accordion cards.

**Print-readiness:**  
Use concise entries.

**Required production fixes:**  
Forward-reference Ch17 for tools rather than over-explaining them here.

---

## Table 6.8 — Principal-Level Workload Tradeoffs

**Number:** Table 6.8  
**Title:** Principal-Level Workload Tradeoffs  
**Existing source file if available:** None  
**Exists or must be created:** Must be created  
**Exact section placement:** §6.14 Principal Interview Lens

**Caption:**  
Principal-level workload discussions require explicit tradeoffs, not isolated optimizations.

**Intro paragraph:**  
This table prepares the reader for interview and architecture-review discussions.

**Explanation paragraph:**  
Rows: batch size, sequence length, quantization, tensor parallelism, chunked prefill, disaggregation, checkpoint frequency, activation checkpointing, data pipeline investment. Columns: helps, hurts, training implication, serving implication, interview wording.

**Key takeaway:**  
A principal answer names the tradeoff and the metric that decides it.

**Web-readiness:**  
Place near end as a summary table.

**Print-readiness:**  
Wide table; use landscape-friendly styling or split into two tables.

**Required production fixes:**  
Keep interview wording short and memorable.

---

## 4. Figure/Table Numbering Guidance

Use stable numbering even if optional figures are later removed:

```text
Fig 6.1 Workload Taxonomy Map
Fig 6.2 Training Step Waterfall
Fig 6.3 Training vs Inference Memory Pressure
Fig 6.4 Prefill vs Decode
Fig 6.5 Static vs Continuous Batching
Fig 6.6 Chunked Prefill
Fig 6.7 Throughput-Latency Knee
Fig 6.8 Online Serving Pipeline
Fig 6.9 Disaggregated Prefill/Decode

Table 6.1 Workload Operating Modes
Table 6.2 Training Step Bottleneck Checklist
Table 6.3 Training vs Inference Memory Budget
Table 6.4 Serving Metrics Glossary
Table 6.5 Batching Strategy Comparison
Table 6.6 Serving Framework Feature Matrix
Table 6.7 What to Measure First
Table 6.8 Principal-Level Workload Tradeoffs
```

---

## 5. Reused Diagram Production Fix Checklist

For every reused diagram:

- [ ] Remove old chapter number from visible diagram title/subtitle.
- [ ] Replace old “Making AI Go Fast” label if it conflicts with current book branding.
- [ ] Ensure figure title matches Ch06 numbering.
- [ ] Confirm no unsupported benchmark multiplier remains.
- [ ] Add `role="img"` and `aria-label` or equivalent alt text in HTML.
- [ ] Test mobile overflow.
- [ ] Test print-to-PDF legibility.
- [ ] Confirm color contrast on dark background.
- [ ] Ensure surrounding caption carries confidence labels for claims.

---

## 6. Web and Print Style Requirements

### Web

- Use same dark visual language as `index.html`.
- Use IBM Plex Sans and IBM Plex Mono.
- Wrap every wide figure and table:

```html
<div class="figure-wrap">...</div>
<div class="table-wrap">...</div>
```

- Add sidebar TOC anchors for each figure-heavy section.
- Use lazy loading only for external images; inline SVGs do not need it.
- Use captions under each figure, not title text only inside SVG.

### Print

- Add CSS rules:

```css
@media print {
  .sidebar, .mobile-nav { display: none; }
  .chapter { max-width: 100%; }
  .figure-wrap, .table-wrap { break-inside: avoid; }
  table { font-size: 8.5pt; }
  pre, .formula { white-space: pre-wrap; }
}
```

- Avoid color-only semantics.
- Keep figures below one print page when possible.
- Split Table 6.8 if it becomes too wide.

---

## 7. Final Figure Plan Decision

**Decision:** Use 9 figures and 8 tables for the production source pack.

**Must-create first:** Fig 6.1, Fig 6.3, Fig 6.6, Fig 6.7, Fig 6.8.  
**Reuse/adapt first:** Fig 6.2, Fig 6.4, Fig 6.5, Fig 6.9.

This gives Ch06 a strong visual identity and avoids overwhelming the reader with deep downstream diagrams that belong in Ch10, Ch11, Ch14, and Ch17.

