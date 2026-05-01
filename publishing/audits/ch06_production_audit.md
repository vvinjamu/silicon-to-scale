---
title: "Chapter 6 Production Audit"
book: "AI/ML Infrastructure from Silicon to Scale"
chapter: "Ch06 — Training and Inference Workloads: From Batch Training to Real-Time Serving"
author: "Venkat Vinjam"
edition_note: "Current as of 2026 edition"
branch_target: "production-v1.0"
status: "Production Planning Pack"
output_path: "publishing/audits/ch06_production_audit.md"
---

# Chapter 6 Production Audit

## 0. Audit Purpose

Chapter 6 should become the reader's first full workload-level chapter. Chapters 1–5 build the hardware, memory, power, and TCO foundation. Chapter 6 must translate that foundation into workload behavior: **batch training, offline/batch inference, online inference, and real-time streaming serving**.

The planned chapter title is:

> **Training and Inference Workloads: From Batch Training to Real-Time Serving**

That title is stronger than the earlier narrower label, **Inference Systems Architecture**, because it creates the missing bridge between foundation chapters and the later specialized chapters on kernels, quantization, compiler optimization, distributed training, KV cache, benchmarking, networking, storage, cluster management, and observability.

The chapter should answer one principal-level question:

> Given a workload, what is the dominant objective, bottleneck, memory pressure, communication pattern, scheduler behavior, and metric that determines whether the system is good?

---

## 1. Inputs Reviewed

### Live GitHub Pages structure

- GitHub Pages URL: `https://vvinjamu.github.io/silicon-to-scale/`
- Branch target: `production-v1.0`
- Observed public positioning: First Edition 2026, Production v1.0, 18 chapters, 6 parts, 40 diagrams, principal/staff engineering audience.
- Observed fast-track paths:
  - Inference fast-track includes Ch01, Ch06, Ch08, Ch09, Ch11, Ch13, Ch17.
  - Training specialist path includes Ch01, Ch04, Ch10, Ch12, Ch14, Ch15, Ch17.
  - Interview prep includes Ch01, Ch02, Ch06, Ch10, Ch11, Ch14, Ch18.

### Uploaded / local book assets

- `/mnt/data/index.html` — current GitHub Pages landing page style reference.
- `/mnt/data/ch00_front_matter-combined.pdf` — front matter, reading paths, table of contents, confidence labels.
- `/mnt/data/ch11_kv_cache_complete-combined.pdf` — downstream KV cache chapter, used to avoid overloading Ch06.
- `/mnt/data/diagrams_batch1.html` — includes existing diagrams for roofline, Ring AllReduce, Prefill vs Decode, PagedAttention, Continuous Batching, ZeRO Stage 3.
- `/mnt/data/diagrams_batch2.html` — includes existing diagram for Disaggregated Prefill-Decode Architecture.
- `/mnt/data/diagrams_batch3.html` — includes existing diagram for Training Step Timeline / CUDA Stream Overlap and profiling tree.
- `/mnt/data/diagram_05_parallelism_topology.html` — existing 4D parallelism physical mapping reference.
- `/mnt/data/diagram_08_observability_stack.html` — existing observability stack with training/inference metrics.

### Important source-pack constraint

The final source-pack chapter should be produced later as:

```text
source/chapters/ch06_training_inference_workloads.md
chapters/ch06_training_inference_workloads.html
```

The source-pack HTML should match the visual style of the current GitHub Pages and previous chapter pages: dark theme, IBM Plex fonts, left sidebar TOC, previous/next nav, responsive tables, confidence labels, callouts, formula blocks, print CSS, and mobile behavior.

---

## 2. Production Intent for Chapter 6

### Reader-facing promise

By the end of Ch06, the reader should be able to classify an AI workload into one of four operating modes and immediately know what to measure first.

| Workload mode | Primary objective | Dominant resource | Failure mode | Best first metric |
|---|---|---|---|---|
| Batch training | Maximize useful tokens/sec/GPU and MFU over long runs | Compute, activation memory, optimizer state, communication, data pipeline | Low MFU, stragglers, checkpoint loss, data stalls | Step time breakdown and MFU |
| Offline inference | Maximize tokens/sec or samples/sec under loose latency | Compute/HBM depending on model and batch | Poor batching, low utilization, expensive throughput | Tokens/sec/GPU |
| Batch API serving | Balance throughput and bounded latency | Scheduler, queueing, HBM, KV cache | p95/p99 latency drift under load | Throughput under SLA |
| Interactive streaming serving | Minimize TTFT/TPOT tail latency at acceptable cost/token | KV cache, HBM bandwidth, scheduler fairness | P99 TTFT/TPOT cliff, queue buildup, OOM/preemption | P99 TTFT, P99 TPOT, KV utilization |

### Chapter boundary

Ch06 should not become the full chapter on distributed training, KV cache, networking, or benchmarking. It should introduce these ideas only as workload consequences and point forward:

- Detailed distributed training topology belongs in Ch10.
- Detailed KV cache math belongs in Ch11.
- Benchmark methodology belongs in Ch12.
- Speculative decoding and MoE infrastructure belong in Ch13.
- NCCL/IB/RDMA deep dive belongs in Ch14.
- Storage and checkpointing deep dive belongs in Ch15.
- Cluster management and multi-tenant scheduling belongs in Ch16A.
- Observability and profiling stack belongs in Ch17.

---

## 3. What Is Strong

### 3.1 Strong chapter placement

Ch06 is positioned at a natural transition point. After the reader learns silicon, memory hierarchy, HBM, power, cooling, and TCO, they need a workload chapter that explains why the same hardware behaves differently under training, offline inference, and interactive serving.

### 3.2 Strong principal-level framing

The chapter can help the reader move beyond local optimization. A principal engineer should not say only, “This kernel is slow.” They should say:

> “This workload is decode-dominated under bursty interactive traffic. P99 TPOT is controlled by scheduler fairness, KV cache occupancy, and effective HBM bandwidth, not just GEMM TFLOPS.”

That is exactly the kind of system-level language the book promises.

### 3.3 Strong existing diagram inventory

Several reusable diagrams already exist:

| Existing asset | Useful for Ch06 | Recommended action |
|---|---|---|
| `diagrams_batch1.html#d7` — Prefill vs Decode Phase | Explains inference split | Reuse/adapt as Fig 6.3 |
| `diagrams_batch1.html#d9` — Continuous Batching | Explains static vs continuous batching | Reuse/adapt as Fig 6.5 |
| `diagrams_batch2.html#d20` — Disaggregated Prefill-Decode | Explains prefill/decode pools | Reuse/adapt as Fig 6.8 |
| `diagrams_batch3.html#d29` — Training Step Timeline | Explains training pipeline and stream overlap | Reuse/adapt as Fig 6.2 |
| `diagram_08_observability_stack.html` | Explains metrics layers | Reference later in §6.13 or defer to Ch17 |
| `diagram_05_parallelism_topology.html` | Explains distributed training communication | Use small excerpt or defer to Ch10 |
| `diagrams_batch1.html#d6` — Ring AllReduce | Explains gradient synchronization | Use reference callout, not a full Ch06 deep dive |
| `diagrams_batch1.html#d11` — ZeRO Stage 3 | Explains optimizer/memory sharding | Mention as forward pointer to Ch10 |

### 3.4 Strong terminology set

The chapter naturally introduces terms readers must know for interviews:

- Batch size, microbatch, global batch, sequence length, tokens/step.
- Step time, tokens/sec/GPU, MFU, HFU.
- Forward, backward, gradient sync, optimizer step, checkpointing.
- Prefill, decode, TTFT, TPOT/TBT, E2E latency, P50/P95/P99.
- Static batching, dynamic batching, continuous batching, chunked prefill.
- Online serving, offline inference, queueing, admission control, tail latency.
- Activation memory, optimizer state, KV cache, data loader wait.

### 3.5 Strong bridge to interviews

Ch06 can become a prime interview chapter because it covers the two most common system design prompts:

1. “Design a distributed training system for a 70B model.”
2. “Design a real-time LLM serving system for millions of users.”

It should include a principal interview section that compares how the same model behaves in training versus serving.

---

## 4. What Is Weak or Confusing

### 4.1 Current scope mismatch: Ch06 naming drift

Some uploaded/front-matter material identifies Ch06 as **Inference Systems Architecture**, while the new requested chapter title is **Training and Inference Workloads**. This is not a problem, but it must be resolved before source-pack creation.

**Risk:** If the chapter keeps the old title but adds training material, it may feel unfocused. If it uses the new title but keeps only inference content, the title overpromises.

**Fix:** Use the new title and explicitly make Ch06 the workload taxonomy chapter. Keep deep inference serving architecture, KV cache, and distributed training as downstream chapters.

### 4.2 Risk of duplicating Ch10 and Ch11

Training material can easily become a mini-Ch10. KV cache material can easily become a mini-Ch11.

**Fix:** Ch06 should introduce the workload-level role of activation memory, optimizer state, gradient sync, checkpointing, and KV cache, but leave full formulas and production deep dives to later chapters.

### 4.3 Prefill vs decode formulas need safe wording

The commonly used decode approximation:

```text
TPOT_lower_bound ≈ model_bytes_touched_per_token / effective_HBM_bandwidth
```

is useful but easy to overstate. It is a lower-bound model, not a guaranteed latency prediction.

**Fix:** Label it `[DERIVED FROM SHIPPED]` when using shipped bandwidth and `[ENV-SPECIFIC]` for measured TPOT. State that the formula ignores scheduler overhead, KV cache attention cost, non-ideal bandwidth, communication, quantization format, and runtime overhead.

### 4.4 Continuous batching improvement ranges need caution

Existing diagrams may mention large throughput multipliers. Those can be true in specific synthetic or high-variance workloads, but universal claims like “10–30×” are risky in a production book.

**Fix:** Use safer wording:

> “Continuous batching can materially improve throughput versus static batching, especially for variable-length generation workloads. Exact gains are workload-, model-, and runtime-specific.” `[ENV-SPECIFIC]`

### 4.5 Training data pipeline deserves more attention

Many performance chapters overfocus on GPU compute. Ch06 should explicitly show that data loading, tokenization, sequence packing, CPU transforms, H2D transfer, and checkpoint I/O can create GPU idle time.

**Fix:** Add §6.3 “Training input pipeline and data stalls” or make it part of §6.2 training step waterfall.

### 4.6 Tail latency needs clearer reader treatment

Readers often understand average latency but not P95/P99. Ch06 should make the tail-latency problem tangible.

**Fix:** Add a mini example:

```text
P50 TTFT = 180 ms
P95 TTFT = 650 ms
P99 TTFT = 2.4 s
Average TTFT = 240 ms
```

Then explain why the average hides admission-control and scheduler failure.

---

## 5. Missing Diagrams and Tables

### 5.1 Missing or must-create diagrams

| Needed visual | Existing asset? | Status | Why it matters |
|---|---:|---|---|
| Fig 6.1 Workload Taxonomy Map | No | Must create | Gives the chapter its organizing model. |
| Fig 6.2 Training Step Waterfall | Partial: `diagrams_batch3.html#d29` | Adapt | Shows data/compute/comm/I/O breakdown. |
| Fig 6.3 Training vs Inference Memory Footprint | No | Must create | Explains activations/optimizer state vs KV cache. |
| Fig 6.4 Prefill vs Decode Regime Split | Yes: `diagrams_batch1.html#d7` | Reuse/adapt | Core inference mental model. |
| Fig 6.5 Static vs Continuous Batching Timeline | Yes: `diagrams_batch1.html#d9` | Reuse/adapt | Core scheduler mental model. |
| Fig 6.6 Chunked Prefill Fairness Timeline | Partial in serving diagrams | Must create/adapt | Explains long-prompt interference and tail control. |
| Fig 6.7 Throughput-Latency Knee Curve | No | Must create | Shows why max utilization can hurt interactive serving. |
| Fig 6.8 Model Serving Pipeline | No | Must create | Shows router → scheduler → runtime → kernels → telemetry. |
| Fig 6.9 Disaggregated Prefill/Decode | Yes: `diagrams_batch2.html#d20` | Reuse/adapt | Shows future-facing serving architecture. |

### 5.2 Missing or must-create tables

| Needed table | Status | Why it matters |
|---|---|---|
| Table 6.1 Workload Operating Modes | Must create | Core chapter taxonomy. |
| Table 6.2 Training Step Bottleneck Checklist | Must create | Helps reader diagnose training slowdowns. |
| Table 6.3 Training vs Inference Memory Budget | Must create | Clarifies activations/optimizer/KV differences. |
| Table 6.4 Inference Metrics Glossary | Must create | TTFT, TPOT, P50/P95/P99 definitions. |
| Table 6.5 Batching Strategy Comparison | Must create | Static, dynamic, continuous, chunked prefill. |
| Table 6.6 Serving Framework Feature Matrix | Must create | vLLM, TensorRT-LLM, SGLang; keep source-labeled. |
| Table 6.7 Principal Design Tradeoffs | Must create | Interview-ready tradeoff framing. |
| Table 6.8 Chapter 6 Validation Checklist | Must create | Supports confidence-label discipline. |

---

## 6. Existing Diagram Placement

| Asset | Proposed number | Exact section placement | Placement note |
|---|---:|---|---|
| `diagrams_batch3.html#d29` Training Step Timeline | Fig 6.2 | §6.2 “The Training Step: From Batch to Weight Update” | Use as training waterfall/stream-overlap visual. Simplify labels for print. |
| `diagrams_batch1.html#d7` Prefill vs Decode | Fig 6.4 | §6.5 “Inference Anatomy: Prefill and Decode” | Keep formula callouts but soften exact latency claims. |
| `diagrams_batch1.html#d9` Continuous Batching | Fig 6.5 | §6.7 “Static Batching vs Continuous Batching” | Replace any universal throughput multiplier with `[ENV-SPECIFIC]` wording. |
| `diagrams_batch2.html#d20` Disaggregated Prefill-Decode | Fig 6.9 | §6.10 “Disaggregated Prefill/Decode” | Keep architecture; remove specific cost-reduction percentage unless externally validated. |
| `diagram_08_observability_stack.html` | Reference figure, optional Fig 6.10 | §6.13 “What to Measure First” or forward pointer to Ch17 | Use a reduced excerpt if included; full deep dive belongs in Ch17. |
| `diagram_05_parallelism_topology.html` | Not primary Ch06 figure | §6.4 or forward pointer to Ch10 | Mention as “full topology in Ch10,” not a main Ch06 figure. |
| `diagrams_batch1.html#d6` Ring AllReduce | Optional inset | §6.4 “Distributed training communication” | Do not duplicate Ch14; use only for gradient-sync intuition. |
| `diagrams_batch1.html#d11` ZeRO Stage 3 | Optional forward pointer | §6.3 “Training memory pressure” | Detailed ZeRO belongs in Ch10. |

---

## 7. Technical Claims Needing Validation

### 7.1 P0 validation required before source-pack publication

| Claim area | Risk | Required action |
|---|---|---|
| `training_FLOPs ≈ 6 × params × tokens` | Could be presented as exact | Label `[ESTIMATED]`; explain dense Transformer assumption and accounting convention. |
| `inference_FLOPs ≈ 2 × params × generated_tokens` | MoE, attention overhead, batch shape can alter | Label `[ESTIMATED]`; call it a first-order dense forward approximation. |
| Prefill is compute-bound | Not always true for short prompts or poor batching | State “often/typically for long prompts and adequate batch”; label `[DERIVED FROM SHIPPED]` or `[ENV-SPECIFIC]`. |
| Decode is memory-bandwidth-bound | True common case, but crossover depends on batch/runtime | State “often at small batch”; label `[DERIVED FROM SHIPPED]`. |
| `TPOT ≈ model_bytes / HBM_BW` | Lower-bound formula, not measured latency | Add “ideal lower bound”; include effective bandwidth caveat. |
| Continuous batching improvement range | Exact multipliers vary widely | Avoid fixed universal claim; use `[ENV-SPECIFIC]`. |
| Chunked prefill improves P99 | Benefit depends on traffic mix | Use “can reduce long-prefill interference”; label `[ENV-SPECIFIC]`. |
| Disaggregated prefill/decode | Feature maturity varies by framework/version | Validate against official docs and label by exact runtime. |
| Framework feature matrix | Rapidly evolving | Cite official docs and include “current as of 2026 edition.” |
| Utilization guardrails such as 65–75% | Environment-specific | Label `[REPRESENTATIVE]` or `[ENV-SPECIFIC]`; do not make universal. |
| 82% KV cache alert | Existing internal guardrail, not universal standard | Mark `[ENV-SPECIFIC]`; explain tuning by workload. |

### 7.2 P1 validation required for polish

| Claim area | Risk | Required action |
|---|---|---|
| P50/P95/P99 definitions | Low | Add exact percentile definitions. |
| Optimizer-state memory | Medium | Clarify AdamW memory accounting: parameters, gradients, optimizer moments, master weights depending on precision and implementation. |
| Activation memory | Medium | Avoid exact formula unless architecture-specific; provide conceptual scaling with layers, batch, sequence, hidden size. |
| Checkpointing overhead | Medium | Mark all overhead ranges `[ENV-SPECIFIC]`. |
| Data loader bottlenecks | Low | Ground in PyTorch DataLoader / pinned memory / worker discussion; avoid universal worker counts. |
| AllReduce communication | Medium | Reserve full formula for Ch14, but include gradient sync as a stage. |

### 7.3 P2 validation for later source-pack refinement

| Claim area | Risk | Required action |
|---|---|---|
| Example numbers for 70B on H100/H200 | Medium | Use only as representative unless tied to public benchmark. |
| Serving-framework comparison | Medium | Keep table capability-focused, not ranking-focused. |
| Multi-modal serving notes | Low | Include only if chapter scope permits. |
| Cost/token formulas | Medium | Use as framework; actual cost is `[ENV-SPECIFIC]`. |

---

## 8. Reader-Experience Improvements

### 8.1 Start with a concrete scenario

Open with a short story:

> “A 70B model trains efficiently at high batch with 70% MFU, but the same model feels slow in chat because decode is memory-bound, P99 TTFT is queue-dominated, and KV cache admission control is failing.”

This immediately explains why training and serving are different even when the model and GPU are the same.

### 8.2 Use a repeated diagnostic template

For each workload mode, use the same diagnostic pattern:

```text
1. What is the objective?
2. What is the unit of work?
3. What is the bottleneck?
4. What metric proves it?
5. What optimization lever moves it?
6. What can go wrong at scale?
```

### 8.3 Keep formulas shallow but memorable

Ch06 should include only the formulas needed to reason quickly:

```text
Tokens per step = global_batch × sequence_length
Tokens/sec/GPU = tokens_per_step / step_time / GPU_count
Training FLOPs ≈ 6 × parameters × training_tokens
TTFT = queue_time + prefill_time + first_decode_time
TPOT_lower_bound ≈ model_bytes_touched_per_token / effective_HBM_bandwidth
Cost/token = total_serving_cost / generated_tokens
```

### 8.4 Separate teaching numbers from hardware claims

Use a visible callout:

> “The values in this chapter are teaching values unless explicitly labeled `[SHIPPED]`. Serving performance is heavily dependent on runtime version, prompt distribution, output length, parallelism, quantization, GPU memory utilization, and scheduler policy.”

### 8.5 Add “What to measure first” boxes

End major sections with a compact diagnostic box:

- Training slow? Measure step-time breakdown before kernel counters.
- GPU idle? Check DataLoader wait, H2D, CPU tokenization, checkpoint, and comm overlap.
- TTFT slow? Split queue vs prefill vs first decode.
- TPOT slow? Check batch size, HBM bandwidth, KV utilization, preemption, and scheduler fairness.
- P99 bad but average good? Check arrival bursts, long prompts, queueing, and cache eviction.

---

## 9. Principal-Level Interview Improvements

### 9.1 Add a dedicated “Principal Interview Lens” section

Recommended placement: near the end, before key takeaways.

Title:

> **6.14 Principal Interview Lens: Discussing Workloads Like an Architect**

The section should include three mini-whiteboard prompts.

#### Prompt 1 — Training system diagnosis

> “Your 1,024-GPU training job has 38% MFU. GPU utilization looks high, but step time is worse than expected. What do you check?”

Expected answer structure:

1. Establish baseline: model size, batch size, sequence length, step time, target MFU.
2. Break step time into forward, backward, AllReduce, optimizer, data loading, checkpoint.
3. Look for stragglers and rank skew.
4. Check communication overlap and NCCL algorithm selection.
5. Check data pipeline and sequence packing.
6. Recommend one experiment at a time.

#### Prompt 2 — Serving tail latency

> “P50 TTFT is 180 ms, but P99 TTFT is 2.4 seconds during traffic bursts. Average GPU utilization is only 62%. What is happening?”

Expected answer structure:

1. Average utilization can hide queueing and head-of-line blocking.
2. Split TTFT into queue, prefill, first decode.
3. Look for long-prompt prefill monopolization.
4. Check scheduler policy, chunked prefill, continuous batching, and admission control.
5. Check KV cache occupancy and evictions.
6. Use request-trace replay, not synthetic average prompts.

#### Prompt 3 — Training vs inference hardware choice

> “You can buy GPUs with higher FLOPS but lower HBM capacity, or GPUs with more HBM but lower peak FLOPS. Which do you choose?”

Expected answer structure:

1. For training: evaluate MFU, activation memory, optimizer state, parallelism, interconnect.
2. For serving: evaluate KV cache capacity, decode bandwidth, context length, concurrency, cost/token.
3. State that no single answer exists without workload shape.
4. Derive a simple memory and throughput model.
5. Recommend a benchmark plan and decision threshold.

### 9.2 Add interview-ready one-liners

Include a small table:

| Concept | Principal-level one-liner |
|---|---|
| Training | “Training is steady-state throughput optimization under memory, communication, and failure-recovery constraints.” |
| Interactive serving | “Serving is tail-latency optimization under bursty arrivals, KV pressure, and cost/token constraints.” |
| Prefill | “Prefill looks like batched Transformer inference over prompt tokens.” |
| Decode | “Decode is the regime where HBM bandwidth and scheduler policy often dominate.” |
| Continuous batching | “Continuous batching trades scheduler complexity for higher utilization under variable-length requests.” |
| Chunked prefill | “Chunked prefill protects active decoders from long-prompt head-of-line blocking.” |
| P99 | “P99 tells you whether the system survives real traffic, not whether the average demo looks good.” |

---

## 10. Print-Readiness Risks

| Risk | Severity | Fix |
|---|---:|---|
| Wide tables will overflow PDF pages. | P0 | Use short columns in HTML; for print, reduce font size and allow wrapping. |
| Timeline figures may have labels too small for print. | P0 | Enforce minimum 8–9 pt equivalent text in SVG; test print-to-PDF. |
| Dark diagrams may lose contrast in grayscale print. | P1 | Add shape/pattern distinction, not color-only meaning. |
| Long code/formula blocks may wrap poorly. | P1 | Use compact formula boxes and avoid long inline comments. |
| Reused diagram labels may reference old chapter numbers. | P0 | Replace “Chapter 13” or “Chapter 11” diagram subtitles when reused in Ch06. |
| Production notes may appear reader-facing. | P0 | Keep planning notes only in source comments or remove from final HTML. |
| Tables with URLs may print poorly. | P2 | Put source URLs in endnotes/references rather than main tables. |

---

## 11. Web-Readiness Risks

| Risk | Severity | Fix |
|---|---:|---|
| Existing GitHub landing page links may not yet route to Ch06 HTML. | P0 | Add `chapters/ch06_training_inference_workloads.html` to index navigation. |
| Figure SVGs may not scale on mobile. | P0 | Use responsive containers: `max-width: 100%; overflow-x: auto;`. |
| Tables may be unreadable on phone. | P0 | Use responsive table wrappers and stacked cards for critical comparisons. |
| Sidebar TOC may cover content on small screens. | P1 | Collapse sidebar into top dropdown on mobile. |
| Heavy inline SVGs may slow page load. | P2 | Keep only chapter-specific SVGs inline; link diagram reference pack for others. |
| SEO title may still say “Inference Systems Architecture.” | P0 | Update title/meta to new Ch06 title. |
| Previous/next navigation may point to old chapter names. | P1 | Validate Ch05 → Ch06 → Ch07 links. |
| Anchor IDs may be inconsistent with final headings. | P1 | Use stable IDs such as `#training-step`, `#prefill-decode`, `#continuous-batching`. |

---

## 12. Recommended Chapter Outline for Source Pack

```text
6.0  Chapter Manifesto — Workloads Before Optimizations
6.1  Workload Taxonomy: Training, Offline Inference, Batch API, Interactive Serving
6.2  The Training Step: From Batch to Weight Update
6.3  Training Memory: Activations, Gradients, Optimizer State, and Checkpoints
6.4  Training Bottlenecks: Data, Compute, Communication, and I/O
6.5  Inference Anatomy: Prefill and Decode
6.6  Serving Metrics: TTFT, TPOT, P50/P95/P99, Throughput, Cost/Token
6.7  Static Batching vs Continuous Batching
6.8  Chunked Prefill and Scheduler Fairness
6.9  Online Serving Pipeline: Router, Queue, Scheduler, Runtime, GPU
6.10 Disaggregated Prefill/Decode
6.11 Memory Pressure: Training Activations vs Inference KV Cache
6.12 Throughput-Latency-Cost Tradeoff Framework
6.13 What to Measure First: Diagnostic Playbooks
6.14 Principal Interview Lens
6.15 Key Takeaways
6.16 Review Questions
```

---

## 13. Final Readiness Score

### Current readiness estimate: **78 / 100**

| Category | Score | Notes |
|---|---:|---|
| Chapter concept | 9/10 | Strong and necessary bridge chapter. |
| Alignment to book promise | 9/10 | Strong principal-level workload framing. |
| Existing diagram support | 8/10 | Several excellent existing diagrams are reusable. |
| Technical validation readiness | 7/10 | Good formulas, but feature and benchmark claims must be carefully labeled. |
| Reader experience | 7/10 | Needs clearer taxonomy and tail-latency examples. |
| Interview usefulness | 8/10 | High potential with dedicated interview section. |
| Print readiness | 6/10 | Wide tables and dense SVG labels need production fixes. |
| Web readiness | 7/10 | Needs final links, mobile table handling, SEO title update. |

### Readiness after P0/P1 fixes: **91 / 100**

The chapter should be safe to move into Source Pack after the P0 and most P1 actions below are complete.

---

## 14. P0 / P1 / P2 Action List

### P0 — Must fix before source-pack creation

1. **Resolve title/scope mismatch.** Use final title: “Training and Inference Workloads: From Batch Training to Real-Time Serving.”
2. **Create workload taxonomy section first.** Do not start directly with inference.
3. **Add Fig 6.1 Workload Taxonomy Map.** This gives the chapter its visual spine.
4. **Adapt existing training timeline as Fig 6.2.** Remove any old chapter references and make it print-readable.
5. **Adapt existing prefill/decode diagram as Fig 6.4.** Add safe wording around compute-bound vs memory-bound.
6. **Adapt existing continuous batching diagram as Fig 6.5.** Remove universal throughput multiplier claims.
7. **Create training vs inference memory table.** Must distinguish activations/optimizer state from KV cache.
8. **Add safe confidence labels to all formulas.** Especially `6 × params × tokens`, `2 × params`, and `TPOT ≈ bytes/BW`.
9. **Validate framework feature matrix against official docs.** vLLM, TensorRT-LLM, SGLang, Dynamo/NIXL.
10. **Ensure Production Notes do not appear as reader-facing content.**

### P1 — Should fix before publication

1. Add tail-latency mini example for P50/P95/P99.
2. Add “What to measure first” diagnostic boxes after training and serving sections.
3. Add principal interview prompts with model answers.
4. Add Table 6.6 framework feature matrix with cautious wording.
5. Add throughput-latency knee curve figure.
6. Add chunked prefill timeline figure.
7. Add print-specific CSS for wide tables and formula blocks.
8. Add mobile-friendly table wrappers.
9. Add final end-of-chapter review questions.
10. Add forward references to Ch10, Ch11, Ch12, Ch14, Ch17, and Ch18.

### P2 — Nice to improve after first publication

1. Add optional benchmark exercise: replay serving trace with variable prompt/output lengths.
2. Add optional exercise: measure DataLoader wait and GPU idle time with `torch.profiler`.
3. Add optional cost/token worksheet.
4. Add mini comparison of offline batch inference vs interactive serving.
5. Add small note on multi-modal serving if space permits.
6. Add extra callout on why average GPU utilization is misleading for real-time serving.
7. Add downloadable SVGs for all Ch06 figures.

---

## 15. Production Decision

**Decision:** Proceed to Source Pack after P0 fixes.

**Rationale:** The chapter has strong strategic value, strong fit with the book’s principal-level positioning, and multiple reusable visual assets. The main remaining risk is not lack of content; it is scope control and safe labeling. If Ch06 is framed as the workload-classification chapter rather than a full serving or training deep dive, it will make the rest of the book easier to read and much stronger for interviews.

