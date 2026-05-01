---
title: "Chapter 6 Technical Validation Plan"
book: "AI/ML Infrastructure from Silicon to Scale"
chapter: "Ch06 — Training and Inference Workloads: From Batch Training to Real-Time Serving"
author: "Venkat Vinjam"
edition_note: "Current as of 2026 edition"
branch_target: "production-v1.0"
status: "Production Planning Pack"
output_path: "publishing/validation/ch06_technical_validation.md"
---

# Chapter 6 Technical Validation Plan

## 0. Validation Scope

This file validates the technical claims planned for:

> **Chapter 6 — Training and Inference Workloads: From Batch Training to Real-Time Serving**

The validation focus is on:

- Training vs inference workload differences.
- Batch training and online inference execution pipelines.
- Prefill vs decode behavior.
- Throughput vs latency and tail percentiles.
- TTFT, TPOT/TBT, P50/P95/P99.
- Static batching, dynamic batching, continuous batching, and chunked prefill.
- Model serving pipeline and scheduler behavior.
- Data loading bottlenecks.
- Activation memory, optimizer state, gradients, checkpoints.
- Distributed training communication.
- Serving framework capability claims.
- Principal-level workload discussion.

This chapter must be careful with exact performance numbers. Methods and formulas are stable; measured values are highly environment-specific.

---

## 1. Confidence Label Policy

Use the existing book labels exactly:

| Label | Use in Ch06 |
|---|---|
| `[SHIPPED]` | Publicly available hardware/software behavior or documented framework feature. |
| `[ANNOUNCED]` | Publicly announced feature or vendor roadmap item not yet broadly validated in stable production. |
| `[DERIVED FROM SHIPPED]` | First-principles formula derived from shipped specs, such as HBM bandwidth or memory size. |
| `[ESTIMATED]` | Approximation used for sizing or reasoning; must show assumptions. |
| `[REPRESENTATIVE]` | Teaching example or conceptual value, not a benchmark claim. |
| `[ENV-SPECIFIC]` | Depends on model, hardware, runtime, traffic distribution, scheduler, precision, or deployment policy. |

---

## 2. Primary Source Types Needed

| Claim category | Preferred source type | Example sources to use during source-pack creation |
|---|---|---|
| vLLM feature support | Official docs / GitHub README | `https://docs.vllm.ai/en/latest/`, `https://github.com/vllm-project/vllm` |
| TensorRT-LLM feature support | Official NVIDIA docs / GitHub | `https://nvidia.github.io/TensorRT-LLM/`, `https://github.com/NVIDIA/TensorRT-LLM` |
| SGLang feature support | Official docs / GitHub | `https://docs.sglang.ai/`, `https://github.com/sgl-project/sglang` |
| Dynamo / disaggregated serving | Official NVIDIA docs/blog | `https://docs.nvidia.com/dynamo/`, NVIDIA Dynamo/TensorRT-LLM disaggregated serving docs |
| PyTorch FSDP/DDP | Official PyTorch docs/tutorials | `https://docs.pytorch.org/docs/stable/fsdp.html`, PyTorch FSDP tutorial |
| Activation checkpointing | Official PyTorch docs/blog | `https://docs.pytorch.org/docs/stable/checkpoint.html`, PyTorch activation checkpointing blog |
| Data loading / pinned memory | Official PyTorch DataLoader docs | `https://docs.pytorch.org/docs/stable/data.html` |
| Training FLOP approximations | Papers + established training compute conventions | Kaplan/Hoffmann/Megatron-style dense Transformer accounting; explain assumptions |
| Serving metric definitions | Operational definitions / MLPerf Inference where applicable | MLPerf Inference docs for benchmark methodology; production definitions in chapter text |
| GPU hardware bandwidth | Vendor spec sheets / appendix | NVIDIA H100/H200/B200 docs; AMD MI300X docs; Appendix A of book |
| AllReduce / distributed communication | NCCL docs / distributed training papers | NCCL docs, Megatron-LM, ZeRO, PyTorch DDP/FSDP |

---

## 3. Validation Matrix

Each planned claim should be validated with the required columns below.

| Claim | Current value/formula | Validation status | Corrected value/safe wording | Confidence label | Source type needed | Recommended final wording | Priority |
|---|---|---|---|---|---|---|---|
| Training and inference are different workload regimes. | Training: throughput/MFU; serving: latency/cost/token. | Safe. | Keep conceptual distinction. | `[SHIPPED]` / conceptual | General architecture practice; internal book framing. | Training and inference use the same model family but optimize different objective functions and fail under different resource pressures. | P0 |
| Batch training is throughput-oriented. | Optimize step time, tokens/sec/GPU, MFU. | Safe. | Add caveat for quality/time-to-train. | `[SHIPPED]` / conceptual | MLPerf Training methodology, distributed training docs. | Batch training optimizes useful tokens processed per unit time at target quality, usually summarized by step time, tokens/sec/GPU, and MFU. | P0 |
| Interactive inference is latency- and cost-sensitive. | Optimize TTFT, TPOT, P99, cost/token. | Safe. | Add “under traffic distribution.” | `[SHIPPED]` / conceptual | Serving docs, MLPerf Inference, production SLO practice. | Interactive serving optimizes tail latency and cost per generated token under bursty arrivals and variable request lengths. | P0 |
| Training step pipeline. | Data load → H2D → forward → loss → backward → gradient sync → optimizer → checkpoint. | Safe. | Add that order can overlap. | `[SHIPPED]` / conceptual | PyTorch training docs, DDP/FSDP docs. | A training step includes data movement, forward compute, backward compute, gradient synchronization, optimizer update, and periodic checkpointing; many stages may overlap. | P0 |
| Tokens per step. | `tokens_per_step = global_batch_size × sequence_length` | Safe with caveat. | Include packed/variable-length caveat. | `[ESTIMATED]` | Training pipeline convention. | For fixed-length training, tokens per step are approximately global batch size times sequence length; packed or variable-length datasets require summing actual non-padding tokens. | P0 |
| Tokens/sec/GPU. | `tokens/sec/GPU = tokens_per_step / step_time / GPU_count` | Safe. | Use actual non-padding tokens. | `[DERIVED FROM SHIPPED]` / math | Internal derivation. | Tokens/sec/GPU is the actual useful token count per step divided by step time and GPU count. | P0 |
| Dense Transformer training FLOPs. | `training_FLOPs ≈ 6 × N_params × N_training_tokens` | Safe with caveat. | State dense decoder-only approximation. | `[ESTIMATED]` | Training compute papers; model FLOP conventions. | For dense Transformer training, a useful first-order estimate is `6 × parameters × training tokens`: roughly 2× for forward and 4× for backward. | P0 |
| Forward-only dense inference FLOPs. | `inference_FLOPs ≈ 2 × N_params × generated_tokens` | Safe with caveat. | Mention attention and MoE caveats. | `[ESTIMATED]` | Transformer FLOP accounting. | Dense forward inference is often estimated as about `2 × parameters` FLOPs per generated token before adding attention, routing, and runtime overheads. | P0 |
| MFU definition. | `MFU = achieved_model_FLOPs/s ÷ peak_hardware_FLOPs/s` | Safe with wording. | Use model FLOPs, not hardware instructions. | `[ESTIMATED]` / `[ENV-SPECIFIC]` | Megatron/MLPerf-style metrics; internal Ch01. | MFU estimates how much of theoretical peak compute is converted into useful model FLOPs. | P0 |
| HFU definition. | Hardware FLOPs issued ÷ peak. | Safe with caveat. | Distinguish from MFU. | `[ENV-SPECIFIC]` | Profiler/hardware counter docs. | HFU measures hardware-level math utilization; MFU measures useful model computation. They diverge when recomputation, padding, or inefficient kernels inflate work. | P1 |
| Data loading can bottleneck training. | CPU/tokenization/H2D can idle GPU. | Safe. | Avoid fixed worker-count rules. | `[ENV-SPECIFIC]` | PyTorch DataLoader docs; profiler traces. | Data loading, tokenization, packing, and host-to-device transfer can starve GPUs; validate with step-time and profiler traces before tuning kernels. | P0 |
| `pin_memory=True` can improve H2D transfer. | Pinned memory enables faster transfer to CUDA devices. | Safe. | Mention depends on transfer path and workload. | `[SHIPPED]` | PyTorch DataLoader docs. | In PyTorch, pinned host memory can improve CPU-to-GPU transfer behavior, but total impact depends on data size, workers, preprocessing, and overlap. | P1 |
| Activation memory grows with batch, sequence, layers, hidden size. | No fixed formula in Ch06. | Safe. | Keep qualitative or defer formulas. | `[ESTIMATED]` | Transformer training references; PyTorch activation checkpointing docs. | Activation memory generally grows with batch size, sequence length, layer count, and hidden dimension; checkpointing trades recomputation for lower activation storage. | P0 |
| Activation checkpointing trades compute for memory. | Recompute selected activations during backward. | Safe. | Include “can increase runtime.” | `[SHIPPED]` | PyTorch checkpoint docs/blog. | Activation checkpointing reduces saved activation memory by recomputing selected regions during backward, trading extra compute for lower peak memory. | P0 |
| AdamW optimizer state can dominate training memory. | Moments and possible master weights add multiple bytes/param. | Safe with implementation caveat. | Avoid universal “16 bytes” unless explaining convention. | `[ESTIMATED]` | PyTorch optimizer behavior; mixed precision docs. | Optimizer state can exceed weight memory during training; exact bytes per parameter depend on precision, master-weight policy, sharding, and optimizer implementation. | P0 |
| AdamW common memory accounting. | Weight + grad + m + v + master weight can be ~12–16 bytes/param depending setup. | Needs caveat. | Use as representative, not exact. | `[REPRESENTATIVE]` | Framework docs / implementation-specific. | A common AdamW mixed-precision budget is multiple copies per parameter for weights, gradients, and optimizer moments; use the framework’s actual state dict to compute exact memory. | P1 |
| Checkpointing is both I/O and recovery policy. | Periodic model/optimizer state writes. | Safe. | Avoid fixed overhead. | `[ENV-SPECIFIC]` | PyTorch checkpoint/FSDP docs, storage docs. | Checkpoint frequency trades lost work after failure against storage bandwidth and step-time overhead. | P0 |
| Distributed training communication occurs during gradient sync. | DDP uses gradient all-reduce; FSDP shards params/grads/optimizer states. | Safe. | Keep deep details for Ch10/Ch14. | `[SHIPPED]` | PyTorch DDP/FSDP docs; NCCL docs. | Data-parallel training synchronizes gradients across ranks; sharded approaches reduce memory at the cost of more complex communication. | P0 |
| FSDP shards parameters, gradients, and optimizer state. | FSDP reduces memory by sharding across workers. | Safe. | Validate version-specific API wording. | `[SHIPPED]` | PyTorch FSDP docs/tutorial. | FSDP reduces per-GPU memory by sharding model parameters, gradients, and optimizer states across ranks. | P1 |
| AllReduce can dominate at scale. | Communication share grows with model/parallelism/topology. | Safe with caveat. | Avoid exact threshold unless representative. | `[ENV-SPECIFIC]` | NCCL docs, Megatron/ZeRO papers. | At scale, gradient synchronization and communication overlap can determine step time; measure AllReduce share before assuming compute is the bottleneck. | P0 |
| Prefill definition. | Processes all prompt tokens and builds initial KV cache. | Safe. | Use “context phase” synonym if relevant. | `[SHIPPED]` / conceptual | vLLM/TensorRT-LLM docs. | Prefill processes the input prompt in parallel and produces the initial KV cache used by subsequent decode steps. | P0 |
| Decode definition. | Generates one output token per active sequence per iteration. | Safe. | Mention cached KV. | `[SHIPPED]` / conceptual | vLLM/TensorRT-LLM docs. | Decode repeatedly generates the next token while attending to previously cached keys and values. | P0 |
| Prefill is often compute-bound. | Long prompts and batched GEMMs use tensor cores well. | Safe with caveat. | Say “often,” not always. | `[DERIVED FROM SHIPPED]` / `[ENV-SPECIFIC]` | Roofline reasoning; profiler traces. | For sufficiently long prompts and efficient batching, prefill is often compute-heavy because it uses large matrix operations over many prompt tokens. | P0 |
| Decode is often memory-bandwidth-bound. | Small-batch decode repeatedly streams weights/KV. | Safe with caveat. | Say “often at small batch.” | `[DERIVED FROM SHIPPED]` / `[ENV-SPECIFIC]` | Roofline reasoning; GPU bandwidth specs. | At small batch, decode is often constrained by HBM bandwidth because each step touches large model weights and active KV cache for little new work. | P0 |
| TTFT decomposition. | `TTFT = queue_time + prefill_time + first_decode_time` | Safe. | Add tokenizer/network if measured end-to-end. | `[ESTIMATED]` / operational model | Serving traces / runtime docs. | A useful operational decomposition is `TTFT ≈ queue + tokenization + prefill + first decode + network overhead`; omit components only when measuring inside the server. | P0 |
| TPOT lower bound. | `TPOT_lower_bound ≈ model_bytes_touched_per_token / effective_HBM_BW` | Safe as lower bound. | Explicit lower-bound wording. | `[DERIVED FROM SHIPPED]` | GPU bandwidth spec + model memory. | A first-order lower bound for small-batch decode is model/KV bytes touched per token divided by effective memory bandwidth; real TPOT is higher due to non-ideal bandwidth, attention, scheduler, and runtime overhead. | P0 |
| P50 definition. | Median request latency. | Safe. | Exact wording. | `[SHIPPED]` / stats | Standard statistics. | P50 means 50% of requests are at or below that latency. | P0 |
| P95 definition. | 95th percentile. | Safe. | Exact wording. | `[SHIPPED]` / stats | Standard statistics. | P95 means 95% of requests are at or below that latency and 5% are worse. | P0 |
| P99 definition. | 99th percentile. | Safe. | Exact wording. | `[SHIPPED]` / stats | Standard statistics. | P99 means 99% of requests are at or below that latency and 1% are worse. | P0 |
| Throughput-latency tradeoff. | Larger batches improve throughput but can worsen queue time/tail latency. | Safe. | Include queueing caveat. | `[ENV-SPECIFIC]` | Queueing theory; serving measurements. | Larger batches often improve hardware efficiency and cost/token, but can increase queueing delay and tail latency under interactive traffic. | P0 |
| Cost per token. | `cost/token = total serving cost / tokens served` | Safe. | Define numerator. | `[ESTIMATED]` / `[ENV-SPECIFIC]` | Internal TCO model, finance assumptions. | Cost per token divides amortized infrastructure and operating cost by delivered tokens over the same time window. | P1 |
| Static batching. | Requests start/finish as a batch. | Safe. | Some systems implement variants. | `[SHIPPED]` / conceptual | Serving framework docs. | Static batching is simple but wastes capacity when generation lengths vary because completed slots wait for the longest request. | P0 |
| Dynamic batching. | Groups arrivals within a short window. | Safe. | Distinguish from continuous batching. | `[SHIPPED]` / conceptual | Serving framework docs. | Dynamic batching groups nearby arrivals to improve efficiency, but it still may not refill decode slots at every iteration. | P1 |
| Continuous batching. | Admits new requests at decode iteration boundaries. | Safe. | Exact implementation differs by engine. | `[SHIPPED]` | vLLM/SGLang/TensorRT-LLM docs. | Continuous batching refills available decode slots as requests complete, improving utilization for variable-length generation. | P0 |
| Continuous batching improves throughput vs static batching. | Existing diagrams may imply large multipliers. | Safe if qualified. | Do not use universal exact multiplier. | `[ENV-SPECIFIC]` | Benchmarks with workload description. | Continuous batching can materially improve throughput for variable-length serving, but the gain depends on traffic mix, model, runtime, and SLO constraints. | P0 |
| Chunked prefill. | Splits long prefill into chunks and interleaves with decode. | Safe. | Validate exact framework behavior. | `[SHIPPED]` for feature, `[ENV-SPECIFIC]` for benefit | vLLM/TensorRT-LLM/SGLang docs. | Chunked prefill breaks long prompts into smaller work units so active decode streams can continue making progress. | P0 |
| Chunked prefill improves P99. | Can reduce long-prompt interference. | Needs caveat. | Use “can reduce,” not “will.” | `[ENV-SPECIFIC]` | Runtime docs + measured trace. | Chunked prefill can reduce P99 TTFT/TPOT when long prompts otherwise block decode iterations; validate with production trace replay. | P0 |
| Online serving pipeline. | Gateway → tokenizer → router → queue → scheduler → runtime → GPU → stream. | Safe. | Components vary by product. | `[REPRESENTATIVE]` | Serving architecture docs. | A production serving path typically includes request routing, tokenization, admission control, scheduling, runtime execution, token streaming, and telemetry. | P0 |
| Admission control protects latency and memory. | Reject/queue/preempt based on capacity. | Safe. | Avoid universal thresholds. | `[ENV-SPECIFIC]` | Runtime docs, SRE practice. | Admission control prevents overload from turning into global tail-latency failure or KV-memory exhaustion. | P1 |
| KV cache grows with active sequences and context length. | `KV ∝ active_sequences × sequence_length × layers × kv_heads × head_dim × bytes` | Safe. | Deep formula belongs to Ch11. | `[DERIVED FROM SHIPPED]` | Ch11 KV formula; vLLM PagedAttention paper/docs. | In serving, KV cache scales with active sequence count and context length, which is why long-context concurrency is often memory-limited. | P0 |
| KV cache is the central serving memory pressure. | Especially decoder-only LLMs. | Safe. | Mention model-dependent. | `[SHIPPED]` / conceptual | vLLM/PagedAttention docs; Ch11. | For decoder-only LLM serving, KV cache is often the memory resource that determines concurrency after weights are loaded. | P0 |
| 82% KV utilization alert. | Existing internal guardrail. | Needs caveat. | Do not present as universal. | `[ENV-SPECIFIC]` | Production policy / measured deployment. | A KV-utilization alert threshold such as 80–85% can be a useful production guardrail, but it must be tuned to the runtime’s eviction and preemption behavior. | P1 |
| Disaggregated prefill/decode. | Separate prefill and decode GPU pools with KV transfer. | Safe as architecture. | Feature maturity varies. | `[SHIPPED]` / `[ANNOUNCED]` depending runtime | NVIDIA Dynamo/TRT-LLM/vLLM/SGLang docs. | Disaggregated serving separates compute-heavy prefill from bandwidth-sensitive decode and transfers KV state between pools. | P0 |
| Disaggregation improves cost/latency. | Existing assets may say 30–50% cost reduction. | Risky. | Remove exact number unless sourced. | `[ENV-SPECIFIC]` | Specific benchmark report. | Disaggregation can improve cost or latency when prefill and decode resource needs differ significantly; quantify only for a measured workload. | P0 |
| vLLM supports PagedAttention, continuous batching, chunked prefill, prefix caching. | Current docs list these features. | Safe; validate final version. | Add version/date. | `[SHIPPED]` | Official vLLM docs/GitHub. | Current vLLM documentation describes PagedAttention, continuous batching, chunked prefill, prefix caching, and related serving optimizations. | P0 |
| vLLM disaggregated prefill/decode support. | Docs include disaggregated prefilling feature pages. | Version-specific. | Mark experimental/version-specific if docs say so. | `[SHIPPED]` or `[ANNOUNCED]` based on final docs | Official vLLM docs. | vLLM documents disaggregated prefill capabilities; exact maturity and deployment pattern should be validated for the selected release. | P1 |
| TensorRT-LLM supports in-flight batching, KV cache, paged attention, chunked context/prefill. | Current docs list these concepts. | Safe; validate release. | Use official terms. | `[SHIPPED]` | NVIDIA TensorRT-LLM docs. | TensorRT-LLM documents in-flight batching, KV cache, paged/context attention, chunked context, scheduler, and parallelism features. | P0 |
| TensorRT-LLM disaggregated serving. | NVIDIA docs/blog describe it. | Safe for documented releases. | Version-specific. | `[SHIPPED]` / `[ANNOUNCED]` | NVIDIA docs/blog. | NVIDIA TensorRT-LLM/Dynamo documentation describes disaggregated serving patterns for separating context/prefill and generation/decode phases. | P1 |
| SGLang supports RadixAttention and prefix caching. | Current docs list RadixAttention. | Safe. | Add version/date. | `[SHIPPED]` | SGLang official docs/GitHub. | SGLang documentation describes RadixAttention for prefix caching and efficient reuse of shared KV state. | P0 |
| SGLang supports continuous batching, disaggregation, speculative decoding. | Current project docs mention these features. | Safe but version-specific. | Validate exact release. | `[SHIPPED]` / `[ANNOUNCED]` | SGLang docs/GitHub. | SGLang lists modern serving features including prefix caching, continuous batching, chunked prefill, speculative decoding, and parallelism support; validate exact release behavior before final publication. | P1 |
| Offline inference prioritizes throughput over latency. | Batch jobs can queue work. | Safe. | Mention deadline constraints. | `[SHIPPED]` / conceptual | Inference benchmark practice. | Offline inference can use larger batches and looser latency targets, so throughput per GPU-hour is usually the dominant metric. | P1 |
| Batch API serving sits between offline and interactive. | Bounded latency, higher batching. | Safe. | Define. | `[REPRESENTATIVE]` | Serving architecture practice. | Batch API serving accepts some queueing delay to improve throughput, but still enforces a latency SLO. | P2 |
| Average GPU utilization can mislead for serving. | P99 may fail at moderate average utilization. | Safe with caveat. | Use queueing explanation. | `[ENV-SPECIFIC]` | Queueing theory, production traces. | Average utilization can hide bursty queues, long prompts, and tail latency; serving systems must track percentile latency and queue depth. | P0 |
| 65–75% serving utilization target. | Existing heuristic. | Needs caveat. | Use representative range only. | `[REPRESENTATIVE]` | Operational heuristic / trace-specific. | Some interactive systems intentionally run below maximum utilization to preserve P99 headroom; the safe operating range must be derived from trace replay. | P1 |
| Principal-level answer must name tradeoffs. | Architecture communication principle. | Safe. | Keep. | `[SHIPPED]` / conceptual | Ch18 framing; interview practice. | A principal-level workload discussion names the metric, bottleneck, tradeoff, and validation experiment before proposing an optimization. | P0 |

---

## 4. Formula Validation Details

### 4.1 Training FLOP approximation

Current formula:

```text
training_FLOPs ≈ 6 × N_params × N_training_tokens
```

Validation status: safe as `[ESTIMATED]` for dense Transformer training.

Corrected/safe wording:

> For dense Transformer training, a widely used first-order estimate is `6 × parameters × training tokens`: roughly `2 × parameters` FLOPs per token for the forward pass and roughly `4 × parameters` for the backward pass. This is an estimate, not an exact accounting of attention, embeddings, MoE routing, recomputation, padding, or implementation details.

Priority: P0.

### 4.2 Inference forward FLOP approximation

Current formula:

```text
inference_FLOPs ≈ 2 × N_params × generated_tokens
```

Validation status: safe as `[ESTIMATED]` for dense forward inference.

Corrected/safe wording:

> Dense forward inference is often estimated as `2 × parameters` FLOPs per generated token, before adding attention, KV-cache, MoE routing, and runtime overheads.

Priority: P0.

### 4.3 TTFT decomposition

Current formula:

```text
TTFT = queue_time + prefill_time + first_decode_time
```

Corrected/safe wording:

```text
TTFT ≈ queue_time + tokenization_time + prefill_time + first_decode_time + response_overhead
```

Use a shorter formula inside model-server scope if tokenization/network are outside the measurement boundary.

Confidence label: `[ESTIMATED]` or `[ENV-SPECIFIC]` for measured values.

Priority: P0.

### 4.4 TPOT lower-bound model

Current formula:

```text
TPOT_lower_bound ≈ bytes_touched_per_decode_step / effective_HBM_bandwidth
```

Corrected/safe wording:

> At small batch, decode TPOT is often bounded below by the bytes touched per decode step divided by effective HBM bandwidth. Real TPOT is higher because kernels do not achieve perfect bandwidth, attention and KV-cache reads add overhead, scheduling adds delay, and multi-GPU communication may be present.

Confidence label: `[DERIVED FROM SHIPPED]` for the formula using shipped HBM specs; `[ENV-SPECIFIC]` for actual measurements.

Priority: P0.

### 4.5 Tokens/sec/GPU

Formula:

```text
tokens_per_second_per_GPU = useful_tokens_per_step / step_time_seconds / GPU_count
```

Safe wording:

> Use useful non-padding tokens when measuring training throughput; otherwise padding-heavy workloads will overstate progress.

Confidence label: `[DERIVED FROM SHIPPED]` / math.

Priority: P0.

### 4.6 Cost/token

Formula:

```text
cost_per_token = total_cost_over_window / tokens_served_over_window
```

Safe wording:

> The numerator must include the same accounting window as the denominator: GPU amortization or rental cost, power, cooling, host infrastructure, networking, storage, software overhead, and operations as appropriate.

Confidence label: `[ESTIMATED]` / `[ENV-SPECIFIC]`.

Priority: P1.

---

## 5. Benchmark and Numeric Claim Rules

### 5.1 Allowed numeric claim categories

| Category | Allowed? | Required label |
|---|---:|---|
| Pure definitions such as P95/P99 | Yes | No hardware label needed, but can mark conceptual |
| Formula-derived examples with shown assumptions | Yes | `[ESTIMATED]` or `[DERIVED FROM SHIPPED]` |
| Public framework feature support | Yes | `[SHIPPED]` if official docs confirm |
| Measured latency/throughput numbers | Only with source or as teaching example | `[ENV-SPECIFIC]` or `[REPRESENTATIVE]` |
| Framework performance multipliers | Avoid unless benchmark source is specific | `[ENV-SPECIFIC]` |
| Universal utilization thresholds | Avoid | `[REPRESENTATIVE]` only if teaching heuristic |

### 5.2 Unsafe wording to avoid

Do not write:

- “Continuous batching improves throughput by 10–30×.”
- “Decode is always memory-bound.”
- “Prefill is always compute-bound.”
- “Run serving at 70% utilization.”
- “82% KV utilization is the universal alert threshold.”
- “Disaggregated serving reduces cost by 30–50%.”

Use instead:

- “Continuous batching can materially improve throughput for variable-length generation; exact gains are workload- and runtime-specific.”
- “Decode is often memory-bandwidth-bound at small batch.”
- “Prefill is often compute-heavy for long prompts and adequate batching.”
- “Interactive serving usually needs headroom below maximum utilization to protect P99 latency.”
- “KV alert thresholds must be tuned to runtime behavior and traffic distribution.”
- “Disaggregated serving can improve cost or latency when prefill and decode resource profiles differ significantly.”

---

## 6. Framework Feature Validation Notes

### vLLM

Planned safe wording:

> Current vLLM documentation describes a high-throughput serving engine with PagedAttention, continuous batching, chunked prefill, prefix caching, quantization options, and related serving optimizations. Disaggregated prefill/decode support should be described with the exact maturity level and release wording used by the selected vLLM docs.

Confidence labels:

- Core documented features: `[SHIPPED]`
- Experimental/version-specific features: `[ANNOUNCED]` or `[SHIPPED]` based on current docs
- Performance impact: `[ENV-SPECIFIC]`

Priority: P0.

### TensorRT-LLM

Planned safe wording:

> TensorRT-LLM documents optimized LLM inference features such as in-flight batching, paged/context attention, KV-cache management, scheduler behavior, chunked context/prefill, quantization, and multi-GPU/multi-node deployment patterns. Disaggregated serving should be tied to the exact NVIDIA docs or blog version referenced.

Confidence labels:

- Documented feature support: `[SHIPPED]`
- Performance claims: `[ENV-SPECIFIC]`

Priority: P0.

### SGLang

Planned safe wording:

> SGLang documentation describes high-performance serving with RadixAttention/prefix caching and modern serving features such as continuous batching, chunked prefill, speculative decoding, and parallelism support. Exact feature behavior should be validated against the release used.

Confidence labels:

- Documented features: `[SHIPPED]`
- Workload-specific gains: `[ENV-SPECIFIC]`

Priority: P0.

### NVIDIA Dynamo / NIXL

Planned safe wording:

> NVIDIA Dynamo documentation describes distributed inference serving patterns, including disaggregated serving and KV-aware routing/transfer mechanisms. Use `[SHIPPED]` only for documented release behavior and `[ENV-SPECIFIC]` for transfer bandwidth or latency impact.

Priority: P1.

---

## 7. Source-Pack Validation Checklist

Before creating `ch06_training_inference_workloads.md` and `.html`, verify:

- [ ] New Ch06 title is used consistently in file name, page title, SEO meta, H1, sidebar, and previous/next navigation.
- [ ] Every hardware number has a confidence label.
- [ ] Every framework feature claim is matched to official docs or marked version-specific.
- [ ] Every formula states assumptions.
- [ ] No diagram carries an unsupported universal benchmark multiplier.
- [ ] Continuous batching, chunked prefill, and disaggregation benefits are marked `[ENV-SPECIFIC]` unless backed by a specific benchmark.
- [ ] The chapter distinguishes training activation/optimizer memory from inference KV-cache memory.
- [ ] P50/P95/P99 definitions are exact.
- [ ] TTFT/TPOT measurement boundaries are clear.
- [ ] Training data pipeline bottlenecks are included.
- [ ] Distributed training communication is introduced without duplicating Ch10/Ch14.
- [ ] KV cache is introduced without duplicating Ch11.
- [ ] Principal interview section includes training and serving prompts.
- [ ] Print CSS handles tables and SVGs.
- [ ] Mobile layout handles wide timelines and tables.

---

## 8. Priority Summary

### P0 claims to validate or safely word

1. Training vs inference workload distinction.
2. Training step pipeline.
3. `6 × params × tokens` training FLOP approximation.
4. Training memory components.
5. Activation checkpointing tradeoff.
6. Prefill and decode definitions.
7. Prefill compute-heavy caveat.
8. Decode memory-bandwidth caveat.
9. TTFT and TPOT definitions/formulas.
10. P50/P95/P99 definitions.
11. Continuous batching behavior.
12. Chunked prefill behavior.
13. Framework feature matrix.
14. Disaggregated prefill/decode architecture.
15. Removal or qualification of exact throughput/cost multipliers.

### P1 claims to validate or safely word

1. FSDP memory sharding.
2. AdamW exact memory accounting.
3. DataLoader pinned memory advice.
4. KV utilization thresholds.
5. Serving utilization headroom heuristics.
6. Cost/token formulas.
7. Checkpoint overhead ranges.
8. Dynamo/NIXL maturity and transfer claims.

### P2 claims to validate or defer

1. Multi-modal serving architecture details.
2. Optional framework comparisons beyond vLLM/TensorRT-LLM/SGLang.
3. Exact benchmark examples for H100/H200/MI300X.
4. Specific cost-reduction claims from disaggregation or prefix caching.

---

## 9. Final Validation Decision

**Decision:** Chapter 6 is technically safe to proceed into source-pack drafting if the P0 safe-wording rules are applied.

**Main risk:** Overstating runtime-specific serving performance and presenting representative heuristics as universal facts.

**Main fix:** Use confidence labels aggressively, keep exact values as examples unless publicly sourced, and phrase serving behavior in terms of workload-dependent bottlenecks rather than absolute rules.

