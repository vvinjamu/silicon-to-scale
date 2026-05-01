# Chapter 6 — Training and Inference Workloads: From Batch Training to Real-Time Serving

> “A GPU cluster does not run AI in the abstract. It runs workload modes, each with its own bottleneck, metric, and operating point.”

---

## Chapter Overview

Chapter 5 explained why power, thermal design, and physical infrastructure determine whether AI hardware can sustain performance.

Chapter 6 moves from physical infrastructure into workload execution.

The first five chapters built the foundation: roofline thinking, transformer arithmetic, GPU architecture, HBM behavior, and the power and thermal limits of AI infrastructure. Chapter 6 changes the unit of analysis from **hardware** to **workload**.

This shift matters because a GPU cluster does not run “AI” in the abstract. It runs specific workload modes:

- A training job that consumes a fixed dataset for days or weeks.
- An offline inference job that scores a large batch with loose latency constraints.
- A batch API service that accepts user traffic but can tolerate bounded queueing.
- An interactive streaming service where first-token and per-token latency determine user experience.

All four may use the same model architecture and the same GPU type, but they stress the system differently. Training stresses compute, activation memory, optimizer state, data pipeline throughput, checkpointing, and collective communication. Inference stresses request scheduling, KV cache, prefill/decode phase balance, HBM bandwidth, percentile latency, and cost per token.

A principal performance architect should be able to look at a workload and immediately ask:

1. What is the primary objective: throughput, latency, cost, quality, or recovery time?
2. What is the dominant resource: compute, HBM, network, storage, scheduler, or queueing headroom?
3. What metric proves the system is healthy?
4. What metric proves it is failing?
5. Which optimization moves the bottleneck rather than merely moving a number?

This chapter gives that operating model.

> **Current as of 2026 edition:** Hardware, framework features, serving engines, and runtime APIs change quickly. Quantitative claims in this chapter use confidence labels so the reader can distinguish shipped behavior, first-principles estimates, representative examples, and environment-specific production outcomes.

---

## 6.0 Workloads in One Page

Training and inference are not two versions of the same performance problem.

They optimize different objectives:

```text
Training:
  maximize useful tokens/sec/GPU, MFU, stability, and recoverability

Offline inference:
  maximize batch throughput and cost efficiency

Batch API serving:
  maximize throughput under bounded latency

Interactive serving:
  minimize TTFT and TPOT tail latency at acceptable cost/token
```

The fastest way to reason about any workload is to name four things:

1. The objective function.
2. The dominant resource.
3. The health metric.
4. The failure metric.

> **Key Takeaway:** Workload classification comes before optimization. A training bottleneck, an offline inference bottleneck, and an interactive serving bottleneck require different evidence and different fixes.

---

## 6.1 Workload Taxonomy: Four Modes, Four Objective Functions

“Training vs inference” is the common split, but production systems need a finer taxonomy. The most useful first classification is not model type. It is **how much the system can control the batch** and **how much the user cares about latency**.

## Figure Placeholder — Fig 6.1

```markdown
![Fig 6.1 — Training vs Inference Workload Map](../assets/diagrams/svg/ch06_fig_6_1_training_inference_workload_map.svg)

**Fig 6.1 — Training vs Inference Workload Map.** Training, offline inference, batch API serving, and interactive serving optimize different objective functions. The first production mistake is treating all four as the same performance problem.
```

**Figure intro:**  
Start this chapter by showing that “AI workload” is not one performance problem. Place batch training, offline inference, batch API serving, and interactive serving on a two-axis workload map.

**Figure explanation:**  
The X-axis should represent latency sensitivity. The Y-axis should represent batch dynamism. Training is stable and throughput-oriented; interactive serving is dynamic and tail-latency-oriented.

> **Key Takeaway:** The first production mistake is treating all workload modes as if they should maximize the same metric.

## Table 6.1 — Training vs Inference Workload Comparison

| Workload mode | Primary objective | Batch shape | Dominant resources | Main failure mode | First metric to inspect | Confidence |
|---|---|---|---|---|---|---|
| Batch training | Maximize useful tokens/sec/GPU and MFU over long runs | Static or planned microbatches | Compute, activation memory, optimizer state, network collectives, data pipeline | Low MFU, stragglers, checkpoint loss, data stalls | Step time breakdown + MFU | [SHIPPED] |
| Offline inference | Maximize tokens/sec or samples/sec under loose latency | Large static/dynamic batches | Compute and HBM, depending on model and batch | Poor batching, low utilization, high cost per token | Tokens/sec/GPU | [SHIPPED] |
| Batch API serving | Balance throughput and bounded latency | Dynamic, but queueing allowed | Scheduler, queue depth, runtime batching, HBM, KV cache | P95/P99 drift under load | Throughput under SLA | [ENV-SPECIFIC] |
| Interactive streaming serving | Minimize TTFT and TPOT tail latency at acceptable cost | Highly dynamic | KV cache, HBM bandwidth, scheduler fairness, queueing headroom | P99 TTFT/TPOT cliff, OOM, preemption, user-visible stalls | P99 TTFT, P99 TPOT, KV utilization | [ENV-SPECIFIC] |

The table is intentionally metric-first. A principal engineer does not begin with a tool. They begin with the objective function.

**Training objective:** keep the expensive training fleet doing useful work for as much of the wall-clock time as possible, while preserving numerical stability and recoverability.

**Serving objective:** deliver tokens to users inside a latency SLO at the lowest cost per token, while handling bursty demand, heterogeneous prompts, and multi-tenant fairness.

The distinction is simple, but it drives almost every architectural decision in the rest of the book.

---

## 6.2 Batch Training Pipeline: The Anatomy of a Step

A training job turns data into updated model weights. One step appears simple in Python, but at scale it becomes a pipeline of CPU work, GPU kernels, memory allocation, network collectives, and storage writes.

```python
# Representative PyTorch-style training step [REPRESENTATIVE]
for batch in dataloader:
    batch = move_to_gpu(batch)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch.inputs)
    loss = criterion(outputs, batch.labels)
    loss.backward()
    optimizer.step()

    if step % checkpoint_interval == 0:
        save_checkpoint(model, optimizer, step)
```

This loop hides the real performance story. The wall-clock step is not only forward and backward compute.

## Figure Placeholder — Fig 6.2

```markdown
![Fig 6.2 — Training Step Waterfall](../assets/diagrams/svg/ch06_fig_6_2_training_step_waterfall.svg)

**Fig 6.2 — Training Step Waterfall.** A training step is not just forward and backward compute. Data stalls, gradient synchronization, optimizer state updates, and checkpoint I/O can dominate at scale.
```

**Figure intro:**  
Show a horizontal training-step waterfall: data load, host-to-device transfer, forward pass, loss, backward pass, gradient synchronization, optimizer step, and checkpoint.

**Figure explanation:**  
Color forward/backward as compute-heavy, optimizer update as memory-heavy, gradient synchronization as communication-heavy, and checkpoint as I/O-heavy. Leave room for future measured examples.

> **Key Takeaway:** A training step is a pipeline; the slowest stage, not the Python loop, determines throughput.

## Table 6.2 — Training Step Breakdown

| Stage | What happens | Common bottleneck | What to measure first | Typical mitigation | Confidence |
|---|---|---|---|---|---|
| Data read | Read samples from storage/object store/local cache | Storage or metadata latency | Data-loader wait time, disk/object-store throughput | Sharding, caching, prefetching, larger read blocks | [ENV-SPECIFIC] |
| Tokenize / decode / augment | CPU-side preprocessing | CPU saturation or Python overhead | CPU utilization, dataloader queue depth | Pre-tokenize, increase workers, vectorize preprocessing | [ENV-SPECIFIC] |
| Batch pack | Create fixed-length or packed sequences | Padding waste, CPU overhead | Useful tokens / padded tokens | Sequence packing, bucketing, dynamic shapes | [REPRESENTATIVE] |
| H2D transfer | Move batch to GPU | PCIe/NVLink host path, pageable memory | Copy time, overlap with compute | Pinned memory, non-blocking copy, prefetch | [SHIPPED] |
| Forward pass | Compute activations and logits | Tensor core efficiency, HBM traffic | Kernel timeline, achieved TFLOPS | Fusion, compiled graphs, FlashAttention, better shapes | [DERIVED FROM SHIPPED] |
| Backward pass | Compute gradients | Compute + activation reads | Backward time, activation memory | Gradient checkpointing, fusion, mixed precision | [DERIVED FROM SHIPPED] |
| Gradient sync | AllReduce or reduce-scatter gradients | Network / topology / overlap | Collective time, bus bandwidth, bucket timing | Bucket tuning, topology-aware placement, overlap | [ENV-SPECIFIC] |
| Optimizer step | Update weights and optimizer states | Memory bandwidth and optimizer state size | Optimizer time, HBM bytes | Fused optimizer, ZeRO/FSDP, lower precision states | [ENV-SPECIFIC] |
| Checkpoint | Persist model/optimizer/RNG state | Filesystem/object-store throughput | Checkpoint duration, GB/s, pause time | Async checkpoint, sharded checkpoint, lower frequency | [ENV-SPECIFIC] |

### Training FLOP Rule of Thumb

For dense Transformer training, a common first-order estimate is:

```text
training_FLOPs ≈ 6 × N_parameters × N_training_tokens
```

Confidence label: **[ESTIMATED]**

The intuition is that dense forward inference costs roughly `2 × parameters` FLOPs per token, while training includes forward, backward through activations, and gradient computation. The approximation is useful for first-order sizing, but it does not exactly account for attention overhead, embeddings, padding, activation recomputation, MoE sparsity, communication, or implementation details.

A principal-level answer should say:

> “I would start with the 6N dense training estimate for a first sizing pass, then correct it using measured step time and MFU from the actual stack.”

---

## 6.3 Training Metrics: Step Time, Tokens/sec/GPU, MFU, and Stall Time

Training health is not a single number. A job can report high aggregate throughput while wasting GPUs on padding, data stalls, or synchronization imbalance. The core training metrics are:

```text
step_time = data_time + forward_time + backward_time + communication_time + optimizer_time + checkpoint_overhead
```

```text
tokens_per_second_per_GPU = useful_training_tokens / (step_time_seconds × GPU_count)
```

```text
MFU = achieved_model_FLOPs_per_second / theoretical_peak_FLOPs_per_second
```

Confidence labels: **[DERIVED FROM SHIPPED]** for the formulas, **[ENV-SPECIFIC]** for measured values.

### Useful tokens matter

For language model training, the numerator should be **useful non-padding tokens**, not merely allocated token positions. If a batch contains 40% padding, then reporting padded tokens/sec makes the system look healthier than it is.

### MFU vs GPU utilization

GPU utilization tells you whether the GPU is busy. MFU tells you whether the busy time is doing useful model math. A GPU can be “utilized” while running inefficient kernels, waiting on memory, or spinning through padding-heavy sequences.

A practical hierarchy:

1. **Step time** tells you whether the job is progressing.
2. **Tokens/sec/GPU** tells you throughput normalized by GPU count.
3. **MFU** tells you how close useful model math is to hardware potential.
4. **Stall breakdown** tells you where to optimize.

For production training, the most useful dashboard is not a single utilization chart. It is a stacked step-time view with data, compute, communication, optimizer, and checkpoint components.

---

## 6.4 Training Memory: Weights, Gradients, Activations, Optimizer State

Training requires more memory than inference because the system must retain or reconstruct information needed for backpropagation.

### Training memory categories

| Memory category | Why it exists | Scales with | Typical pressure | Common mitigation | Confidence |
|---|---|---|---|---|---|
| Weights | Model parameters | Parameter count × precision | Always resident or sharded | Tensor parallelism, FSDP/ZeRO, quantized loading for inference only | [SHIPPED] |
| Gradients | Parameter gradients | Parameter count × precision | During backward and optimizer step | Gradient sharding, reduce-scatter, accumulation strategy | [SHIPPED] |
| Optimizer state | Adam moments, master weights, metadata | Often multiple copies per parameter | Large memory multiplier | ZeRO/FSDP, optimizer sharding, lower precision optimizer | [ENV-SPECIFIC] |
| Activations | Intermediate tensors for backward | Batch × sequence × hidden × layers | Often the training memory limiter | Activation checkpointing/recomputation, sequence parallelism | [DERIVED FROM SHIPPED] |
| Temporary buffers | Workspace, communication buckets, fused-kernel scratch | Runtime and operator dependent | Fragmentation / OOM | Preallocation, graph capture, bucket tuning | [ENV-SPECIFIC] |

### AdamW state rule of thumb

A common mixed-precision Adam-style training budget is roughly **many bytes per parameter** once weights, gradients, master weights, and optimizer states are included. The exact value depends on precision choices and sharding strategy.

Use safe wording:

> “Adam-style training commonly requires multiple state tensors in addition to the model weights. Without sharding, optimizer state can dominate memory. The exact bytes per parameter depend on precision, framework, and optimizer implementation.” **[ENV-SPECIFIC]**

Do not present a single universal value unless the precision convention is stated.

### Activation memory is not optional

In inference, intermediate activations can be freed once the forward pass finishes. In training, backward needs them. If memory is insufficient, the system can recompute activations during backward. This trades extra compute for lower memory.

```text
activation_checkpointing_tradeoff:
    lower_activation_memory
    higher_recompute_FLOPs
    longer_step_time
```

Confidence label: **[DERIVED FROM SHIPPED]**

A principal-level discussion should not say only, “Use checkpointing.” It should say:

> “Checkpointing is a memory-for-compute trade. I would enable it only after estimating whether activation memory is the limiting factor and measuring the recompute cost against the gain in batch size or model size.”

---

## 6.5 Distributed Training Communication: Why Scaling Is Not Free

When a model trains across multiple GPUs, the GPUs must exchange information. In data parallel training, each replica computes gradients for a different microbatch, then synchronizes gradients so all replicas apply the same update.

The basic pattern is:

```text
local forward/backward → gradients → AllReduce or reduce-scatter → optimizer update
```

Confidence label: **[SHIPPED]**

Communication time becomes visible when:

- Gradients are large.
- Compute per byte is low.
- Network bandwidth is limited.
- Buckets are not overlapped with backward compute.
- One rank is slower and creates a straggler.
- Topology placement crosses a slow boundary.

### The principal-level mental model

Distributed training scaling efficiency is controlled by the ratio:

```text
communication_time / useful_compute_time
```

If communication time is small and well overlapped, scaling is efficient. If communication time grows with GPU count and becomes exposed on the critical path, adding GPUs can reduce time-to-train less than expected.

This chapter introduces the communication problem. Chapter 10 covers 4D parallelism and ZeRO/FSDP in detail. Chapter 14 covers NCCL, InfiniBand, RDMA, AllReduce algorithms, and network topology.

---

## 6.6 Data Loading and Checkpointing: The Non-GPU Bottlenecks

Training clusters are often purchased as if GPUs are the only bottleneck. Production traces usually teach a harder lesson: the job is only as fast as the slowest pipeline stage.

### Data loading bottlenecks

A GPU can starve when data arrives late. Causes include:

- Small random reads from object storage.
- Too few dataloader workers.
- CPU tokenization or image decode overhead.
- Compression/decompression bottlenecks.
- Inefficient shuffling across remote storage.
- Padding waste from poor sequence packing.

The direct symptom is a gap before forward kernels start. The dashboard symptom is low MFU even though the model code is correct.

### Checkpointing bottlenecks

Checkpointing protects long training jobs from failure, but it writes large state: model weights, optimizer state, scheduler state, RNG state, dataloader position, and sometimes activation/offload metadata.

A safe checkpoint objective is:

```text
checkpoint_overhead_fraction = checkpoint_pause_time / checkpoint_interval_wall_time
```

Confidence label: **[DERIVED FROM SHIPPED]**

If checkpointing pauses a large job too often, the cluster loses expensive GPU-hours. If checkpointing happens too rarely, failures lose too much work. This is a reliability-throughput tradeoff, not a pure storage decision.

---

## 6.7 Inference Anatomy: Prefill and Decode

Autoregressive LLM inference has two phases:

1. **Prefill:** process the input prompt and populate the KV cache.
2. **Decode:** generate output tokens one at a time, reusing the KV cache.

## Figure Placeholder — Fig 6.3

```markdown
![Fig 6.3 — Prefill vs Decode Regime Split](../assets/diagrams/svg/ch06_fig_6_3_prefill_decode_regime_split.svg)

**Fig 6.3 — Prefill vs Decode Regime Split.** Prefill processes the prompt in parallel and is usually compute-heavy. Decode generates one token at a time and is often bounded by HBM bandwidth and KV-cache traffic at small batch.
```

**Figure intro:**  
Show a two-panel diagram: prefill processes many prompt tokens in parallel; decode emits one new token at a time.

**Figure explanation:**  
Prefill uses large batched GEMMs and is often compute-heavy for long prompts. Decode repeatedly reads weights and KV state and is often memory-bandwidth-sensitive at small batch.

> **Key Takeaway:** LLM serving is not one forward pass; it is a phase-split workload with different bottlenecks per phase.

## Table 6.3 — Prefill vs Decode Regime Comparison

| Dimension | Prefill | Decode | Confidence |
|---|---|---|---|
| Work performed | Process all prompt tokens | Generate one token per active sequence per iteration | [SHIPPED] |
| Parallelism | High across sequence length and batch | Limited by one-step autoregressive dependency | [SHIPPED] |
| Common bottleneck | Tensor core compute for long prompts; attention for long context | HBM bandwidth, KV cache reads, scheduler overhead, sampling | [DERIVED FROM SHIPPED] |
| User-visible metric | TTFT | TPOT / inter-token latency | [SHIPPED] |
| Memory pressure | KV cache allocation begins | KV cache grows with output length | [DERIVED FROM SHIPPED] |
| Scheduler issue | Long prompts can monopolize runtime | Short iterations require careful batching | [ENV-SPECIFIC] |
| Optimization examples | Chunked prefill, prefix caching, disaggregated prefill | Continuous batching, KV cache efficiency, quantization, fair scheduling | [SHIPPED] |

### TTFT decomposition

A practical serving measurement boundary is:

```text
TTFT ≈ queue_time + tokenization_time + prefill_time + first_decode_time + response_overhead
```

Confidence label: **[ESTIMATED]** for the decomposition, **[ENV-SPECIFIC]** for measured values.

If the runtime measures only inside the engine, simplify to:

```text
engine_TTFT ≈ scheduler_wait + prefill_time + first_decode_time
```

### TPOT lower-bound model

For small-batch decode, a useful lower-bound model is:

```text
TPOT_lower_bound ≈ bytes_touched_per_decode_step / effective_HBM_bandwidth
```

Confidence label: **[DERIVED FROM SHIPPED]**

This is not a prediction of actual latency. Real TPOT is higher because kernels do not achieve ideal bandwidth, attention reads KV state, sampling and scheduler overhead exist, and multi-GPU communication may be present. The value of the formula is that it tells you why adding peak TFLOPS may not improve small-batch decode.

---

## 6.8 Serving Metrics: TTFT, TPOT, P50/P95/P99, Throughput, and Cost

Training usually optimizes long-window throughput. Serving must optimize the user-visible latency distribution.

## Table 6.4 — Serving Metrics and Root Causes

| Metric | Meaning | Why it matters | Common root causes when bad | Confidence |
|---|---|---|---|---|
| TTFT | Time to first token | User perception of responsiveness | Queueing, long prefill, cold cache, router delay | [SHIPPED] |
| TPOT / TBT | Time per output token / time between tokens | Streaming smoothness | Decode HBM pressure, KV cache traffic, scheduler interference | [SHIPPED] |
| E2E latency | Total request duration | Full user wait time | Long output, slow decode, queueing | [SHIPPED] |
| P50 | Median request behavior | Typical user experience | Useful but can hide failures | [SHIPPED] |
| P95 | High percentile tail | Early production pain indicator | Bursts, long prompts, overload, preemption | [SHIPPED] |
| P99 | Extreme tail | SLO / incident indicator | Queueing cliff, OOM, retries, noisy neighbors | [SHIPPED] |
| Tokens/sec/GPU | Throughput normalized by GPU count | Efficiency and capacity planning | Poor batching, low batch occupancy, inefficient kernels | [ENV-SPECIFIC] |
| Cost/token | Economic efficiency | Product viability at scale | Low utilization, expensive hardware, overprovisioning | [ENV-SPECIFIC] |
| KV utilization | Fraction of KV memory budget in use | Admission and OOM risk | Long context, high concurrency, fragmentation | [ENV-SPECIFIC] |

### Representative SLO example

A chatbot might target values such as:

| Metric | Representative target | Confidence |
|---|---:|---|
| TTFT P50 | < 300 ms | [REPRESENTATIVE] |
| TTFT P99 | < 800 ms | [REPRESENTATIVE] |
| TPOT P50 | < 40 ms/token | [REPRESENTATIVE] |
| TPOT P99 | < 80 ms/token | [REPRESENTATIVE] |

These are teaching examples, not universal targets. A coding assistant, voice agent, batch summarization job, and enterprise RAG service will have different acceptable values.

### Percentiles beat averages

Averages hide user pain. If 99 users receive tokens quickly and one user waits five seconds, the average may look fine while the product feels broken. A principal-level serving dashboard must track **P50, P95, and P99** separately and must correlate tail latency with queue depth, prompt length, output length, runtime batch size, and KV utilization.

---

## 6.9 Static Batching, Dynamic Batching, and Continuous Batching

Batching is the core throughput lever for inference. But unlike training, serving batches are not known in advance. Requests arrive at different times, prompts have different lengths, and outputs terminate at different steps.

### Static batching

Static batching waits for a batch, runs all requests together, and does not admit new work until the batch completes.

Problem: output lengths vary. If one request generates 200 tokens and another generates 20 tokens, the short request’s slot becomes idle while the long request continues.

### Dynamic batching

Dynamic batching forms a batch from requests that arrive within a small time window. It improves throughput but still often treats a batch as a unit.

### Continuous batching

Continuous batching admits and removes requests at decode iteration boundaries. When a sequence finishes, a new sequence can enter the next iteration.

## Figure Placeholder — Fig 6.4

```markdown
![Fig 6.4 — Static vs Continuous Batching Timeline](../assets/diagrams/svg/ch06_fig_6_4_static_vs_continuous_batching_timeline.svg)

**Fig 6.4 — Static vs Continuous Batching Timeline.** Static batching wastes GPU slots when shorter sequences finish early. Continuous batching admits new requests at token-iteration boundaries, keeping the decode batch full.
```

**Figure intro:**  
Make continuous batching intuitive with a timeline of requests that finish at different decode lengths.

**Figure explanation:**  
The static-batching panel should show idle slots waiting for the longest request. The continuous-batching panel should show new requests entering as soon as a slot becomes free at an iteration boundary.

> **Key Takeaway:** Continuous batching improves utilization by keeping decode slots full without waiting for a whole batch to finish.

Confidence label: **[SHIPPED]** for continuous batching as a modern serving-engine feature; **[ENV-SPECIFIC]** for the throughput improvement in a specific deployment.

## Table 6.5 — Scheduling and Runtime Levers

| Lever | Helps most when | Main benefit | Main risk | Confidence |
|---|---|---|---|---|
| Larger batch | Throughput-limited offline or batch API workload | Higher arithmetic intensity and GPU utilization | Higher queueing latency | [ENV-SPECIFIC] |
| Continuous batching | Output lengths are heterogeneous | Reduces slot waste | More complex scheduler and fairness policy | [SHIPPED] |
| Chunked prefill | Long prompts interfere with active decode | Reduces decode latency spikes | Chunk size must be tuned | [ENV-SPECIFIC] |
| Prefix caching | Many requests share system prompt/RAG context | Reduces prefill compute and TTFT | Requires cache hit tracking and routing awareness | [SHIPPED] |
| KV quantization | KV memory limits concurrency | More active sequences per GPU | Accuracy must be validated | [ENV-SPECIFIC] |
| Admission control | Load approaches memory or latency cliff | Protects P99 and avoids OOM | Rejections or queueing must be product-managed | [ENV-SPECIFIC] |
| Disaggregated P/D | Prefill and decode resource needs diverge | Better fleet specialization | KV transfer latency and reliability complexity | [ENV-SPECIFIC] |

---

## 6.10 Chunked Prefill: Protecting Decode from Long Prompts

A long prompt can be a latency hazard. Without chunking, a 32K-token prefill can monopolize compute and delay active decode requests. Users already receiving streamed output may see a TPOT spike.

Chunked prefill breaks a long prefill into smaller chunks and interleaves decode iterations between chunks.

## Figure Placeholder — Fig 6.5

```markdown
![Fig 6.5 — Chunked Prefill Scheduler](../assets/diagrams/svg/ch06_fig_6_5_chunked_prefill_scheduler.svg)

**Fig 6.5 — Chunked Prefill Scheduler.** Chunking a long prompt prevents prefill from monopolizing the GPU and protects active decode traffic from large TPOT spikes.
```

**Figure intro:**  
Long prompts can monopolize the runtime and delay active decode traffic. The figure should show how chunked prefill creates scheduling points.

**Figure explanation:**  
The bad case shows a long prompt blocking decode. The chunked case alternates prefill chunks with decode iterations so active users continue receiving tokens.

> **Key Takeaway:** Chunked prefill trades some prefill scheduling complexity for better decode fairness and tail latency.

A safe production statement is:

> “Chunked prefill can reduce long-prompt interference with decode latency, but the best chunk size and policy depend on model, hardware, runtime, traffic mix, and SLO.” **[ENV-SPECIFIC]**

Do not claim a universal improvement multiplier.

---

## 6.11 Disaggregated Prefill/Decode Serving

At small scale, one engine may handle both prefill and decode on the same GPU. At larger scale, prefill and decode can have sufficiently different resource profiles that separating them becomes attractive.

- **Prefill pool:** optimized for prompt processing and compute-heavy bursts.
- **Decode pool:** optimized for streaming decode, KV residency, and stable TPOT.
- **KV transfer path:** moves KV state from prefill workers to decode workers.
- **Router/admission controller:** decides which path a request takes.

## Figure Placeholder — Fig 6.8

```markdown
![Fig 6.8 — Disaggregated Prefill/Decode Architecture](../assets/diagrams/svg/ch06_fig_6_8_disaggregated_prefill_decode_architecture.svg)

**Fig 6.8 — Disaggregated Prefill/Decode Architecture.** Prefill and decode stress different hardware resources. Disaggregation can improve utilization at fleet scale, but introduces KV-transfer latency, routing, and reliability complexity.
```

**Figure intro:**  
Show why larger fleets may separate prefill from decode when the two phases stress different resources.

**Figure explanation:**  
The figure should show a router deciding between local and disaggregated paths, a compute-oriented prefill pool, a KV transfer path, a decode pool, and observability around TTFT, TPOT, and KV-transfer latency.

> **Key Takeaway:** Disaggregation is a fleet-level optimization, not a free speedup; it exchanges utilization gains for transfer and routing complexity.

Confidence labels: **[SHIPPED]** where a framework documents the feature; **[ENV-SPECIFIC]** for any throughput, latency, or cost gain.

A principal-level design discussion should include the failure cases:

- What if KV transfer fails?
- What if prefill queue is empty but decode pool is saturated?
- What if decode pool has KV memory but not enough compute headroom?
- What if a long prompt should be chunked instead of disaggregated?
- What if routing breaks prefix-cache locality?

---

## 6.12 KV Cache as the Serving Resource Bridge

The full KV cache chapter is Chapter 11. Chapter 6 only needs the serving-level intuition:

```text
KV_cache_bytes ∝ active_sequences × sequence_length × layers × KV_heads × head_dim × bytes_per_element
```

Confidence label: **[DERIVED FROM SHIPPED]**

The important operational facts are:

1. KV cache grows with active concurrency.
2. KV cache grows with context length.
3. Long-context users consume disproportionate serving capacity.
4. KV pressure can force preemption, eviction, offload, or rejection.
5. A runtime can have available compute but insufficient KV memory for new requests.

A representative guardrail is:

> “A KV-utilization alert threshold around 80–85% can be useful, but it must be tuned to the model, runtime, page size, preemption behavior, traffic distribution, and SLO.” **[ENV-SPECIFIC]**

Avoid saying “82% is always the threshold.” In this book, treat it as a production heuristic to be validated.

---

## 6.13 Runtime Serving Stack and Framework Choices

A serving request touches more layers than the model runtime alone.

## Figure Placeholder — Fig 6.7

```markdown
![Fig 6.7 — Runtime Serving Stack](../assets/diagrams/svg/ch06_fig_6_7_runtime_serving_stack.svg)

**Fig 6.7 — Runtime Serving Stack.** Serving performance is a stack problem: API routing, scheduler policy, runtime implementation, kernels, and GPU memory all shape user-visible latency.
```

**Figure intro:**  
Connect product request flow to runtime and GPU execution. The serving system is more than a model.forward call.

**Figure explanation:**  
Show client, gateway, router, scheduler, runtime, kernels, GPU, HBM weights, KV cache, and streaming tokens. Side metrics should include TTFT, TPOT, KV utilization, and prefix-cache hit rate.

> **Key Takeaway:** Serving latency is created across the stack; profiler evidence must be correlated with request-level metrics.

## Table 6.6 — Runtime Framework Comparison

| Runtime / stack | Strong fit | Key features to validate before production | Main caution | Confidence |
|---|---|---|---|---|
| vLLM | General high-throughput LLM serving, OpenAI-compatible APIs, research-to-production iteration | PagedAttention, continuous batching, chunked prefill, prefix caching, speculative decoding, disaggregated prefill support | Feature maturity and backend behavior vary by version and hardware | [SHIPPED] |
| TensorRT-LLM / Triton backend | NVIDIA production inference, optimized engines, deployment with Triton | In-flight batching, paged KV cache, chunked context/prefill, scheduler behavior, tensor/pipeline parallel support | Engine build complexity and hardware/vendor specificity | [SHIPPED] |
| SGLang | Agentic workloads, prefix-heavy workloads, structured outputs, high-performance serving | RadixAttention, continuous batching, chunked prefill, prefill-decode disaggregation, speculative decoding, parallelism modes | Fast-moving project; validate release behavior | [SHIPPED] |
| Custom runtime | Highly specialized fleet or product constraints | Tailored scheduler, custom cache policy, deep telemetry integration | High engineering and maintenance burden | [ENV-SPECIFIC] |

A safe framework statement is:

> “Modern LLM serving engines such as vLLM, TensorRT-LLM, and SGLang provide runtime features for batching, KV-cache management, and scheduling. The exact feature set and maturity must be validated against the specific release and hardware backend.” **[SHIPPED]** / **[ENV-SPECIFIC]**

---

## 6.14 Throughput-Latency Knee and Multi-Tenant Serving

Interactive serving should not blindly maximize average GPU utilization. Queueing systems have a knee: as request rate approaches service capacity, latency increases nonlinearly. P50 may remain acceptable while P99 becomes unacceptable.

## Figure Placeholder — Fig 6.6

```markdown
![Fig 6.6 — Throughput-Latency Knee Curve](../assets/diagrams/svg/ch06_fig_6_6_throughput_latency_knee_curve.svg)

**Fig 6.6 — Throughput-Latency Knee Curve.** Interactive serving should operate below the queueing cliff. The highest tokens/sec point is often not the best production operating point.
```

**Figure intro:**  
Serving systems often look healthy until they approach a utilization knee, where queueing delay grows faster than throughput.

**Figure explanation:**  
The curve should label underutilized, efficient operating range, knee, and SLA breach regions. Values are representative and environment-specific.

> **Key Takeaway:** Maximizing utilization can violate P99 latency; production serving needs headroom.

Representative operating targets such as 65–75% utilization for interactive serving are teaching examples only. The real target must be derived from trace replay and load testing.

### Multi-tenant risks

Multi-tenant serving adds additional pressure:

- One tenant’s long prompts can hurt another tenant’s decode latency.
- One tenant’s high output length can occupy decode slots for too long.
- Prefix cache locality can conflict with load balancing.
- Priority traffic needs reserved capacity or preemption policy.
- KV memory must be isolated or fairly shared.

A principal design should explicitly define fairness and isolation, not just throughput.

---

## 6.15 Symptom-to-Bottleneck Diagnostic Matrix

## Table 6.7 — Symptom-to-Bottleneck Diagnostic Matrix

| Symptom | Likely bottleneck | First evidence to collect | First safe experiment | Confidence |
|---|---|---|---|---|
| Training MFU is low, GPU gaps before forward | Data loading or H2D stalls | Nsight Systems timeline, dataloader wait, CPU utilization | Increase workers, pin memory, prefetch, cache data | [ENV-SPECIFIC] |
| Training scales poorly beyond one node | Communication exposed | AllReduce time, bus bandwidth, per-rank step time | Bucket tuning, topology-aware placement, overlap analysis | [ENV-SPECIFIC] |
| Training OOM with modest batch | Activation or optimizer state memory | Memory snapshot, activation size estimate, optimizer state size | Enable checkpointing or sharding; reduce sequence/batch | [ENV-SPECIFIC] |
| Checkpoint pauses dominate | Storage throughput or serialization | Checkpoint duration and GB/s | Sharded or async checkpoint; adjust interval | [ENV-SPECIFIC] |
| TTFT P99 spikes | Queueing or long prefill | Prompt length histogram, prefill time, queue depth | Chunked prefill, prefix caching, admission policy | [ENV-SPECIFIC] |
| TPOT P99 spikes during traffic bursts | Decode scheduler or HBM/KV pressure | Decode batch size, KV utilization, per-token timeline | Reserve headroom, continuous batching tuning, KV policy | [ENV-SPECIFIC] |
| High GPU utilization but poor user latency | Queueing cliff | P50/P95/P99 vs request rate | Lower admission rate, separate tiers, autoscale | [ENV-SPECIFIC] |
| OOM or preemptions under long context | KV cache exhaustion | KV utilization, active sequence lengths | Route long-context tier, KV quantization, admission limit | [ENV-SPECIFIC] |
| Cost/token too high | Low utilization or overprovisioning | Tokens/sec/GPU, idle time, batch occupancy | Tune batching, consolidate tenants, offline tier | [ENV-SPECIFIC] |

Diagnostic discipline matters. Do not tune before you know which resource is on the critical path.

---

## 6.16 Principal-Level Workload Discussion

In interviews and architecture reviews, the strongest answer is not a list of technologies. It is a structured workload diagnosis.

### Prompt A: “Design a training system for a 70B model.”

A principal-level answer should cover:

1. **Objective:** minimize time-to-train at acceptable cost and reliability risk.
2. **Workload shape:** tokens, sequence length, global batch, precision, target MFU.
3. **Memory budget:** weights, gradients, optimizer state, activations, temporary buffers.
4. **Parallelism need:** data, tensor, pipeline, sequence/context, ZeRO/FSDP.
5. **Communication:** gradient sync, topology, overlap, straggler detection.
6. **Data pipeline:** storage, tokenization, packing, shuffle, cache.
7. **Checkpoint/recovery:** interval, write bandwidth, restart time, lost-work budget.
8. **Measurement:** step breakdown, MFU, tokens/sec/GPU, per-rank variance.

A weak answer says: “Use distributed training and NCCL.”

A strong answer says:

> “I would first size memory and compute from parameters, sequence length, and global batch. Then I would choose the minimum parallelism needed to fit memory and the topology that keeps communication inside the fastest fabric. I would validate with step-time decomposition and MFU before optimizing kernels.”

### Prompt B: “Design an LLM serving system for real-time chat.”

A principal-level answer should cover:

1. **Objective:** TTFT/TPOT P99 under SLO at target QPS and cost/token.
2. **Traffic shape:** prompt length, output length, arrival bursts, tenant mix.
3. **Phase split:** prefill vs decode, chunked prefill, prefix caching.
4. **Memory:** weights, KV cache, context length, admission thresholds.
5. **Scheduler:** continuous batching, fairness, priority, preemption.
6. **Runtime:** vLLM/TensorRT-LLM/SGLang/custom tradeoffs.
7. **Scale-out:** routing, sticky sessions, autoscaling, optional P/D disaggregation.
8. **Measurement:** TTFT, TPOT, P50/P95/P99, queue depth, KV utilization, tokens/sec/GPU.

A weak answer says: “Use vLLM because it is fast.”

A strong answer says:

> “I would choose the runtime based on traffic shape, prefix reuse, hardware target, and tail-latency SLO. I would tune scheduler policy with trace replay, not just synthetic throughput, because average tokens/sec can improve while P99 TTFT fails.”

---

## 6.17 Anti-Patterns

Avoid these production mistakes:

1. **Treating training and serving as the same workload.** Training optimizes long-window throughput; serving optimizes latency distribution under dynamic arrivals.
2. **Reporting average latency only.** P99 is where production pain lives.
3. **Maximizing utilization without SLO headroom.** High utilization can create a queueing cliff.
4. **Ignoring useful tokens.** Padded tokens can inflate training throughput metrics.
5. **Blaming kernels before checking data stalls.** Empty GPU gaps often come from the input pipeline.
6. **Using fixed batch intuition for serving.** Interactive serving requires iteration-level scheduling.
7. **Forgetting optimizer state.** Training memory is much larger than inference memory.
8. **Underestimating checkpoint cost.** Reliability has a throughput price.
9. **Universalizing framework benchmark claims.** Runtime performance depends on model, hardware, traffic distribution, and configuration.
10. **Forgetting economics.** A serving system that is fast but too expensive per token may still fail.

---

## 6.18 Key Takeaways

1. Training, offline inference, batch API serving, and interactive serving have different objective functions.
2. Training bottlenecks include compute, activation memory, optimizer state, data loading, communication, and checkpoint I/O.
3. Dense Transformer training FLOPs are often estimated as `6 × parameters × training tokens`, but this is an approximation.
4. LLM inference has two distinct phases: prefill and decode.
5. TTFT is primarily shaped by queueing and prefill; TPOT is shaped by decode efficiency, scheduler policy, HBM/KV pressure, and batch behavior.
6. Percentile latency matters more than average latency for interactive serving.
7. Continuous batching improves decode slot utilization by admitting work at iteration boundaries.
8. Chunked prefill protects active decode requests from long-prompt interference.
9. KV cache is a first-class serving resource because it scales with active sequences and context length.
10. Principal-level workload analysis begins with objective, bottleneck, metric, and validation experiment — not with a tool name.

---

## 6.19 Review Questions

1. Explain the difference between batch training, offline inference, batch API serving, and interactive serving.
2. Why is useful tokens/sec/GPU better than padded tokens/sec/GPU for training?
3. Derive the dense training FLOP approximation `6 × parameters × tokens`. What does it omit?
4. What memory categories make training more memory-intensive than inference?
5. Why can activation checkpointing reduce memory but increase step time?
6. What evidence would show that a training job is data-loader-bound?
7. What evidence would show that distributed training is communication-bound?
8. Define TTFT and TPOT. Which serving phase dominates each?
9. Why is decode often memory-bandwidth-sensitive at small batch?
10. Explain static batching, dynamic batching, and continuous batching.
11. Why can a long prefill harm active decode traffic?
12. What problem does chunked prefill solve?
13. What new complexity does disaggregated prefill/decode introduce?
14. Why is P99 latency more important than average latency for interactive serving?
15. What is the relationship between KV cache, context length, and serving concurrency?
16. How would you design an experiment to choose a safe KV admission threshold?
17. Why might maximizing GPU utilization hurt user experience?
18. Compare vLLM, TensorRT-LLM, and SGLang at a high level. What would you validate before choosing one?
19. In an interview, how would you explain the difference between a training bottleneck and a serving bottleneck?
20. Build a diagnostic plan for this symptom: TTFT P99 doubled after enabling long-context requests.

---

## 6.20 Production Notes for This Chapter

### Source files used

- `ch06_production_audit.md`
- `ch06_figure_integration_plan.md`
- `ch06_technical_validation.md`
- `index.html`
- `diagrams_batch1.html`
- `diagrams_batch2.html`
- `diagrams_batch3.html`
- `diagram_05_parallelism_topology.html`
- `diagram_08_observability_stack.html`
- `ch00_front_matter-combined.pdf`
- `ch11_kv_cache_complete-combined.pdf`

### External validation references to preserve in editorial notes

- vLLM official documentation and GitHub for PagedAttention, continuous batching, chunked prefill, prefix caching, and disaggregated prefill.
- NVIDIA TensorRT-LLM documentation for in-flight batching, paged KV cache, chunked context/prefill, scheduler, and parallelism features.
- SGLang official documentation/GitHub for RadixAttention, continuous batching, prefill-decode disaggregation, speculative decoding, and chunked prefill.
- PyTorch official documentation for `DistributedDataParallel`, `DataLoader`, pinned memory, and performance tuning guidance.

### Production fixes before final print pass

- Replace all placeholder figure blocks with final SVG/HTML figure assets.
- Confirm the final `index.html` links to `chapters/ch06_training_inference_workloads.html`.
- Ensure chapter numbers and next/previous navigation match the final book ordering.
- Re-check framework feature wording against the exact release versions used in the final 2026 edition.
- Keep all exact benchmark values marked `[REPRESENTATIVE]` or `[ENV-SPECIFIC]` unless measured or cited.
- Do not over-expand Ch06 into Ch10 or Ch11; keep those deep dives separate.

---

## 6.21 Bridge to Chapter 7

Chapter 6 showed why training and serving workloads stress AI infrastructure differently.

Chapter 7 moves one level lower into the execution unit behind many of those workload behaviors:

```text
What actually happens inside a GPU kernel, and how do memory access, occupancy, Tensor Cores, fusion, and profiling determine performance?
```

The next chapter connects workload symptoms to kernel-level causes and teaches how to decide when kernel work is the right optimization move.
