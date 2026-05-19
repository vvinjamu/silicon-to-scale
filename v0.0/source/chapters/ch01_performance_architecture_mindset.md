# Chapter 1 — The AI/ML Performance Architecture Mindset

> “Performance engineering is not about making one function faster. It is about understanding the whole system — from transistors to tokens — and knowing exactly where the bottleneck lives before you touch anything.”

---

## Chapter Overview

This chapter is the analytical foundation for the rest of this book.

The central idea is simple:

> Every AI/ML performance problem is a question of **compute**, **memory movement**, **communication**, or **overhead**.

A principal AI/ML performance architect does not start by tuning random kernels, changing batch sizes, or adding GPUs. A principal architect first classifies the workload, estimates the bottleneck, chooses the right measurement, and only then proposes an optimization.

This chapter teaches that mindset.

By the end of this chapter, you should be able to:

- Explain why AI performance is not only about peak TFLOPS.
- Use arithmetic intensity to classify workloads.
- Apply the roofline model to reason about compute-bound and memory-bound behavior.
- Calculate the H100 dense BF16 ridge point from first principles.
- Distinguish MFU from HFU.
- Explain why a system can have busy GPUs but poor useful model progress.
- Use a hypothesis-driven profiling loop.
- Speak about performance like a principal architect in interviews and design reviews.

---

## 1.0 The One-Sentence Model

The entire book can be summarized as:

> **Every optimization decision in AI/ML infrastructure is a question of arithmetic intensity, memory bandwidth, communication volume, and system overhead.**

The first three quantities explain most of the physics. The fourth explains most of the production pain.

A workload may be slow because:

1. It does not have enough math per byte moved.
2. It cannot feed the GPU fast enough from memory.
3. It spends too much time communicating across devices or nodes.
4. It has orchestration, scheduling, kernel-launch, I/O, or framework overhead.

The skill is not memorizing every tool. The skill is knowing **which question to ask first**.

---

## 1.1 The Three Quantities of AI Performance

Before looking at a profiler trace, ask:

1. **How much math is required?**
2. **How much data must move locally?**
3. **How much data must move across devices, nodes, racks, or storage?**

These map to three core quantities:

| Quantity | Meaning | Common Bottleneck |
|---|---|---|
| Arithmetic intensity | FLOPs per byte moved | Low reuse, memory-bound kernels |
| Memory bandwidth | Bytes per second through HBM, cache, PCIe, or storage | HBM-bound decode, LayerNorm, embeddings |
| Communication volume | Bytes exchanged across GPUs or nodes | AllReduce, tensor parallel collectives, pipeline transfers |

The most common mistake is to optimize the easiest layer to see rather than the layer that limits the system.

A slow kernel is not always a kernel problem. A slow training step is not always a GPU problem. A slow inference request is not always a model problem.

### Figure Placeholder — Fig 1.1

```markdown
![Fig 1.1 — The Three Quantities of AI Performance](../assets/diagrams/svg/ch01_fig_1_1_three_quantities.svg)

**Fig 1.1 — The Three Quantities of AI Performance.** Every AI/ML performance problem can be reduced to arithmetic intensity, memory bandwidth, and communication volume. Compute tells you how much math exists, memory bandwidth tells you how fast data can move locally, and communication volume tells you how much data must cross device, node, rack, or cluster boundaries.
```

**Figure intro:**  
Before looking at any profiler trace, a principal performance engineer asks a simpler question: what quantity is most likely limiting this workload? For a single kernel, the answer is often arithmetic intensity or memory bandwidth. For distributed training and serving, communication volume becomes just as important.

**Figure explanation:**  
A compute-bound GEMM needs better Tensor Core utilization. A memory-bound decode step needs less HBM traffic or better cache behavior. A communication-bound AllReduce needs overlap, topology awareness, or reduced synchronization volume. Chapter 1 teaches how to classify these cases before reaching for tools.

> **Key Takeaway:** Do not begin with “How do I make this faster?” Begin with “Which quantity is limiting the system: compute, memory movement, communication, or overhead?”

---

## 1.2 The Seven-Layer AI/ML Performance Stack

AI performance is not a single-layer problem.

A model may run slowly because of:

- Tensor Core underutilization
- HBM bandwidth limits
- Poor memory access patterns
- Compiler graph breaks
- Too many small kernels
- Poor batching
- KV cache pressure
- NCCL communication bottlenecks
- Data-loader stalls
- Scheduler gaps
- Thermal throttling
- Network congestion
- Storage checkpoint stalls
- Fleet-level queueing

A principal architect must reason across the entire stack.

### Figure Placeholder — Fig 1.2

```markdown
![Fig 1.2 — The Seven-Layer AI/ML Performance Stack](../assets/diagrams/svg/ch01_fig_1_2_performance_stack.svg)

**Fig 1.2 — The Seven-Layer AI/ML Performance Stack.** AI/ML performance problems can originate at any layer: silicon, memory, kernel, compiler/runtime, model architecture, distributed system, or fleet operations. Principal-level performance engineering means reasoning across all seven layers instead of optimizing only the nearest function.
```

### The Seven Layers

| Layer | Example Questions | Metrics / Tools |
|---|---|---|
| 1. Silicon / accelerator | Are Tensor Cores active? Is the GPU throttling? | SM utilization, clocks, power, temperature, DCGM |
| 2. Memory hierarchy | Are we limited by HBM, L2, cache misses, or PCIe? | HBM bandwidth, cache hit rate, Nsight Compute |
| 3. Kernel execution | Are kernels coalesced, tiled, fused, and occupancy-efficient? | Occupancy, stalls, Tensor Core utilization |
| 4. Compiler / runtime | Are graph breaks or launch gaps hurting throughput? | torch.compile logs, PyTorch profiler, Nsight Systems |
| 5. Model architecture | Is attention, FFN, KV cache, or MoE routing the limiter? | FLOPs, activation size, KV bytes, token latency |
| 6. Distributed system | Is the bottleneck AllReduce, All-to-All, pipeline bubble, or topology? | NCCL busbw, overlap, IB counters |
| 7. Fleet / product / cost | Are GPUs busy doing useful work? Is cost/token acceptable? | MFU, queue time, P99 latency, $/token |

The stack is not a checklist to memorize. It is a routing system for diagnosis.

If GPU utilization is low, do not immediately tune CUDA. First ask whether the GPU is idle because of data loading, communication, scheduling, or queueing.

If a kernel is slow, do not immediately rewrite it. First ask whether it is memory-bound, compute-bound, or launch-overhead-bound.

> **Key Takeaway:** The best optimization is often not located where the symptom appears. Always map the symptom to a layer before choosing a fix.

---

## 1.3 Roofline Analysis: The Universal Performance Model

The roofline model is the first calculation every AI/ML performance engineer should learn.

It answers:

> Given the operation’s arithmetic intensity and the hardware’s peak compute and memory bandwidth, what is the maximum performance we should expect?

The roofline model is:

```text
Achievable Performance <= min(Peak_FLOPS, Arithmetic_Intensity × Peak_Memory_Bandwidth)
```

Where:

```text
Arithmetic Intensity = FLOPs / Bytes Moved
```

For single-GPU analysis, “bytes moved” usually means HBM bytes. For distributed systems, the same idea can be applied to NVLink, PCIe, InfiniBand, storage, or any data-movement tier.

### Confidence Label

[ESTIMATED] The roofline model estimates an upper bound. It is not a guarantee. Real performance also depends on instruction mix, kernel implementation, occupancy, cache behavior, launch overhead, compiler behavior, and scheduling.

---

## 1.4 The H100 Dense BF16 Ridge Point

For Chapter 1, we use dense, non-sparse BF16 Tensor Core peak throughput.

[SHIPPED] H100 SXM5 dense BF16 Tensor Core peak:

```text
Peak Compute = 989.4 TFLOP/s
```

[SHIPPED] H100 SXM5 HBM bandwidth:

```text
Peak HBM Bandwidth = 3.35 TB/s
```

[DERIVED FROM SHIPPED] Ridge point:

```text
Ridge Point = Peak Compute / Peak Memory Bandwidth

Ridge Point = 989.4 TFLOP/s / 3.35 TB/s
            = 295.34 FLOP/byte
            ≈ 295 FLOP/byte
```

This means:

- Operations below ~295 FLOP/byte are usually memory-bandwidth-limited.
- Operations above ~295 FLOP/byte may become compute-limited if the kernel uses the hardware efficiently.

### Dense vs Sparse BF16

Be careful with H100 BF16 numbers.

NVIDIA also lists a much higher BF16 Tensor Core number when structured sparsity is used. That value is approximately 1,979 TFLOPS. This book uses **dense non-sparse BF16** for standard roofline calculations unless explicitly discussing sparsity.

| H100 SXM5 Mode | Approximate BF16 Peak | Use in This Chapter |
|---|---:|---|
| Dense / non-sparse BF16 | 989.4 TFLOPS | Yes |
| BF16 with structured sparsity | ~1,979 TFLOPS | Only when discussing sparsity |

If you accidentally use sparse peak compute in a dense workload roofline, the ridge point changes:

```text
Sparse ridge point ≈ 1,979 / 3.35 ≈ 591 FLOP/byte
```

That would give the reader the wrong intuition for dense model workloads.

---

## Mental Math Checkpoint 1

**Question:** H100 SXM5 dense BF16 peak is 989.4 TFLOPS and HBM bandwidth is 3.35 TB/s. What is the ridge point?

**Answer:**

```text
989.4 / 3.35 = 295.34 FLOP/byte ≈ 295 FLOP/byte
```

**Interpretation:**  
An operation with arithmetic intensity much lower than 295 FLOP/byte is likely memory-bound on H100. An operation much higher than 295 FLOP/byte has enough data reuse to become compute-bound.

---

### Figure Placeholder — Fig 1.3

```markdown
![Fig 1.3 — Roofline Model for H100 SXM5](../assets/diagrams/png_300dpi/ch01_fig_1_3_h100_roofline.png)

**Fig 1.3 — Roofline Model for H100 SXM5.** The roofline model separates memory-bound operations from compute-bound operations. The ridge point is the arithmetic intensity at which memory bandwidth is no longer the limiter and peak compute becomes the roof.
```

**Figure intro:**  
The roofline model gives a visual answer to whether more compute will help. If the operation sits below the memory-bandwidth slope, it is limited by data movement. If it reaches the flat compute roof, further memory optimization may not help as much as improving compute utilization.

**Figure explanation:**  
Operations to the left of the ridge point are memory-bound. Operations to the right have enough reuse to become compute-bound. This does not mean every operation to the right automatically reaches peak TFLOPS. It means memory bandwidth is no longer the first-order limiter.

> **Key Takeaway:** The roofline model tells you whether more FLOPs will help. If the workload is below the ridge point, adding compute will not fix the bottleneck.

---

## Table 1.1 — H100 Roofline Quick Reference

| Quantity | Value | Confidence Label | Why It Matters |
|---|---:|---|---|
| H100 SXM5 dense BF16 peak | 989.4 TFLOPS | [SHIPPED] | Compute roof for dense BF16 |
| H100 SXM5 HBM bandwidth | 3.35 TB/s | [SHIPPED] | Memory bandwidth roof |
| Dense BF16 ridge point | ~295 FLOP/byte | [DERIVED FROM SHIPPED] | Separates memory-bound from compute-bound |
| Decode attention example | Very low arithmetic intensity | [REPRESENTATIVE] | Often HBM/KV-cache limited |
| LayerNorm / RMSNorm example | Low arithmetic intensity | [REPRESENTATIVE] | Often memory-bandwidth bound |
| Small-batch GEMM | Moderate arithmetic intensity | [REPRESENTATIVE] | May underutilize Tensor Cores |
| Large-batch GEMM | High arithmetic intensity | [REPRESENTATIVE] | Can become compute-bound |

The exact arithmetic intensity depends on tensor shape, datatype, implementation, reuse, cache behavior, and fusion. The table is not meant to replace measurement. It is meant to build intuition before measurement.

---

## 1.5 Arithmetic Intensity: The Master Classifier

Arithmetic intensity is:

```text
AI = FLOPs / Bytes Moved
```

High arithmetic intensity means the operation performs a lot of math for every byte it moves. Low arithmetic intensity means the operation moves a lot of data for relatively little math.

### Example 1 — Low Arithmetic Intensity

A normalization kernel might read a tensor, compute a few statistics, scale values, and write the result.

It performs some math, but it also streams through memory. If each element is touched once or twice with little reuse, the operation is usually memory-bound.

### Example 2 — High Arithmetic Intensity

A large GEMM reuses matrix tiles many times. A tile loaded into shared memory or cache can participate in many multiply-accumulate operations.

That reuse increases arithmetic intensity. High arithmetic intensity is why GEMM can approach Tensor Core peak throughput.

### Example 3 — Decode Attention

Decode attention often reads a growing KV cache for each generated token. The amount of memory read grows with sequence length, while the amount of useful new computation per token is limited.

This is why LLM decode can be memory-bandwidth-bound even on very powerful GPUs.

[REPRESENTATIVE] Decode bottlenecks vary by implementation, context length, model shape, KV precision, batching, and attention kernel.

---

## 1.6 The Three Performance Regimes

Once you estimate arithmetic intensity, classify the workload.

### Figure Placeholder — Fig 1.4

```markdown
![Fig 1.4 — The Three Performance Regimes](../assets/diagrams/svg/ch01_fig_1_4_three_regimes.svg)

**Fig 1.4 — The Three Performance Regimes.** A workload can be compute-bound, memory-bound, or communication/overhead-bound. The correct optimization depends on which regime dominates.
```

**Figure intro:**  
Not every slow workload should be optimized the same way. A compute-bound problem, a memory-bound problem, and a communication-bound problem require different fixes.

**Figure explanation:**  
The purpose of classification is to prevent wrong-regime optimization. Optimizing a GEMM kernel will not fix a KV-cache memory bottleneck. Buying faster GPUs will not fix a saturated AllReduce. Increasing average throughput will not fix a P99 queueing issue.

> **Key Takeaway:** Classify first, optimize second. Wrong-regime optimization creates impressive local improvements that do not move system KPIs.

---

## Table 1.2 — Optimization Regime Decision Table

| Regime | Common Symptom | Metric to Check | First Tools | Likely Fix |
|---|---|---|---|---|
| Compute-bound | High compute utilization but below peak | Achieved TFLOPS, Tensor Core utilization | Nsight Compute, rocprof-compute | Improve tiling, fusion, Tensor Core usage |
| Memory-bound | High HBM traffic, low FLOP utilization | HBM bandwidth, cache hit rate, arithmetic intensity | Nsight Compute, roofline | Reduce memory traffic, improve reuse, fuse ops |
| Communication-bound | Scaling drops with more GPUs | AllReduce time, busbw, overlap | Nsight Systems, NCCL tests, IB counters | Overlap communication, tune topology, reduce synchronization |
| Overhead-bound | GPU gaps, many short kernels | Kernel gaps, launch overhead, queue time | Nsight Systems, PyTorch profiler | CUDA Graphs, batching, scheduling fixes |
| I/O-bound | GPU waits for input data | DataLoader wait, storage throughput | PyTorch profiler, storage metrics | Prefetch, cache, parallelize data pipeline |

This table is the operational form of the roofline mindset. It maps symptoms to metrics, tools, and likely fixes.

---

## Table 1.3 — Wrong Fix vs Right First Question

| Symptom | Common Wrong Fix | Right First Question |
|---|---|---|
| Decode latency is high | Optimize GEMM kernel | Is decode memory-bandwidth-bound due to KV cache reads? |
| Training MFU is low | Tune one CUDA kernel | Is the GPU idle due to communication, data loading, or pipeline bubbles? |
| More GPUs do not improve throughput | Add faster GPUs | What percentage of step time is communication? |
| HBM OOM during serving | Reduce batch size only | Can KV cache be paged, quantized, shared, or tiered? |
| Kernel is slow | Rewrite in CUDA immediately | Is it memory-bound, compute-bound, or launch-overhead-bound? |
| P99 latency is high | Increase average throughput | Which queue or stage dominates tail latency? |

> **Key Takeaway:** Every symptom deserves a diagnostic question before an optimization proposal.

---

## 1.7 MFU and HFU: Useful Work vs Busy Hardware

GPU utilization alone is not enough.

A system can have high GPU activity and still make poor model progress.

That is why large-scale AI training discussions often use two related metrics:

- **MFU** — Model FLOPs Utilization
- **HFU** — Hardware FLOPs Utilization

## Table 1.4 — MFU vs HFU

| Metric | Measures | Includes Recomputation? | Best Used For | Common Misread |
|---|---|---|---|---|
| MFU | Useful model FLOPs / peak hardware FLOPs | Usually no | Training efficiency and scaling quality | Can look low even if hardware is busy with overhead |
| HFU | Actual hardware FLOPs / peak hardware FLOPs | Yes | Hardware saturation and kernel activity | Can look good while useful model progress is poor |

### MFU

MFU asks:

> How much useful model progress did the system achieve relative to theoretical peak hardware capability?

A simplified definition is:

```text
MFU = Useful Model FLOPs per second / Theoretical Peak FLOPs per second
```

### HFU

HFU asks:

> How much hardware work was executed relative to theoretical peak hardware capability?

A simplified definition is:

```text
HFU = Actual Hardware FLOPs per second / Theoretical Peak FLOPs per second
```

HFU can be higher than MFU when the system performs recomputation, rematerialization, or extra implementation-specific work that keeps the GPU busy without increasing useful model progress.

[ENV-SPECIFIC] MFU and HFU depend on model size, sequence length, precision, hardware, framework, compiler, parallelism strategy, recomputation settings, and measurement methodology.

---

## Mental Math Checkpoint 2

Suppose an 8×H100 node has:

```text
Dense BF16 peak per GPU = 989.4 TFLOPS
Number of GPUs = 8
Total peak = 8 × 989.4 = 7,915.2 TFLOPS
```

If the training job achieves 3,500 useful model TFLOPS:

```text
MFU = 3,500 / 7,915.2 = 0.442 = 44.2%
```

Interpretation:

- This may be reasonable for a large distributed training job.
- But the number alone is not a diagnosis.
- You still need to determine whether the gap is due to memory, communication, pipeline bubbles, recomputation, data loading, or kernel inefficiency.

[ENV-SPECIFIC] Sustained MFU in the 40–60% range can indicate a reasonably efficient large-scale training system in many contexts, while higher values may reflect excellent tuning. These ranges are not universal.

---

### Figure Placeholder — Fig 1.5

```markdown
![Fig 1.5 — From Tokens to Utilization: MFU and HFU in the Training Loop](../assets/diagrams/svg/ch01_fig_1_5_mfu_hfu_system_view.svg)

**Fig 1.5 — From Tokens to Utilization: MFU and HFU in the Training Loop.** MFU converts model progress into useful FLOPs per second. HFU measures how much hardware work was executed. The gap between the two helps identify recomputation, overhead, and inefficiency.
```

**Figure intro:**  
A training job is a repeated loop: forward pass, backward pass, gradient communication, optimizer update, checkpointing, and input pipeline. MFU and HFU summarize how effectively that loop uses hardware.

**Figure explanation:**  
When MFU is low, do not automatically assume a kernel is bad. The system may be losing time to AllReduce, pipeline bubbles, data loading, checkpointing, stragglers, or framework overhead.

> **Key Takeaway:** HFU asks “Are the GPUs busy?” MFU asks “Are the GPUs busy doing useful model work?”

---

## 1.8 FLOPs per Token: Practical Approximations

For dense decoder-only transformers, two common approximations are useful:

[ESTIMATED]

```text
Inference FLOPs/token ≈ 2N
Training FLOPs/token ≈ 6N
```

Where:

```text
N = non-embedding parameter count
```

### Why Inference Is Approximately 2N

A dense forward pass through model weights performs roughly one multiply and one add per parameter.

That gives approximately:

```text
2 FLOPs per parameter per token
```

So:

```text
Inference FLOPs/token ≈ 2N
```

### Why Training Is Approximately 6N

Training includes:

1. Forward pass
2. Backward pass for activations
3. Backward pass for weights

A rough mental model is:

```text
Training FLOPs/token ≈ 3 × forward pass
                          ≈ 3 × 2N
                          ≈ 6N
```

### Important Assumptions

These approximations are useful for back-of-the-envelope reasoning, but exact FLOPs depend on:

- Architecture
- Context length
- Attention implementation
- Vocabulary projection
- Embedding treatment
- Activation recomputation
- MoE sparsity
- Precision
- Kernel implementation

Use these approximations to start a discussion, not to end one.

---

## 1.9 Measurement Discipline: Profile After You Have a Hypothesis

A profiler is not a fishing tool.

A profiler should confirm or reject a hypothesis.

Bad workflow:

```text
Open profiler → stare at timeline → chase the biggest bar → make random change
```

Better workflow:

```text
Estimate → classify → hypothesize → measure → change one variable → validate → document
```

### Figure Placeholder — Fig 1.6

```markdown
![Fig 1.6 — The Performance Engineer’s Hypothesis Loop](../assets/diagrams/svg/ch01_fig_1_6_profiling_hypothesis_loop.svg)

**Fig 1.6 — The Performance Engineer’s Hypothesis Loop.** Production optimization should follow a loop: classify the workload, form a hypothesis, measure the right metric, change one variable, validate the outcome, and decide whether to continue or stop.
```

**Figure intro:**  
Profilers are powerful, but they should not be used as random search tools. A profiler should confirm or reject a hypothesis. The hypothesis loop turns optimization into a reproducible engineering process.

**Figure explanation:**  
This loop creates a written trail: what was expected, what was measured, what changed, and what moved. That trail matters in architecture reviews, incident reviews, hardware roadmap discussions, and interviews.

> **Key Takeaway:** The profiler confirms the hypothesis; it should not be the only source of the hypothesis.

---

## Table 1.5 — Metrics by Stack Layer

| Layer | Symptom | Metric | Tool |
|---|---|---|---|
| Silicon / GPU | Low math throughput | SM utilization, Tensor Core utilization, clocks | DCGM, Nsight Compute |
| Memory | Low FLOPs, high traffic | HBM bandwidth, cache hit rate | Nsight Compute |
| Kernel | Slow operation | Kernel duration, occupancy, stalls | Nsight Compute, rocprof-compute |
| Runtime / compiler | Many tiny kernels or graph breaks | Launch gaps, graph breaks | Nsight Systems, PyTorch profiler |
| Distributed system | Poor scaling | AllReduce time, busbw, overlap | NCCL tests, Nsight Systems |
| Storage / input | GPU idle between steps | Data wait, loader time | PyTorch profiler, storage metrics |
| Fleet | Poor utilization or cost | MFU, queue time, P99, $/token | Prometheus, Grafana, cost model |

Use the metric that matches the layer. A kernel profiler cannot fully explain a queueing, communication, or fleet-utilization problem.

---

## 1.10 Communication Volume: The Cluster-Scale Extension

Roofline analysis teaches single-device reasoning. But modern AI systems often run across many GPUs.

At scale, performance may be limited by:

- Gradient AllReduce
- Tensor-parallel collectives
- Pipeline stage transfers
- Expert-parallel All-to-All
- Checkpoint writes
- KV cache transfers
- Scheduler delays
- Cross-rack network topology
- Stragglers and fail-slow nodes

[REPRESENTATIVE] A workload can be highly optimized on one GPU and still scale poorly across hundreds or thousands of GPUs because communication volume grows, synchronization becomes more expensive, and idle time accumulates.

The principal-level question becomes:

> How much useful compute survives communication, synchronization, and orchestration overhead?

This is why Chapters 10, 14, 15, 16, and 17 build directly on Chapter 1.

---

## 1.11 How to Explain This in a Principal Interview

In a principal-level interview, do not jump straight into tools.

Start with classification.

A strong answer sounds like this:

> Before proposing an optimization, I classify the workload. I estimate arithmetic intensity, compare it to the hardware ridge point, and determine whether the workload is compute-bound, memory-bound, communication-bound, or overhead-bound. That tells me whether to optimize Tensor Core utilization, reduce HBM traffic, improve communication overlap, or eliminate scheduling gaps.

### Scenario 1 — GEMM Is 40% Slower Than Expected

Weak answer:

> I would profile and optimize the kernel.

Better answer:

> I would first estimate whether the GEMM shape has enough arithmetic intensity to be compute-bound. If yes, I would inspect Tensor Core utilization, tile shape, occupancy, memory layout, and whether the framework dispatched the expected kernel. If not, I would check memory movement, layout conversion, and launch overhead.

### Scenario 2 — Decode Throughput Is Poor

Weak answer:

> I would use a faster GPU or optimize matmul.

Better answer:

> Decode is often memory-bandwidth-sensitive because each generated token reads from the KV cache. I would check HBM bandwidth, KV precision, cache layout, batch scheduling, prefix reuse, and whether the serving system is limited by memory traffic rather than compute.

### Scenario 3 — 256-GPU Training MFU Is Low

Weak answer:

> I would tune CUDA kernels.

Better answer:

> At 256 GPUs, I would first break down step time into compute, communication, pipeline bubble, data loading, and checkpoint overhead. Low MFU could be caused by AllReduce serialization, poor overlap, stragglers, pipeline imbalance, or input stalls. I would use Nsight Systems, NCCL tests, framework timers, and fleet telemetry before touching individual kernels.

---

## 1.12 Chapter Cheat Sheet

### Core Formulas

```text
Arithmetic Intensity = FLOPs / Bytes Moved
```

```text
Achievable Performance <= min(Peak_FLOPS, AI × Peak_Memory_Bandwidth)
```

```text
Ridge Point = Peak_FLOPS / Peak_Memory_Bandwidth
```

```text
MFU = Useful Model FLOPs per second / Peak Hardware FLOPs per second
```

```text
HFU = Actual Hardware FLOPs per second / Peak Hardware FLOPs per second
```

```text
Inference FLOPs/token ≈ 2N
Training FLOPs/token ≈ 6N
```

### H100 Dense BF16 Numbers

[SHIPPED]

```text
H100 SXM5 dense BF16 peak ≈ 989.4 TFLOPS
H100 SXM5 HBM bandwidth ≈ 3.35 TB/s
```

[DERIVED FROM SHIPPED]

```text
H100 dense BF16 ridge point ≈ 295 FLOP/byte
```

### The Five Questions

Before optimizing, ask:

1. Is it compute-bound?
2. Is it memory-bound?
3. Is it communication-bound?
4. Is it overhead-bound?
5. Which metric proves that?

---

## 1.13 Key Takeaways

1. AI/ML performance engineering begins with classification, not tuning.
2. Arithmetic intensity tells you how much compute exists per byte moved.
3. The roofline model tells you whether memory bandwidth or compute peak is the likely limiter.
4. H100 SXM5 dense BF16 ridge point is approximately 295 FLOP/byte.
5. Dense and sparse BF16 peaks must not be mixed in the same roofline analysis.
6. MFU measures useful model progress; HFU measures actual hardware work.
7. Low MFU is a system symptom, not automatically a kernel problem.
8. Communication volume becomes a first-order constraint at distributed scale.
9. Profilers should confirm hypotheses, not replace them.
10. Principal-level performance work connects measurements to architecture decisions.

---

## 1.14 Review Questions

### Conceptual

1. What is arithmetic intensity, and why does it matter?
2. Why is peak TFLOPS alone not enough to predict model performance?
3. What is the ridge point in roofline analysis?
4. Why should dense and sparse BF16 peaks not be mixed in the same calculation?
5. What is the difference between MFU and HFU?
6. Why can high HFU coexist with low MFU?
7. What are three reasons a GPU may be idle even when the model has enough compute work?
8. Why can a workload scale poorly even if single-GPU kernels are optimized?

### Calculation

1. H100 SXM5 dense BF16 peak is 989.4 TFLOPS and HBM bandwidth is 3.35 TB/s. Calculate the ridge point.
2. If an operation has arithmetic intensity of 50 FLOP/byte on H100, is it likely memory-bound or compute-bound?
3. If an operation has arithmetic intensity of 400 FLOP/byte on H100, what else must be true for it to approach peak compute?
4. An 8×H100 node has a dense BF16 peak of 7,915.2 TFLOPS. If useful model throughput is 4,000 TFLOPS, what is MFU?
5. A dense 70B model uses the approximation `inference FLOPs/token ≈ 2N`. Estimate FLOPs per generated token.

### Principal-Level Interview Practice

1. You observe low MFU on a 512-GPU training job. Walk through your first five diagnostic questions.
2. A team says decode latency is high and proposes optimizing GEMM kernels. How would you respond?
3. A model is memory-bound according to roofline analysis. Give three possible optimization strategies.
4. A training workload becomes slower per GPU as GPU count increases. What measurements would you collect?
5. Explain the roofline model to a senior engineering manager in two minutes.

---

## 1.15 Production Notes for This Chapter

### Figure Assets Needed

| Figure | Status |
|---|---|
| Fig 1.1 — Three Quantities of AI Performance | Must be created |
| Fig 1.2 — Seven-Layer AI/ML Performance Stack | Must be created |
| Fig 1.3 — H100 Roofline Model | Existing asset; needs print export |
| Fig 1.4 — Three Performance Regimes | Must be created |
| Fig 1.5 — MFU/HFU System View | Must be created |
| Fig 1.6 — Profiling Hypothesis Loop | Must be created |

### Table Assets Included

| Table | Status |
|---|---|
| Table 1.1 — H100 Roofline Quick Reference | Included |
| Table 1.2 — Optimization Regime Decision Table | Included |
| Table 1.3 — Wrong Fix vs Right First Question | Included |
| Table 1.4 — MFU vs HFU | Included |
| Table 1.5 — Metrics by Stack Layer | Included |

### Confidence Labels Used

| Label | Use in Chapter 1 |
|---|---|
| [SHIPPED] | Vendor-published hardware values |
| [DERIVED FROM SHIPPED] | Ridge point calculations from vendor values |
| [ESTIMATED] | Roofline ceilings and FLOPs/token approximations |
| [REPRESENTATIVE] | Example workload behaviors |
| [ENV-SPECIFIC] | MFU/HFU ranges and measured performance claims |

### Source Notes to Add in Final Book

Use official/public sources for:

- H100 SXM5 dense BF16 and HBM bandwidth
- H100 sparse BF16 note
- MI300X BF16 and HBM bandwidth if included in Chapter 1 or Appendix A
- Original Roofline model
- MFU/HFU definitions
- Transformer FLOPs/token approximations

---

## 1.16 Bridge to Chapter 2

Chapter 1 gave the performance lens: compute, memory, communication, and overhead.

Chapter 2 applies that lens to the transformer itself.

The next question is:

> What exactly happens inside a transformer block, and why do GEMM, attention, normalization, activation functions, and KV cache behavior create such different performance regimes?

That is where the AI/ML performance architect stops treating the model as a black box and starts reasoning from shapes, FLOPs, memory traffic, and data reuse.
