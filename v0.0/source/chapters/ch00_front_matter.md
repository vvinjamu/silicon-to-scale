# Ch00 — Front Matter

# AI/ML Infrastructure from Silicon to Scale

## The Principal Performance Architect’s Complete Guide to Designing, Optimizing, and Operating AI Systems — from GPU Microarchitecture to Hundred-Thousand-GPU Clusters

**Author:** Venkat Vinjam  
**Role:** AI/ML Performance Architect · AMD · Ex-Intel  
**Edition:** First Edition · 2026  
**Format:** Production v1.0 Source Draft  
**Coverage:** 6 Parts · 18 Chapters · 3 Appendices  
**Hardware Generation:** NVIDIA H100/H200/B100/GB200, AMD MI300X/MI350, Intel Gaudi 3  
**Software Stack:** CUDA · PyTorch 2.x · Triton · vLLM · TensorRT-LLM · Megatron-LM · NCCL/RCCL · SLURM · Kubernetes

---

## Copyright and Edition Note

© 2026 Venkat Vinjam. All rights reserved.

No part of this publication may be reproduced, distributed, stored, transmitted, or used in derivative works without written permission from the author, except for brief quotations used in reviews, commentary, or educational discussion.

This book is provided as a technical and educational reference. Hardware specifications, software APIs, performance numbers, pricing, and cloud infrastructure details change rapidly. Readers should validate implementation-specific decisions against current vendor documentation, measured benchmarks, and their own production environment.

**Edition:** First Edition  
**Production Branch:** `production-v1.0`  
**Intended Outputs:** Online HTML book · Digital PDF · Print-ready edition

---

## Epigraph

> “Performance engineering is not about making one function faster.  
> It is about understanding the whole system — from transistors to tokens —  
> and knowing exactly where the bottleneck lives before you touch anything.”

— Venkat Vinjam

---

# Note on Accuracy and Currency

AI infrastructure evolves faster than most technical books can track.

A GPU specification can change across SKUs. A compiler backend can improve between PyTorch releases. A serving framework can change default behavior in a minor version. A vendor may announce hardware months before broad production availability. A benchmark can be accurate for one cluster and misleading for another.

For that reason, this book uses **confidence labels** for quantitative claims and examples.

## Confidence Label System

| Label | Meaning | How to Use It |
|---|---|---|
| `[SHIPPED]` | Verified shipping hardware or software capability from public vendor documentation or generally available releases | Use for stable hardware specifications, released software features, and known APIs |
| `[ANNOUNCED]` | Vendor-disclosed but not necessarily broadly shipping or independently measured | Treat as directional until verified in production |
| `[DERIVED FROM SHIPPED]` | Calculated from shipped values, such as ridge points derived from peak FLOPs and bandwidth | Trust the method, but verify the input numbers |
| `[ESTIMATED]` | Calculated or inferred from first principles, simplified formulas, or partial information | Use for mental models and planning, not final procurement decisions |
| `[REPRESENTATIVE]` | Illustrative example using plausible values | Useful for learning; not a universal claim |
| `[ENV-SPECIFIC]` | Cluster-, framework-, model-, or workload-dependent | Measure in your environment before making decisions |

## Code and Configuration Labels

Code blocks and configuration examples follow the same philosophy.

A snippet may be:

- Conceptual pseudocode
- A representative command
- A version-specific example
- A production-oriented pattern
- A simplified teaching example

When exact behavior matters, confirm against the current documentation for the framework, accelerator, driver, runtime, scheduler, or cloud provider you are using.

## The Most Important Accuracy Rule

The methods in this book are more stable than the numbers.

Arithmetic intensity, memory bandwidth, communication volume, roofline analysis, MFU, HFU, queueing behavior, and bottleneck classification remain useful across hardware generations.

Specific numbers require periodic refresh.

---

# Preface

## Why This Book Exists

This book was written for a specific engineer at a specific career inflection point.

You may have spent years — perhaps a decade or more — mastering performance engineering in systems you understand deeply: CPU microarchitecture, firmware, Linux, HPC clusters, cloud infrastructure, networking, storage, embedded systems, or application-level optimization.

You know how to profile.  
You know how to reason about bottlenecks.  
You know how to debug difficult systems.  
You understand latency, throughput, utilization, tradeoffs, and measurement discipline.

But the center of gravity in performance engineering has shifted.

The dominant computing problem of the next decade is not only web-scale distributed systems or traditional HPC simulation. It is the training and serving of large AI models on accelerated infrastructure.

The strongest principal-level roles in this space require a rare combination:

- GPU and accelerator architecture depth
- Transformer and LLM performance intuition
- Kernel and compiler awareness
- Distributed training and serving systems knowledge
- Networking and storage reasoning
- Observability and fleet operations discipline
- Business-level cost and TCO communication
- The ability to explain quantitative tradeoffs to hardware, software, product, and leadership teams

This book is designed to help you make that transition.

It assumes you can already think in systems.

It teaches you the AI/ML infrastructure domain.

---

## Who This Book Is For

This book is for engineers who want to become stronger AI/ML infrastructure performance architects.

It is especially useful for:

### 1. Systems Performance Engineers Moving into AI/ML

You already understand low-level performance concepts, but you want to connect them to GPUs, transformers, training, inference, and LLM serving.

You may come from:

- CPU performance
- HPC
- Linux systems
- Embedded systems
- Firmware
- Cloud infrastructure
- Storage
- Networking
- Distributed systems

This book gives you the AI-specific mental models.

### 2. GPU / Accelerator Engineers Expanding Up the Stack

You understand hardware or kernels, but you want to reason about the full stack:

```text
model → framework → compiler/runtime → kernels → collectives → scheduler → cluster → product KPI
```

This book helps you connect local optimizations to system-level outcomes.

### 3. ML Infrastructure Engineers Seeking Deeper Performance Intuition

You may already run vLLM, TensorRT-LLM, SGLang, Kubernetes, SLURM, or distributed training jobs.

This book helps explain why systems behave the way they do:

- Why decode is often memory-bound
- Why KV cache dominates serving capacity
- Why AllReduce can limit scaling
- Why MFU falls as clusters grow
- Why faster GPUs do not always improve throughput
- Why P99 latency and average throughput can disagree

### 4. Senior, Staff, and Principal Engineers Preparing for Interviews

This book is directly useful for principal-level system design and performance interviews.

It helps you answer questions like:

- How would you design an LLM inference serving system?
- How would you debug low MFU in a 1,000-GPU training job?
- How would you choose between H100, H200, MI300X, or future accelerators?
- How would you explain a memory-bound workload?
- How would you reason about cost per token?
- How would you build observability for a GPU fleet?

### 5. Technical Leaders Making AI Infrastructure Decisions

If you need to influence hardware roadmaps, cluster purchases, serving architecture, benchmarking strategy, or production operations, this book gives you the language and quantitative frameworks to communicate those decisions.

---

## What This Book Is Not

This book is intentionally not several things.

### This Is Not an Introduction to Machine Learning

It does not teach what a neural network is from scratch.

It does not walk you through training your first model.

It does not explain gradient descent as if the reader has never seen ML before.

There are excellent resources for that. This is not one of them.

### This Is Not a CUDA Programming Tutorial

You will not finish this book and immediately become a production CUDA kernel engineer.

Chapter 7 gives you the mental model and vocabulary to reason about kernel performance, tiling, memory coalescing, Tensor Core usage, FlashAttention, Triton, and profiling.

But high-performance GPU kernel engineering is a craft that takes focused practice.

This book respects that boundary.

### This Is Not a Paper Survey

Relevant papers are referenced and summarized, but the goal is not to catalog every research contribution.

The goal is to extract the engineering insight.

### This Is Not Vendor Marketing

The book discusses NVIDIA, AMD, Intel, CUDA, ROCm, PyTorch, Triton, vLLM, TensorRT-LLM, NCCL, RCCL, SLURM, Kubernetes, and related systems.

The goal is not to promote one vendor or framework.

The goal is to teach reasoning that survives hardware and software generations.

### This Is Not a Static Benchmark Book

Numbers are useful, but numbers age.

Methods last longer.

The book gives both, but the method is the durable part.

---

# What Makes This Book Different

There are GPU programming guides.  
There are distributed systems books.  
There are ML papers.  
There are blog posts on vLLM, FlashAttention, Megatron-LM, NCCL, and inference serving.  
There are vendor whitepapers.  
There are interview-preparation notes.

What is harder to find is one unified guide that connects all of these into a practical performance architecture framework.

This book is different in five ways.

---

## 1. Silicon to Scale, Literally

The book starts near the hardware:

- SMs
- Tensor Cores
- HBM
- NVLink
- PCIe
- Memory hierarchy
- Roofline analysis

It ends at production-scale architecture:

- Distributed training
- LLM serving
- Cluster scheduling
- Networking
- Storage
- Observability
- Cost per token
- Principal-level system design

The connecting thread is always:

```text
arithmetic intensity → memory bandwidth → communication volume → system efficiency
```

---

## 2. Formulas, Diagrams, and Production Numbers

Every performance claim should connect to a model, measurement, or system behavior.

The book uses:

- Formulas
- Diagrams
- Worked examples
- Capacity-planning calculations
- Hardware reference tables
- Measurement workflows
- Production anti-patterns
- Principal-level interview explanations

The goal is not only to know terminology.

The goal is to reason.

---

## 3. A Confidence Label System

AI hardware and software move quickly.

The confidence-label system makes the book more honest and more useful.

You should always know whether you are reading:

- A shipped hardware specification
- A vendor announcement
- A derived calculation
- A representative example
- An environment-specific observation

This helps the reader avoid overgeneralizing.

---

## 4. Vertical Integration

The book connects ideas across layers.

For example:

- Transformer attention explains why KV cache exists.
- KV cache explains why PagedAttention matters.
- PagedAttention explains why serving capacity depends on memory layout.
- Memory layout explains why HBM capacity and bandwidth shape cost per token.
- Cost per token explains why architecture choices matter to the business.

The reader learns to reason across layers, not only within one layer.

---

## 5. Interview Preparation Built In

The book is technical, but it is also designed to help engineers communicate at principal level.

Each major concept is framed so the reader can explain it in:

- Architecture reviews
- Debug investigations
- Interview system design rounds
- Hardware/software tradeoff discussions
- Leadership updates
- Written design documents

Chapter 18 turns the technical content into principal-level communication patterns, STAR+Impact stories, and system design walkthroughs.

---

# The Central Identity This Book Builds

The strongest AI/ML Principal Performance Architect identity is not:

> “I optimized a CUDA kernel.”

It is:

> “I understand the AI system as a whole — from GPU execution units and HBM bandwidth to cluster topology, serving latency, observability, and cost per token — and I can reason quantitatively about tradeoffs across every layer.”

That is the engineer this book is written for.

That is the engineer this book helps you become.

---

# How to Use This Book

This book can be read front-to-back, and that is the best path for complete mastery.

But most readers come with a specific goal:

- An interview is coming.
- A serving system is slow.
- A training cluster is underutilized.
- A hardware decision must be made.
- A career transition is underway.
- A principal-level architecture story must be built.

Use the path that matches your goal.

---

## Path 1 — The Interview-Focused Reader

### Situation

You have an interview at a company building or operating AI/ML infrastructure at scale.

You may be preparing for roles involving:

- LLM inference performance
- GPU cluster health
- Distributed training
- Performance and efficiency
- AI infrastructure architecture
- Systems debugging
- Principal/staff-level technical leadership

You have limited time.

### Goal

Build enough depth to answer system design and performance questions at senior/staff/principal level.

You do not need to become a CUDA expert in two weeks.

You need to be analytically dangerous.

### Two-Week Plan

#### Week 1 — Analytical Foundation

| Day | Focus | Chapters |
|---|---|---|
| Day 1–2 | Roofline, arithmetic intensity, MFU, bottleneck thinking | Ch01 |
| Day 3 | Transformer performance shapes | Ch02 |
| Day 4 | GPU architecture and memory hierarchy | Ch03A, Ch04 |
| Day 5 | Inference regimes: prefill vs decode | Ch06 |
| Day 6 | KV cache sizing and PagedAttention | Ch11 |
| Day 7 | Review formulas and build cheat sheet | Ch01, Ch06, Ch11 |

#### Week 2 — Scale and System Design

| Day | Focus | Chapters |
|---|---|---|
| Day 8 | Distributed training parallelism | Ch10 |
| Day 9 | Networking and collectives | Ch14 |
| Day 10 | Observability and profiling tools | Ch17 |
| Day 11 | Benchmarking and regression thinking | Ch12 |
| Day 12 | Principal architect communication | Ch18 |
| Day 13 | Mock design: LLM inference serving | Ch18 |
| Day 14 | Mock design: distributed training system | Ch18 |

### Most Important Practice

Run two timed mock interviews:

1. LLM inference serving at scale
2. Distributed LLM training cluster

Set a timer for 45 minutes.

Do not read from the chapter during the mock.

Afterward, compare your answer to the book and fix the gaps.

---

## Path 2 — The Practicing Inference Engineer

### Situation

You are actively building or operating LLM inference systems.

You may be using:

- vLLM
- TensorRT-LLM
- SGLang
- Triton
- Ray Serve
- Kubernetes
- Custom model-serving infrastructure

You understand the basics, but you want stronger performance intuition.

### Goal

Optimize serving systems with quantitative reasoning instead of trial-and-error tuning.

### Recommended Path

| Phase | Chapters | Focus |
|---|---|---|
| Start | Ch01 | Roofline and bottleneck classification |
| Core inference | Ch06 | Prefill, decode, batching, TTFT, TPOT |
| Memory pressure | Ch11 | KV cache, PagedAttention, prefix caching, tiering |
| Precision | Ch08 | Quantization and KV precision |
| Kernels | Ch07 | FlashAttention, fused kernels, profiling |
| Compiler/runtime | Ch09 | torch.compile, Triton, TensorRT |
| Operations | Ch17 | Telemetry, alerts, fail-slow detection |
| Cost | Ch05 | Cost per token and TCO |

### One Project to Complete

Profile a real or representative vLLM deployment during a mixed prefill/decode workload.

Measure:

- TTFT
- TPOT
- GPU utilization
- HBM bandwidth
- KV cache usage
- Queue depth
- Number of preemptions
- Prefix cache hit rate
- P50/P95/P99 latency

Then answer:

> Is the serving system compute-bound, memory-bound, scheduler-bound, or queue-bound?

---

## Path 3 — The Training Infrastructure Architect

### Situation

You are designing, procuring, or operating GPU training clusters.

You need to reason about:

- 4D parallelism
- Network topology
- Storage pipelines
- Fault tolerance
- Cluster utilization
- MFU
- TCO
- Power and thermal limits

### Goal

Design and operate large-scale AI training infrastructure with quantitative confidence.

### Recommended Path

| Phase | Chapters | Focus |
|---|---|---|
| Foundation | Ch01, Ch02 | Performance mindset and transformer compute |
| Hardware | Ch03A, Ch03B, Ch04, Ch05 | GPUs, HBM, power, TCO |
| Training systems | Ch10 | TP, PP, DP, CP, ZeRO, FSDP |
| Measurement | Ch12 | Benchmarking and regression detection |
| Networking | Ch14 | NCCL, InfiniBand, topology, collectives |
| Storage | Ch15 | Data pipelines and checkpointing |
| Cluster management | Ch16 | SLURM, Kubernetes, scheduling, tenancy |
| Observability | Ch17 | Fleet health, telemetry, stragglers |

### One Project to Complete

Build a TCO model for one of these decisions:

- 256× H100 SXM vs 512× A100
- H100 vs H200 for serving
- H100 vs MI300X for memory-heavy inference
- Smaller fast cluster vs larger cheaper cluster
- Cloud rental vs owned cluster

Include:

- Capex
- Opex
- Power
- Cooling
- Network
- Storage
- Expected MFU
- Cost per token or cost per training run

Present the recommendation as if to a VP of Engineering.

---

## Path 4 — The Deep Learner / Full-Stack Transition

### Situation

You are transitioning into AI/ML infrastructure from an adjacent field.

You have time.

You want the full principal-level knowledge base.

### Goal

Build durable, full-stack AI infrastructure performance capability.

### Recommended Path

Read the book in order:

```text
Part I → Part II → Part III → Part IV → Part V → Part VI
```

For each chapter:

1. Read fully.
2. Rewrite the key formulas by hand.
3. Recreate the diagrams in your own notes.
4. Answer the review questions without looking.
5. Run a small benchmark if possible.
6. Add one page to your personal reference sheet.
7. Practice explaining the chapter in five minutes.

After each part:

- Run one mock architecture question.
- Identify what you still cannot explain clearly.
- Return to the relevant chapter and repair the gap.

---

# Quick Reference by Time Budget

| Time Budget | Goal | Priority Chapters |
|---|---|---|
| 2 hours | Emergency interview prep | Ch01, Ch06, Ch11, Ch14, Ch18 |
| 8 hours | Solid interview foundation | Ch01, Ch02, Ch03A, Ch06, Ch10, Ch11, Ch14 |
| Weekend | Thorough interview prep | Ch01, Ch02, Ch06, Ch07, Ch10, Ch11, Ch13, Ch14, Ch18 |
| 2 weeks | Principal interview readiness | Ch01, Ch02, Ch03A, Ch04, Ch06, Ch10, Ch11, Ch14, Ch17, Ch18 |
| 4 weeks | Systematic study | Parts I–IV plus Ch14 and Ch17 |
| 8–12 weeks | Complete mastery pass | All chapters and appendices |
| 12 months | Principal-level transformation | Full curriculum below |

---

# The 12-Month Principal-Level Curriculum

This curriculum is for engineers who want to build the full principal AI/ML infrastructure identity, not only prepare for an interview.

Each two-month block includes:

- Chapters to read deeply
- A hands-on project
- Papers or technical references to study
- A benchmark or measurement task
- A communication deliverable

The recommended pace is 10–15 hours per week.

---

## Months 1–2 — Foundations

### Goal

You can explain roofline analysis, transformer architecture, and GPU memory hierarchy from first principles.

### Read Deeply

- Ch01 — AI/ML Performance Architecture Mindset
- Ch02 — Transformer Architecture Deep Dive

### Hands-On Project

Implement and profile a minimal transformer forward pass in PyTorch.

Measure:

- GEMM time
- Attention time
- Normalization time
- Memory use
- Achieved TFLOPS
- HBM bandwidth if available

Then classify the operations using arithmetic intensity.

### Papers / References

- *Attention Is All You Need*
- Original Roofline paper
- Transformer scaling laws or FLOPs references

### Benchmark

Run a matrix multiplication benchmark and compare achieved TFLOPS to theoretical GPU peak.

### Communication Deliverable

Write a one-page explanation:

> Why transformer performance is mostly a question of GEMM, memory bandwidth, and attention data movement.

---

## Months 3–4 — Hardware Deep Dive

### Goal

You can reason about accelerator selection, HBM bandwidth, memory capacity, and power constraints.

### Read Deeply

- Ch03A — GPU and Accelerator Architecture Fundamentals
- Ch03B — GPU Architecture Roadmap
- Ch04 — Memory Hierarchy and HBM Deep Dive
- Ch05 — Thermal Design, Power Management, and TCO

### Hands-On Project

Measure memory bandwidth and compute throughput on your available GPU.

Compare:

- Vector operation
- GEMM
- Attention-like workload
- Reduction or normalization

Plot each on a roofline chart.

### Papers / References

- NVIDIA H100 architecture documentation
- AMD CDNA / MI300X architecture material
- GPU microbenchmarking references

### Benchmark

Run memory bandwidth and GEMM benchmarks.

### Communication Deliverable

Write a hardware recommendation memo:

> Which accelerator would you choose for a memory-heavy LLM serving workload and why?

---

## Months 5–6 — Optimization Techniques

### Goal

You can identify kernel, compiler, fusion, and quantization opportunities.

### Read Deeply

- Ch07 — GPU Kernels and CUDA Optimization
- Ch08 — Quantization
- Ch09 — Operator Fusion and Compiler Optimization

### Hands-On Project

Profile a transformer block before and after enabling compiler optimization or fused kernels.

Measure:

- Kernel count
- Memory traffic
- Runtime
- Graph breaks
- Achieved throughput

### Papers / References

- FlashAttention
- FlashAttention-2
- Triton documentation
- PyTorch 2.x compile documentation
- Quantization references such as GPTQ/AWQ/SmoothQuant

### Benchmark

Run a forward pass with and without `torch.compile` or an equivalent optimization path.

### Communication Deliverable

Write a short optimization report:

> Which operation was the bottleneck, what changed, and what metric improved?

---

## Months 7–8 — Training and Serving Systems

### Goal

You can reason about distributed training memory, serving capacity, and KV cache economics.

### Read Deeply

- Ch06 — Inference Systems Architecture
- Ch10 — Training Systems Architecture at Scale
- Ch11 — KV Cache: The Heart of LLM Serving
- Ch12 — Benchmarking and Performance Measurement

### Hands-On Project

Build a memory budget for a 70B model.

Include:

- Weight memory
- Optimizer memory
- Gradient memory
- Activation memory
- KV cache memory
- Tensor parallelism assumptions
- ZeRO/FSDP assumptions

### Papers / References

- Megatron-LM
- ZeRO
- PagedAttention / vLLM
- MLPerf methodology

### Benchmark

Run an inference throughput benchmark over multiple batch sizes or sequence lengths.

### Communication Deliverable

Write a capacity-planning memo:

> How many users can this serving system support under a 4K average context and 32K peak context?

---

## Months 9–10 — Infrastructure at Scale

### Goal

You can reason about network topology, collectives, storage pipelines, checkpointing, and observability.

### Read Deeply

- Ch14 — Networking and Collective Communication
- Ch15 — Storage, Data Pipelines, and I/O Performance
- Ch16 — Cluster Management and Multi-Tenant Systems
- Ch17 — Telemetry, Observability, and Profiling Tools

### Hands-On Project

Run or study a multi-GPU communication benchmark.

Measure or model:

- AllReduce latency
- Bus bandwidth
- Message-size scaling
- Topology effects
- Communication/compute overlap

### Papers / References

- NCCL documentation
- InfiniBand/RDMA references
- Distributed training scaling papers
- Cluster observability references

### Benchmark

Run `nccl-tests` or equivalent.

### Communication Deliverable

Write an incident-style analysis:

> A 512-GPU training job has low MFU. What telemetry do you collect, and how do you isolate the bottleneck?

---

## Months 11–12 — Architecture Synthesis and Principal Identity

### Goal

You can operate at principal level: synthesize tradeoffs, communicate clearly, and guide infrastructure decisions.

### Read Deeply

- Ch05 — TCO and Power
- Ch18 — Principal Architect Career, Interviews, and Professional Development
- Revisit Ch01, Ch10, Ch11, Ch14, and Ch17

### Hands-On Project

Create a complete architecture recommendation.

Example topic:

> Should we use 256× H100 SXM, 512× H100 PCIe, or a memory-rich accelerator tier for a 70B training and serving workload?

Include:

- FLOP budget
- Memory budget
- Network analysis
- Storage/checkpointing plan
- Observability plan
- Cost model
- Risks
- Recommendation

### Papers / References

- Large-model training system reports
- Frontier model infrastructure reports
- Vendor architecture guides
- Serving-system papers

### Benchmark

Run a timed system design mock interview.

### Communication Deliverable

Create a 3–5 page principal-level architecture memo suitable for review by engineering leadership.

---

# Curriculum Summary

| Months | Focus | Chapters | Key Project |
|---|---|---|---|
| 1–2 | Foundations | Ch01, Ch02 | Transformer profiling from scratch |
| 3–4 | Hardware | Ch03A, Ch03B, Ch04, Ch05 | Roofline and memory bandwidth measurement |
| 5–6 | Optimization | Ch07, Ch08, Ch09 | Fusion / compiler / quantization experiment |
| 7–8 | Systems | Ch06, Ch10, Ch11, Ch12 | 70B memory and serving capacity budget |
| 9–10 | Infrastructure | Ch14, Ch15, Ch16, Ch17 | Network, storage, and observability analysis |
| 11–12 | Synthesis | Ch05, Ch18, review core chapters | Principal architecture recommendation |

---

# Final Table of Contents

## Front Matter

- Title Page
- Copyright and Edition Note
- Note on Accuracy and Currency
- Confidence Label System
- Preface
- Who This Book Is For
- What This Book Is Not
- What Makes This Book Different
- How to Use This Book
- Four Reader Paths
- Quick Reference by Time Budget
- 12-Month Principal-Level Curriculum
- Table of Contents

---

# Part I — Foundations

The performance mindset, transformer architecture, and accelerator fundamentals that support the rest of the book.

## Chapter 1 — The AI/ML Performance Architecture Mindset

- The three quantities of AI performance
- Seven-layer performance stack
- Roofline analysis
- Arithmetic intensity
- H100 dense BF16 ridge point
- Three performance regimes
- MFU vs HFU
- Measurement discipline
- Hypothesis-driven profiling
- Principal interview framing

## Chapter 2 — The Transformer Architecture: A Performance Engineer’s Deep Dive

- Transformer as a GEMM machine
- Tokenization and embeddings
- Multi-head self-attention
- GQA, MQA, and MLA
- RoPE
- Feed-forward networks
- SwiGLU
- LayerNorm and RMSNorm
- Residual connections
- Decoder-only architecture
- FLOP budget analysis
- Memory footprint analysis

## Chapter 3A — GPU and Accelerator Architecture Fundamentals

- SMs, warps, and threads
- Tensor Cores
- GPU memory hierarchy
- NVLink and NVSwitch
- SXM vs PCIe
- TPUs, Gaudi, and custom silicon
- Accelerator selection

## Chapter 3B — GPU Architecture Roadmap

- NVIDIA Ampere, Hopper, Blackwell, Rubin
- AMD CDNA generations
- MI300X and future roadmap concepts
- Interconnect evolution
- HBM evolution
- Hardware roadmap interpretation

---

# Part II — Hardware Performance

The physical constraints that bound every optimization.

## Chapter 4 — Memory Hierarchy and HBM Deep Dive

- Why memory is often the bottleneck
- HBM physical architecture
- HBM bandwidth theory vs sustained reality
- ECC and reliability
- DRAM roofline
- Cache hierarchy
- Memory access patterns
- Capacity planning

## Chapter 5 — Thermal Design, Power Management, and TCO

- GPU power delivery
- TDP/TBP/sustained power
- Thermal resistance and junction temperature
- Air, liquid, and immersion cooling
- Power throttling
- Rack power budget
- Capex and opex
- Cost per token
- Carbon and sustainability considerations

---

# Part III — LLM Inference, Kernels, and Optimization

The serving, kernel, precision, and compiler techniques that convert hardware potential into useful throughput.

## Chapter 6 — LLM Inference Systems Architecture

- Prefill vs decode
- TTFT and TPOT
- Continuous batching
- Iteration-level scheduling
- Chunked prefill
- Disaggregated prefill/decode
- Tensor parallel inference
- vLLM, TensorRT-LLM, SGLang
- Multi-tenant serving
- Serving observability
- Production optimization hierarchy

## Chapter 7 — GPU Kernels and CUDA Optimization

- Kernel execution model
- Memory coalescing
- Shared memory and bank conflicts
- Warp divergence
- Occupancy
- Tensor Core programming
- GEMM tiling
- FlashAttention
- Triton
- Kernel profiling

## Chapter 8 — Quantization and Precision Optimization

- Why quantization works
- INT8
- FP8
- INT4
- GPTQ
- AWQ
- SmoothQuant
- Quantization-aware training
- KV cache quantization
- Accuracy degradation
- Production decision framework

## Chapter 9 — Operator Fusion and Compiler Optimization

- Why fusion exists
- Memory traffic reduction
- Fusion candidates
- PyTorch 2.x
- TorchDynamo
- AOTAutograd
- TorchInductor
- Triton
- XLA
- TensorRT
- Compile failures
- CUDA Graphs

---

# Part IV — Training and Serving Systems

The distributed systems architectures that scale AI from a single GPU to many thousands.

## Chapter 10 — Training Systems Architecture at Scale

- Training stack
- Data parallelism
- Tensor parallelism
- Pipeline parallelism
- Sequence and context parallelism
- 4D parallelism decision tree
- ZeRO
- FSDP
- Gradient checkpointing
- Fault tolerance
- Training observability

## Chapter 11 — KV Cache: The Heart of LLM Serving

- KV cache memory math
- KV cache vs weight memory
- Context length explosion
- GQA and MQA reduction
- PagedAttention
- Fragmentation
- Prefix caching
- RadixAttention
- KV quantization
- KV tiering
- Multi-GPU KV cache
- Disaggregated prefill/decode
- MLA compression
- KV decision framework

## Chapter 12 — Benchmarking and Performance Measurement

- Benchmark design principles
- MLPerf inference
- MLPerf training
- Microbenchmarks
- End-to-end metrics
- Regression detection
- Capacity planning
- Benchmark harness design
- Regression CI
- Fleet health monitoring

---

# Part V — Advanced Techniques and Cluster Operations

The techniques and operational systems that make large-scale AI infrastructure reliable and efficient.

## Chapter 13 — Speculative Decoding and Mixture-of-Experts Infrastructure

- Speculative decoding theory
- Acceptance math
- Profitability analysis
- Draft model selection
- Deployment patterns
- MoE architecture
- Routing
- Expert parallelism
- All-to-All communication
- Load imbalance
- MoE operations

## Chapter 14 — Networking and Collective Communication

- Communication bottlenecks
- NVLink and NVSwitch
- NVLS
- InfiniBand
- RDMA
- SHARP
- GPUDirect RDMA
- NCCL
- RCCL
- AllReduce algorithms
- Topology sizing
- RoCEv2 and Ethernet AI fabrics
- Measurement workflow

## Chapter 15 — Storage, Data Pipelines, and I/O Performance

- Storage hierarchy
- Dataset pipeline architecture
- Data stalls
- Sequence packing
- Checkpoint storage
- Async checkpointing
- Parallel filesystems
- Object stores
- KV cache tiering to storage
- Storage specification framework

## Chapter 16 — Cluster Management and Multi-Tenant Systems

- SLURM vs Kubernetes
- Gang scheduling
- Fair-share and QOS
- GPU allocation
- MIG and vGPU
- Multi-tenant isolation
- SLA enforcement
- Preemption
- Autoscaling
- Cluster utilization

## Chapter 17 — Telemetry, Observability, and Profiling Tools

- Observability stack
- Prometheus
- Grafana
- DCGM
- XID errors
- Nsight Systems
- Nsight Compute
- ROCm tooling
- Tool selection decision tree
- SLA regression diagnosis
- Fail-slow GPU detection
- Performance regression CI

---

# Part VI — Principal Architect

The identity, communication, and interview patterns of the AI/ML Principal Performance Architect.

## Chapter 18 — The Principal Architect: Career, Interviews, and Professional Development

- What principal means in AI infrastructure
- The four superpowers
- Breadth vs depth
- STAR+Impact story framework
- Story bank
- Differentiating questions
- Written communication strategy
- Building a principal profile
- System design interview playbook
- LLM inference serving walkthrough
- Distributed training system walkthrough
- Reading roadmap and continued growth

---

# Back Matter

## Appendix A — Hardware Reference Tables

- GPU specification comparison
- HBM bandwidth and capacity
- NVLink and NVSwitch reference
- InfiniBand reference
- Roofline ridge points by GPU model

## Appendix B — Profiling Tool Quick Reference

- Tool selection decision tree
- Nsight Systems workflow
- Nsight Compute counters
- DCGM field IDs
- NCCL debug variables
- vLLM metrics
- ROCm equivalents
- `nccl-tests` interpretation

## Appendix C — Glossary

A practical glossary of AI/ML infrastructure and performance terms, from arithmetic intensity to ZeRO-3.

---

# Closing Note to the Reader

The goal of this book is not to make you memorize every number.

The goal is to make you dangerous with first principles.

When a new GPU appears, you should know how to estimate its ridge point.  
When a serving system slows down, you should know whether to inspect compute, memory, queueing, or KV cache pressure.  
When a training cluster underperforms, you should know how to separate kernels, communication, scheduling, storage, and stragglers.  
When leadership asks for a hardware recommendation, you should be able to explain the tradeoff in performance, risk, and cost.

That is the AI/ML Performance Architecture Mindset.

Welcome to *Silicon to Scale*.
