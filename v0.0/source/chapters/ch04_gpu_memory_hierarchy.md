# Chapter 4 — GPU Memory Hierarchy and HBM Deep Dive

> “In AI systems, performance is often decided less by how much math the accelerator can do, and more by how many bytes must move to make that math possible.”

---

## Chapter Overview

Chapter 1 introduced the performance mindset and the Roofline model.

Chapter 2 treated the transformer as a workload: GEMMs, attention, normalization, KV cache, and memory movement.

Chapter 3A explained the GPU as a performance system: SMs, warps, Tensor Cores, HBM, and interconnect.

Chapter 3B showed how accelerator generations evolve, especially through memory capacity, memory bandwidth, interconnect, and precision formats.

Chapter 4 goes deeper into the resource that appears in almost every AI infrastructure bottleneck:

```text
memory
```

This chapter answers:

- Why does HBM matter so much?
- Why can a workload be slow even when peak TFLOPS are enormous?
- Why is decode often memory-bandwidth-sensitive?
- Why does long-context serving become expensive?
- Why does FlashAttention improve performance?
- Why can H200 help inference even when it is not a brand-new architecture?
- Why does MI300X’s 192 GB HBM matter?
- How should a principal engineer diagnose and fix memory bottlenecks?

By the end of this chapter, you should be able to:

- Explain GPU memory hierarchy clearly.
- Distinguish registers, shared memory/L1, L2, HBM, host memory, and network memory paths.
- Calculate simple model memory requirements.
- Calculate and explain KV-cache memory growth.
- Use arithmetic intensity and ridge point to identify memory-bound workloads.
- Explain why HBM capacity and HBM bandwidth matter for LLM inference.
- Explain FlashAttention as a memory-traffic optimization.
- Choose memory optimization techniques based on the limiting memory tier.
- Explain GPU memory hierarchy in a principal-level interview.

---

## 4.0 Memory in One Page

A GPU has enormous compute capability, but compute is useful only when data arrives on time.

The memory hierarchy is a distance map:

```text
Registers       closest to compute, tiny, fastest
Shared/L1       on-chip, per-SM, fast tile reuse
L2              on-chip, shared across SMs
HBM             high-bandwidth device memory, large but expensive to access
Host memory     CPU-side memory, very large but much farther away
Network memory  remote GPUs/nodes, dominated by fabric and scheduling
```

A simple mental model:

```text
If the data is in registers, math can be fast.
If the data is in shared memory/L1, tile reuse is possible.
If the data is in L2, cross-SM reuse may help.
If the data is in HBM, bandwidth and capacity dominate.
If the data is in host memory, PCIe/NUMA latency can dominate.
If the data is across the network, the parallelism strategy matters.
```

> **Key Takeaway:** Memory optimization means keeping useful data closer to compute, moving fewer bytes, and increasing reuse before falling back to HBM, host memory, or the network.

---

## 4.1 GPU Memory Hierarchy

Every memory optimization starts with one question:

```text
Where does the data live when the math needs it?
```

That answer determines whether the workload sees:

- fast on-chip reuse,
- HBM bandwidth pressure,
- host-device transfer overhead,
- or distributed communication latency.

---

## Figure Placeholder — Fig 4.1

```markdown
![Fig 4.1 — GPU Memory Hierarchy Pyramid](../assets/diagrams/png_300dpi/ch04_fig_4_1_memory_hierarchy_pyramid.png)

**Fig 4.1 — GPU Memory Hierarchy Pyramid.** GPU memory is a hierarchy of capacity, latency, and bandwidth tradeoffs: registers and shared memory are closest to the math, while HBM, host memory, and network memory paths are farther away and more expensive to access.
```

**Figure intro:**  
Every memory optimization starts with one question: where does the data live when the math needs it? The answer determines whether the workload sees fast on-chip reuse, HBM bandwidth pressure, PCIe latency, or network communication overhead.

**Figure explanation:**  
The pyramid shows why locality matters. Data reused in registers or shared memory can feed compute efficiently. Data repeatedly fetched from HBM may become the bottleneck. Data that spills to host memory or crosses the network can dominate latency and destroy throughput.

> **Key Takeaway:** The farther data is from the compute units, the more carefully the workload must reuse, compress, tile, fuse, or schedule it.

---

## Table 4.1 — GPU Memory Tier Summary

| Tier | Scope | Capacity | What It Is Good For | Main Risk |
|---|---|---:|---|---|
| Registers | Per thread | Tiny | Fast operands and accumulator fragments | Spilling if register pressure is high |
| Shared memory / L1 | Per SM | Small | Tiling and producer/consumer reuse | Bank conflicts, limited capacity |
| L2 cache | Whole GPU | Medium | Cross-SM reuse and reduced HBM traffic | Low-reuse workloads bypass benefits |
| HBM | Whole GPU | Large | Model weights, activations, KV cache | Bandwidth and capacity bottlenecks |
| Host memory | CPU side | Very large | Staging and offload | PCIe latency and bandwidth penalty |
| Network memory path | Remote GPUs/nodes | Remote | Distributed workloads | Fabric latency, congestion, collectives |

A memory-bound workload is not simply “using memory.” It is using the wrong memory tier too often, with too little reuse, or with too many bytes per unit of useful math.

> **Key Takeaway:** The memory tier that supplies the critical path usually determines the optimization strategy.

---

## 4.2 Bandwidth View: Why Bytes Moved Dominate

The memory hierarchy is not only about capacity. It is also about bandwidth.

A GPU can have massive Tensor Core throughput, but if the workload streams too many bytes from HBM per unit of math, the compute units wait.

This is why two workloads can run very differently on the same GPU:

```text
Large GEMM with high reuse:
  Often compute-bound or Tensor-Core-bound.

LayerNorm / RMSNorm:
  Often HBM-bandwidth-sensitive because values are streamed with limited reuse.

Decode with KV cache:
  Often memory-bandwidth-sensitive because each token touches weights and prior KV state.

Naive attention:
  Can materialize large intermediate matrices and write/read too much HBM.

FlashAttention:
  Reduces HBM traffic using tiling and on-chip reuse.
```

---

## Figure Placeholder — Fig 4.2

```markdown
![Fig 4.2 — GPU Memory Hierarchy Bandwidth View](../assets/diagrams/png_300dpi/ch04_fig_4_2_gpu_memory_bandwidth_view.png)

**Fig 4.2 — GPU Memory Hierarchy Bandwidth View.** Bandwidth drops sharply as data moves farther from compute. AI performance often depends on minimizing HBM traffic and maximizing on-chip reuse.
```

**Figure intro:**  
The memory hierarchy is not only about capacity. It is also about bandwidth. A workload that repeatedly streams through HBM can become bandwidth-bound even on a GPU with enormous Tensor Core throughput.

**Figure explanation:**  
This bandwidth view connects directly to Roofline analysis. If an operation has low arithmetic intensity, it cannot reach the compute roof no matter how large the TFLOPS number is. Improving the workload often means increasing reuse or reducing bytes moved.

> **Key Takeaway:** Peak compute matters only if the memory hierarchy can feed it.

---

## 4.3 HBM: More Than “VRAM”

HBM stands for High Bandwidth Memory.

It is not just “more GPU memory.” It is a packaging and bandwidth technology.

HBM places stacks of DRAM close to the accelerator package and connects them through a very wide interface. That gives AI accelerators much higher memory bandwidth than conventional off-package memory approaches.

HBM matters because modern AI workloads often need to stream or store:

- model weights,
- activations,
- gradients,
- optimizer state,
- KV cache,
- temporary buffers,
- communication buffers,
- quantization metadata,
- and runtime scheduler state.

---

## Figure Placeholder — Fig 4.3

```markdown
![Fig 4.3 — HBM3e Die Stacking vs GDDR-Style Memory](../assets/diagrams/png_300dpi/ch04_fig_4_3_hbm3e_die_stacking.png)

**Fig 4.3 — HBM3e Die Stacking vs GDDR-Style Memory.** HBM uses stacked memory dies and a very wide interface close to the GPU package, enabling much higher bandwidth and energy efficiency than conventional off-package memory approaches.
```

**Figure intro:**  
HBM is not just “more VRAM.” It is a packaging and bandwidth technology. By stacking DRAM dies near the GPU and using a very wide interface, HBM delivers the bandwidth needed by modern AI accelerators.

**Figure explanation:**  
This physical structure explains why HBM capacity and bandwidth are expensive, power-sensitive, package-sensitive, and central to accelerator roadmap decisions. It also explains why HBM is one of the hardest resources to scale quickly.

> **Key Takeaway:** HBM is a packaging-level performance feature, not just a memory-size feature.

---

## 4.4 Product-Specific HBM Reference Values

Do not say:

```text
HBM3e is 4.8 TB/s.
```

That is not precise enough.

Say:

```text
[SHIPPED] NVIDIA H200 provides 141 GB HBM3e and 4.8 TB/s peak memory bandwidth.
```

HBM values are product-specific. They depend on the accelerator, memory stack, package, interface width, clocking, SKU, and system level.

---

## Table 4.2 — HBM Reference Values by Accelerator

| Accelerator / Product | Product Level | Memory | Peak Bandwidth | Confidence |
|---|---|---:|---:|---|
| H100 SXM5 | GPU | 80 GB HBM3 | 3.35 TB/s | `[SHIPPED]` |
| H200 | GPU | 141 GB HBM3e | 4.8 TB/s | `[SHIPPED]` |
| MI300X | GPU / OAM accelerator | 192 GB HBM3 | ≈5.3 TB/s | `[SHIPPED]` |
| MI325X | GPU / OAM accelerator | 256 GB HBM3e | ≈6 TB/s | `[SHIPPED]` or `[ANNOUNCED]` |
| MI350 Series | GPU / OAM accelerator | 288 GB HBM3e | ≈8 TB/s | `[SHIPPED]` if official product source |
| DGX B200 | System | 1,440 GB total GPU memory | 64 TB/s total HBM3e | `[SHIPPED]` system-level |

[SHIPPED] H100 SXM5 80 GB provides 80 GB HBM3 with 3.35 TB/s peak memory bandwidth.

[SHIPPED] NVIDIA H200 provides 141 GB HBM3e with 4.8 TB/s peak memory bandwidth, making it especially relevant for memory-capacity and memory-bandwidth constrained workloads.

[SHIPPED] AMD Instinct MI300X provides 192 GB HBM3 with approximately 5.3 TB/s peak local memory bandwidth.

[SHIPPED] DGX B200 lists 1,440 GB total GPU memory and 64 TB/s total HBM3e bandwidth across the system. These are system-level values and must not be presented as per-GPU values.

> **Key Takeaway:** Always ask whether an HBM number is per GPU, per module, per system, or per rack.

---

## 4.5 Memory-Bound vs Compute-Bound

A workload is not memory-bound simply because memory is involved.

A workload is memory-bound when the amount of useful math per byte moved is too low to reach the compute roof.

The key quantity is arithmetic intensity:

```text
Arithmetic intensity = FLOPs / bytes moved
```

The hardware ridge point is:

```text
Ridge point = peak compute / memory bandwidth
```

For H100 SXM5 dense BF16:

[DERIVED FROM SHIPPED]

```text
Dense BF16 peak ≈ 989.4 TFLOPS
HBM bandwidth    ≈ 3.35 TB/s

Ridge point ≈ 989.4 / 3.35
            ≈ 295 FLOP/byte
```

This means that a workload needs roughly 295 FLOPs per byte moved from HBM to become compute-bound on this simplified dense BF16 Roofline view.

If a workload has much lower arithmetic intensity, more Tensor Core peak will not solve the bottleneck.

---

## Figure Placeholder — Fig 4.4

```markdown
![Fig 4.4 — Memory-Bound vs Compute-Bound Decision Flow](../assets/diagrams/svg/ch04_fig_4_4_memory_compute_bound_flow.svg)

**Fig 4.4 — Memory-Bound vs Compute-Bound Decision Flow.** Use arithmetic intensity and ridge point to decide whether the next optimization should target math throughput or data movement.
```

**Figure intro:**  
A workload is not memory-bound because memory is involved. It is memory-bound when bytes moved per unit of math are too high for the hardware ridge point.

**Figure explanation:**  
The decision flow should route the reader from symptoms to action. Low Tensor Core utilization may mean compute underuse, but it may also mean memory stalls. The key is to compare arithmetic intensity to the ridge point and then confirm with profiler metrics.

> **Key Takeaway:** Classify the bottleneck before optimizing. Memory-bound workloads need fewer bytes moved, not just more compute.

---

## Table 4.3 — Memory Bottleneck Diagnostic Signals

| Symptom | Possible Cause | What to Check | Possible Fix |
|---|---|---|---|
| Low compute utilization | Memory stalls | HBM bandwidth, cache hit rate, stall reasons | Fusion, tiling, quantization |
| High HBM bandwidth | Streaming low-reuse kernel | Bytes moved per token/layer | Fuse ops, reduce precision |
| OOM at long context | KV cache growth | KV bytes per token and concurrency | GQA/MQA, KV quantization, paging |
| High PCIe traffic | CPU/GPU fallback or offload | Host-device transfer counters | Keep hot path on GPU |
| Poor scaling | Remote memory/collectives | NCCL/RCCL traces | Topology-aware placement |
| Slow attention | Too much HBM traffic | Attention kernel memory reads/writes | FlashAttention-style tiling |

[ENV-SPECIFIC] High HBM bandwidth utilization with low arithmetic-unit utilization is a strong memory-bound signal, but confirm with arithmetic intensity, cache behavior, stall reasons, and profiler counters.

> **Key Takeaway:** Memory bottlenecks have signatures. Match the symptom to the memory tier before choosing the fix.

---

## 4.6 Model Weight Memory: Simple Mental Math

A dense model’s raw weight memory is approximately:

```text
weight memory = parameter count × bytes per parameter
```

For a 70B model:

[ESTIMATED]

```text
BF16 weights ≈ 70B × 2 bytes = 140 GB
FP8 weights  ≈ 70B × 1 byte  = 70 GB
INT4 weights ≈ 70B × 0.5 byte = 35 GB
```

This excludes:

- KV cache,
- activations,
- temporary buffers,
- quantization metadata,
- runtime allocator fragmentation,
- communication buffers,
- speculative decoding buffers,
- serving scheduler overhead.

That is why a model that “fits by weight math” may still fail in production.

### Practical Reading

- H100 80 GB may fit FP8 weights for a 70B model but has limited KV-cache headroom.
- H200 141 GB gives more memory headroom.
- MI300X 192 GB gives larger HBM capacity.
- MI325X and MI350-class accelerators extend the large-HBM trend.
- System-level products like DGX B200 must be interpreted by product level and parallelism strategy.

> **Key Takeaway:** Weight memory is only the first memory calculation. Serving memory also includes KV cache, runtime buffers, fragmentation, and concurrency.

---

## 4.7 KV Cache Memory

KV cache is one of the most important memory structures in LLM serving.

During decode, each new token uses the prior context stored as keys and values. That means the KV cache grows with sequence length and concurrency.

A useful estimate:

[ESTIMATED]

```text
KV cache bytes ≈ 2 × L × B × S × n_kv × d_head × bytes_per_element
```

---

## Table 4.4 — KV Cache Memory Formula Variables

| Symbol | Meaning | Notes |
|---|---|---|
| `2` | Key and value tensors | One K and one V cache |
| `L` | Number of transformer layers | Model architecture dependent |
| `B` | Batch size / concurrent sequences | Serving concurrency |
| `S` | Sequence length | Grows with context |
| `n_kv` | Number of KV heads | Lower for GQA/MQA |
| `d_head` | Head dimension | Usually hidden size / attention heads |
| `bytes_per_element` | KV dtype size | BF16/FP16=2, FP8/INT8=1 |

The factor 2 accounts for key and value tensors.

For GQA/MQA models, use the number of KV heads, not the number of query heads.

### Why GQA and MQA Matter

Multi-head attention, grouped-query attention, and multi-query attention can have very different KV-cache sizes.

```text
MHA: many KV heads
GQA: fewer KV heads than query heads
MQA: one or very few KV heads
```

Reducing `n_kv` reduces KV-cache memory and bandwidth pressure.

> **Key Takeaway:** KV-cache growth is linear in layers, concurrency, sequence length, KV heads, head dimension, and bytes per element.

---

## Figure Placeholder — Fig 4.5

```markdown
![Fig 4.5 — KV Cache Memory Growth with Sequence Length and Concurrency](../assets/diagrams/svg/ch04_fig_4_5_kv_cache_growth_curve.svg)

**Fig 4.5 — KV Cache Memory Growth with Sequence Length and Concurrency.** KV-cache memory grows linearly with sequence length and concurrent sequences, making long-context serving a memory-capacity and memory-bandwidth problem.
```

**Figure intro:**  
A model may fit in HBM at short context but fail at long context or high concurrency. KV cache is often the difference between a theoretical deployment and a production deployment.

**Figure explanation:**  
The growth curve should show why batch size, sequence length, and KV precision interact. A serving system needs headroom for weights, KV cache, temporary buffers, fragmentation, and scheduler overhead.

> **Key Takeaway:** Long context is not free. It consumes HBM capacity and increases decode memory traffic.

---

## 4.8 Prefill vs Decode Memory Behavior

Prefill and decode stress memory differently.

### Prefill

Prefill processes the prompt.

It usually exposes more parallel work because the model can process many prompt tokens together.

[REPRESENTATIVE] Prefill usually exposes more parallel GEMM work and can achieve higher Tensor Core utilization than decode, depending on batch size, sequence length, and implementation.

### Decode

Decode generates one token at a time per sequence.

[REPRESENTATIVE] LLM decode is often memory-bandwidth-sensitive because each generated token may require reading model weights and prior KV-cache state while exposing limited arithmetic intensity compared with large prefill GEMMs.

This is why decode optimization often focuses on:

- batching strategy,
- KV-cache layout,
- KV-cache paging,
- KV-cache quantization,
- GQA/MQA,
- speculative decoding,
- memory bandwidth,
- and serving scheduler efficiency.

> **Key Takeaway:** Prefill and decode are not the same workload. Prefill often exposes more parallel compute; decode often stresses memory bandwidth and KV-cache behavior.

---

## 4.9 FlashAttention as a Memory Optimization

Attention is a perfect example of why memory hierarchy matters.

A naive attention implementation can materialize large intermediate matrices:

```text
QK^T
softmax(QK^T)
softmax(QK^T)V
```

If the intermediate attention matrix is written to HBM and read back, memory traffic becomes expensive.

FlashAttention changes the memory schedule.

[REPRESENTATIVE] FlashAttention improves attention performance by reducing HBM reads/writes through on-chip tiling and online softmax, avoiding materialization of the full attention matrix in HBM. Exact speedup is workload-, hardware-, and implementation-specific.

---

## Figure Placeholder — Fig 4.6

```markdown
![Fig 4.6 — FlashAttention and HBM Traffic Reduction](../assets/diagrams/png_300dpi/ch04_fig_4_6_flashattention_hbm_traffic.png)

**Fig 4.6 — FlashAttention and HBM Traffic Reduction.** FlashAttention-style kernels reduce HBM reads and writes by tiling attention and keeping intermediate state in on-chip memory instead of materializing the full attention matrix in HBM.
```

**Figure intro:**  
Attention is a perfect example of why memory hierarchy matters. A naive attention implementation can write and reread large intermediate matrices. FlashAttention changes the memory schedule.

**Figure explanation:**  
FlashAttention does not make attention free. It changes the bottleneck by reducing HBM traffic and increasing on-chip reuse. Exact speedups depend on sequence length, head dimension, dtype, GPU generation, and implementation.

> **Key Takeaway:** FlashAttention is a memory optimization: it wins by moving fewer bytes through HBM.

---

## 4.10 Memory Optimization Techniques

Once the memory tier and bottleneck are known, the optimization menu becomes clearer.

Some techniques reduce bytes moved.  
Some improve on-chip reuse.  
Some trade compute for memory.  
Some reduce precision.  
Some change scheduling.  
Some add operational complexity.

---

## Table 4.5 — Memory Optimization Techniques

| Technique | Reduces | Best For | Caveat |
|---|---|---|---|
| Tiling | HBM traffic | GEMM, attention | Requires kernel support |
| Fusion | Intermediate reads/writes | Norm, activation, elementwise chains | May increase register pressure |
| Quantization | Weight/KV bytes | Inference | Quality and kernel support matter |
| KV quantization | KV-cache memory/bandwidth | Decode and long context | Accuracy and implementation matter |
| GQA/MQA | KV heads | LLM inference | Architecture-specific |
| Paging | Fragmentation/OOM | Serving systems | Scheduler overhead |
| Recomputation | Activation memory | Training | More compute |
| CPU offload | HBM capacity | Oversized models | PCIe latency/bandwidth penalty |
| Topology-aware placement | Remote memory/communication | Distributed workloads | Requires scheduler/fabric awareness |

[REPRESENTATIVE] Operator fusion can reduce HBM traffic by avoiding intermediate tensor materialization, but it may increase register pressure, reduce occupancy, or require specialized kernels.

[ESTIMATED] Reducing precision from BF16 to FP8 roughly halves raw value storage, and INT4 roughly quarters it, before metadata and packing overhead.

[REPRESENTATIVE] Production quantization must account for accuracy, kernel support, calibration, serving-stack compatibility, and observability.

[REPRESENTATIVE] CPU offload can reduce HBM capacity pressure, but if it enters the critical path it can become limited by PCIe bandwidth, host-memory latency, NUMA placement, DMA behavior, and software scheduling.

> **Key Takeaway:** Memory optimization is tradeoff management, not a single trick.

---

## 4.11 Wrong Fix vs Right First Question

Memory problems often look like compute problems from far away.

A senior engineer may jump to a fix.

A principal engineer first names the limiting resource.

---

## Table 4.6 — Wrong Fix vs Right First Question for Memory Problems

| Symptom | Wrong Fix | Right First Question |
|---|---|---|
| Low GPU utilization | Add more GPUs | Is the kernel memory-bound or launch/overhead-bound? |
| High decode latency | Use a bigger GPU for TFLOPS | Is KV-cache bandwidth the bottleneck? |
| OOM at long context | Reduce batch blindly | How many bytes does KV cache consume per request? |
| Slow attention | Tune only GEMM | Is attention materializing too much HBM traffic? |
| Slow norm/activation chain | Increase Tensor Core focus | Are elementwise ops streaming HBM repeatedly? |
| Poor multi-GPU scaling | Add faster GPUs | Is remote communication or topology the bottleneck? |

> **Key Takeaway:** The fastest way to waste optimization time is to fix the wrong bottleneck.

---

## 4.12 Memory Optimization Decision Tree

The chapter should end with a practical decision tool.

Once the bottleneck is classified, the solution space becomes smaller and more rational:

```text
Capacity problem?
  Quantize, shard, page, offload, reduce context, reduce concurrency.

Bandwidth problem?
  Fuse, tile, quantize, improve layout, use better kernels, reduce KV traffic.

On-chip reuse problem?
  Use shared memory tiling, blocking, FlashAttention-style schedules.

Host-memory problem?
  Keep hot path on GPU, reduce transfers, prefetch, pin memory, fix NUMA placement.

Distributed memory / communication problem?
  Improve topology placement, overlap communication, adjust parallelism strategy.
```

---

## Figure Placeholder — Fig 4.7

```markdown
![Fig 4.7 — Memory Optimization Decision Tree](../assets/diagrams/svg/ch04_fig_4_7_memory_optimization_decision_tree.svg)

**Fig 4.7 — Memory Optimization Decision Tree.** Choose memory optimizations by identifying whether the bottleneck is HBM capacity, HBM bandwidth, on-chip reuse, PCIe/host offload, or interconnect.
```

**Figure intro:**  
The chapter should end with a practical decision tool. Once the bottleneck is classified, the solution space becomes smaller and more rational.

**Figure explanation:**  
The decision tree should route capacity problems toward quantization, sharding, paging, or offload; bandwidth problems toward fusion, tiling, KV quantization, and layout; and communication problems toward topology-aware placement and overlap.

> **Key Takeaway:** Memory optimization starts by naming the limiting memory tier.

---

## 4.13 How to Explain GPU Memory Hierarchy in a Principal Interview

A weak answer sounds like this:

```text
GPU memory is fast, and HBM has high bandwidth.
```

That is true, but too shallow.

A principal-level answer sounds like this:

> I start with the memory movement path, not just peak TFLOPS. I ask where the data lives, how often it is reused, and whether arithmetic intensity is high enough to use the compute roof. For LLM inference, decode is often HBM- and KV-cache-bandwidth-sensitive. For attention, FlashAttention improves performance by reducing HBM traffic through on-chip tiling. For hardware selection, HBM capacity and bandwidth can dominate over peak TFLOPS.

### Scenario 1 — Why Can a GPU with More TFLOPS Be Slower?

Answer:

```text
If the workload is memory-bound, peak TFLOPS is not the limiting resource. A GPU with more compute but insufficient memory bandwidth, smaller HBM capacity, worse cache behavior, or a weaker software kernel path can underperform on memory-sensitive workloads.
```

### Scenario 2 — Why Does H200 Matter for Inference?

Answer:

```text
H200 matters because it increases HBM capacity and bandwidth. For memory-bound LLM inference and long-context serving, more HBM and more bandwidth can improve model fit, KV-cache headroom, and decode behavior even when the compute architecture is still Hopper-generation.
```

### Scenario 3 — Why Does MI300X Matter?

Answer:

```text
MI300X matters because 192 GB HBM3 changes model-fit and KV-cache economics. For memory-heavy serving, capacity can reduce sharding pressure. The final production decision still depends on ROCm/framework maturity, kernel availability, interconnect, power, and cost.
```

### Scenario 4 — Why Does FlashAttention Help?

Answer:

```text
FlashAttention helps because it changes the memory schedule. Instead of materializing the full attention matrix in HBM, it tiles the computation and uses on-chip memory with online softmax, reducing HBM reads and writes.
```

### Scenario 5 — How Do You Diagnose a Memory Bottleneck?

Answer:

```text
I compare arithmetic intensity to the ridge point, then confirm with profiler metrics: HBM bandwidth, cache behavior, stall reasons, Tensor Core utilization, memory transactions, and host-device transfers. I then choose a fix based on whether the bottleneck is HBM capacity, HBM bandwidth, on-chip reuse, PCIe/offload, or interconnect.
```

---

## 4.14 Chapter Cheat Sheet

### Memory Hierarchy

```text
Registers → Shared/L1 → L2 → HBM → Host Memory → Network
```

### Arithmetic Intensity

```text
Arithmetic intensity = FLOPs / bytes moved
```

### Ridge Point

```text
Ridge point = peak compute / memory bandwidth
```

### H100 Dense BF16 Ridge Point

[DERIVED FROM SHIPPED]

```text
H100 SXM5 dense BF16 peak ≈ 989.4 TFLOPS
H100 SXM5 HBM bandwidth    ≈ 3.35 TB/s
Ridge point                ≈ 295 FLOP/byte
```

### Model Weight Memory

[ESTIMATED]

```text
70B BF16 ≈ 140 GB
70B FP8  ≈ 70 GB
70B INT4 ≈ 35 GB
```

### KV Cache

[ESTIMATED]

```text
KV cache bytes ≈ 2 × L × B × S × n_kv × d_head × bytes_per_element
```

### Memory Optimization Principle

```text
Move fewer bytes.
Reuse data closer to compute.
Use lower precision when quality allows.
Avoid host/network movement in the critical path.
```

---

## 4.15 Key Takeaways

1. AI performance is often limited by bytes moved, not FLOPs available.
2. GPU memory hierarchy is a distance map from registers to remote memory.
3. Registers and shared memory are fast but tiny.
4. HBM is large and high-bandwidth, but still expensive relative to on-chip reuse.
5. Host memory and network memory paths are much slower and must be kept out of the critical path when possible.
6. HBM capacity determines model-fit and KV-cache headroom.
7. HBM bandwidth determines many memory-bound inference behaviors.
8. H100 SXM5 has 80 GB HBM3 and 3.35 TB/s bandwidth.
9. H200 provides 141 GB HBM3e and 4.8 TB/s bandwidth.
10. MI300X provides 192 GB HBM3 and approximately 5.3 TB/s bandwidth.
11. Ridge point connects compute peak and memory bandwidth.
12. H100 dense BF16 ridge point is approximately 295 FLOP/byte.
13. KV-cache memory grows with layers, concurrency, sequence length, KV heads, head dimension, and dtype size.
14. Decode is often memory-bandwidth-sensitive.
15. FlashAttention is fundamentally a memory-traffic optimization.
16. The right fix depends on the limiting memory tier.
17. Principal-level memory analysis starts with bottleneck classification, not a favorite optimization trick.

---

## 4.16 Review Questions

### Conceptual

1. Why is HBM more than just “GPU memory”?
2. What are the main tiers of GPU memory hierarchy?
3. Why are registers and shared memory so important for performance?
4. Why can a workload be memory-bound on a GPU with huge TFLOPS?
5. What is arithmetic intensity?
6. What is a hardware ridge point?
7. Why should dense BF16 peak be used for dense BF16 ridge-point math?
8. Why does HBM capacity matter for LLM inference?
9. Why does HBM bandwidth matter for decode?
10. Why does KV cache grow with sequence length?
11. Why do GQA and MQA reduce KV-cache memory?
12. Why does FlashAttention reduce HBM traffic?
13. Why can CPU offload hurt latency?
14. Why is product level important when comparing HBM values?

### Calculation

1. Estimate BF16 weight memory for a 70B model.
2. Estimate FP8 weight memory for a 70B model.
3. Estimate INT4 weight memory for a 70B model.
4. Calculate H100 SXM5 dense BF16 ridge point using 989.4 TFLOPS and 3.35 TB/s.
5. Write the KV-cache memory formula and explain each variable.
6. If sequence length doubles, what happens to KV-cache memory?
7. If `n_kv` is reduced by 4× using GQA, what happens to KV-cache memory?
8. If KV cache moves from BF16 to FP8, what happens to raw KV storage before metadata?

### Principal-Level Interview Practice

1. Explain GPU memory hierarchy in two minutes.
2. Explain how you would diagnose a memory-bound kernel.
3. Explain why H200 can help inference even if compute is similar to H100.
4. Explain why MI300X’s 192 GB HBM matters.
5. Explain why FlashAttention is a memory optimization.
6. Explain why decode is often memory-bandwidth-sensitive.
7. Explain why long-context serving can become expensive.
8. Explain how you would choose between H100, H200, and MI300X for a 70B serving workload.
9. Explain the difference between HBM capacity and HBM bandwidth.
10. Explain why adding more GPUs may not fix a memory bottleneck.

---

## 4.17 Production Notes for This Chapter

### Figure Assets Needed

| Figure | Status |
|---|---|
| Fig 4.1 — Memory Hierarchy Pyramid | Existing; needs print export |
| Fig 4.2 — GPU Memory Hierarchy Bandwidth View | Existing; needs print export |
| Fig 4.3 — HBM3e Die Stacking vs GDDR-Style Memory | Existing; needs print export |
| Fig 4.4 — Memory-Bound vs Compute-Bound Decision Flow | Must be created |
| Fig 4.5 — KV Cache Memory Growth Curve | Must be created |
| Fig 4.6 — FlashAttention HBM Traffic Reduction | Existing/adapted; needs export |
| Fig 4.7 — Memory Optimization Decision Tree | Must be created |

### Table Assets Included

| Table | Status |
|---|---|
| Table 4.1 — GPU Memory Tier Summary | Included |
| Table 4.2 — HBM Reference Values by Accelerator | Included |
| Table 4.3 — Memory Bottleneck Diagnostic Signals | Included |
| Table 4.4 — KV Cache Memory Formula Variables | Included |
| Table 4.5 — Memory Optimization Techniques | Included |
| Table 4.6 — Wrong Fix vs Right First Question | Included |

### Source Notes to Add in Final Book

Use official or primary sources for:

- NVIDIA H100 product page / datasheet
- NVIDIA H200 product page / datasheet
- AMD MI300X product page / datasheet
- AMD MI325X and MI350 Series product pages if referenced
- NVIDIA DGX B200 product page if system-level values are included
- NVIDIA Hopper architecture whitepaper
- FlashAttention paper and implementation documentation
- CUDA / Nsight Compute documentation
- ROCm / rocprof documentation
- PCI-SIG PCIe specification
- Ch11 KV-cache reference material

---

## 4.18 Bridge to Chapter 5

Chapter 4 showed why memory capacity, bandwidth, and data movement often dominate AI performance.

But memory is not the only physical constraint.

As accelerators become denser, hotter, and more expensive to operate, power and cooling become architecture-level constraints.

Chapter 5 moves from:

```text
How do bytes move?
```

to:

```text
How much power, cooling, and infrastructure are required to sustain the system?
```
