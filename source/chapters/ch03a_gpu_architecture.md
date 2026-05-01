# Chapter 3A — GPU and Accelerator Architecture Fundamentals

> “A GPU is not a faster CPU. It is a throughput machine built around parallel execution, memory hierarchy, matrix engines, and communication fabric.”

---

## Chapter Overview

Chapter 1 gave us the performance mindset: classify the bottleneck before optimizing.

Chapter 2 showed that a transformer is not just a model architecture. It is a workload made of GEMMs, attention, normalization, KV-cache state, and memory movement.

Chapter 3A moves down into the hardware.

The goal of this chapter is not to memorize every GPU specification. Hardware changes too fast for that. The goal is to build a mental model that lets you read any accelerator spec sheet and ask the right questions:

- Does the model fit in HBM?
- Is the workload compute-bound, memory-bound, communication-bound, or overhead-bound?
- Does the workload use Tensor Cores or equivalent matrix engines efficiently?
- Does the topology support the parallelism strategy?
- Does the software stack expose the hardware capability?
- What does the system cost in power, cooling, latency, and operational risk?

A principal AI/ML performance architect does not choose a GPU because one number is large. They choose a system because the hardware, software stack, workload shape, and cost model fit together.

By the end of this chapter, you should be able to:

- Explain why GPUs are throughput machines, not latency machines.
- Explain the CPU vs GPU architectural tradeoff.
- Describe Streaming Multiprocessors, warps, SIMT execution, occupancy, and CUDA hierarchy.
- Explain why Tensor Cores matter for AI workloads.
- Explain why Tensor Core peak throughput is not automatic.
- Reason about GPU memory hierarchy: registers, shared memory/L1, L2, HBM, host memory, and network memory paths.
- Correctly distinguish dense vs sparse peak TFLOPS.
- Explain H100, H200, MI300X, and Blackwell-class hardware values with confidence labels.
- Explain PCIe, NVLink, NVSwitch, and InfiniBand directionality.
- Explain PCIe vs SXM/HGX vs OAM form-factor tradeoffs.
- Compare NVIDIA, AMD, TPU, and Gaudi-style accelerators neutrally.
- Read a GPU spec sheet like a performance architect.
- Explain GPU architecture in a principal-level interview.

---

## 3A.0 The GPU Mental Model

Most engineers first learn performance on CPUs.

That shapes their instincts:

```text
CPU instinct:
Make one thread fast.
Avoid branch misprediction.
Improve cache locality.
Reduce latency.
```

Those instincts are useful, but incomplete for AI infrastructure.

A GPU is designed differently:

```text
GPU instinct:
Run many threads.
Hide latency with occupancy.
Use massive memory bandwidth.
Feed matrix engines.
Maximize throughput.
```

A CPU spends a large fraction of its silicon budget on sophisticated control logic: branch prediction, out-of-order execution, speculative execution, large caches, and latency optimization.

A GPU spends more of its silicon budget on parallel execution units, high-bandwidth memory interfaces, and matrix-multiply engines.

That is why the GPU is powerful for transformers.

A transformer exposes enormous amounts of parallel matrix work:

- QKV projection GEMMs
- Output projection GEMMs
- FFN / SwiGLU GEMMs
- Attention kernels
- Batched inference
- Large training batches
- Distributed collective operations

But the GPU is not magic. It needs the workload to be shaped correctly. A small, divergent, memory-bound, poorly aligned workload can leave most of the GPU idle.

---

## Figure Placeholder — Fig 3A.1

```markdown
![Fig 3A.1 — GPU vs CPU Architecture Comparison](../assets/diagrams/png_300dpi/ch03a_fig_3a_1_gpu_vs_cpu_architecture.png)

**Fig 3A.1 — GPU vs CPU Architecture Comparison.** A CPU is optimized for low-latency execution of a smaller number of complex threads. A GPU is optimized for high-throughput execution of many simpler threads. AI workloads benefit from the GPU model when they expose enough parallel matrix and tensor work.
```

**Figure intro:**  
The first mistake many engineers make is treating a GPU as a faster CPU. It is not. CPUs and GPUs make different architectural tradeoffs. CPUs spend more silicon on control and low-latency execution. GPUs spend more silicon on parallel execution units and memory bandwidth.

**Figure explanation:**  
This difference explains why GPUs are powerful for transformer workloads. Training and prefill contain large GEMMs and attention kernels that expose massive parallelism. But the same difference also explains why branch divergence, small batches, poor memory locality, and decode memory pressure can underutilize the GPU.

> **Key Takeaway:** A CPU is a latency machine. A GPU is a throughput machine. AI performance depends on exposing enough parallel work to keep the GPU’s execution units and memory system busy.

---

## 3A.1 GPU in One Page

A GPU can be summarized like this:

```text
Threads are grouped into warps.
Warps are scheduled on SMs.
SMs execute instructions on CUDA cores, Tensor Cores, load/store units, and special-function units.
Tensor Cores provide the matrix math throughput that powers modern AI.
Registers and shared memory are closest to compute.
L2 and HBM feed the whole GPU.
NVLink/NVSwitch connect GPUs inside a node.
InfiniBand/Ethernet connect nodes across a cluster.
```

This chapter expands that summary one layer at a time.

---

## 3A.2 The GPU Memory Hierarchy

Most GPU performance problems are data movement problems.

The arithmetic units are fast. The question is whether data arrives on time, in the right layout, at the right reuse distance.

For transformer workloads:

- QKV and FFN GEMMs want high Tensor Core utilization.
- FlashAttention wants on-chip tile reuse.
- LayerNorm and RMSNorm often stream through HBM.
- RoPE should usually be fused with nearby operations.
- KV cache stresses HBM capacity and bandwidth.
- Tensor parallelism stresses interconnect bandwidth and latency.

The memory hierarchy explains why.

The closer data is to the SM, the faster and more efficient it is. But closer memory is smaller. Farther memory is larger but slower and more expensive to access.

---

## Figure Placeholder — Fig 3A.2

```markdown
![Fig 3A.2 — GPU Memory Hierarchy](../assets/diagrams/png_300dpi/ch03a_fig_3a_2_gpu_memory_hierarchy.png)

**Fig 3A.2 — GPU Memory Hierarchy.** GPU performance depends on how often data can be reused in registers, shared memory/L1, and L2 before falling back to HBM or host memory. The farther data travels, the more expensive it becomes in latency, bandwidth pressure, and energy.
```

**Figure intro:**  
Most GPU performance problems are data movement problems. Tensor Cores can execute enormous amounts of math, but they are only useful if data arrives in the right shape at the right time.

**Figure explanation:**  
Registers and shared memory are close to the math but limited in capacity. HBM provides high bandwidth but is much slower and more energy-expensive than on-chip storage. PCIe and host memory are farther away still. A strong GPU performance engineer tries to maximize reuse before data leaves the closest usable memory tier.

> **Key Takeaway:** GPU optimization is often the art of moving data fewer times and reusing it closer to the compute units.

---

## Table 3A.1 — GPU Memory Hierarchy Summary

| Tier | Scope | Capacity | Performance Role | Common Optimization |
|---|---|---:|---|---|
| Registers | Per thread | Tiny | Fastest operand storage | Keep hot values in registers |
| Shared memory / L1 | Per SM | Small | Explicit tile reuse and low-latency sharing | Tiling, bank-conflict avoidance |
| L2 cache | Whole GPU | Medium | Cross-SM reuse and HBM traffic reduction | Locality, reuse, streaming behavior |
| HBM | Whole GPU | Large | Main high-bandwidth model, activation, and KV storage | Coalescing, fusion, quantization, cache layout |
| PCIe / Host memory | CPU-GPU path | Very large | Data staging, offload, slow fallback | Avoid in critical path, prefetch, pin memory |
| Network memory path | Multi-node | Remote | Distributed training and serving communication | Overlap, topology-aware placement |

The same transformer operation can behave differently depending on which memory tier it stresses.

A GEMM with good tile reuse can be compute-bound. A normalization kernel that streams through HBM with little reuse can be memory-bound. A distributed training step can become network-bound even when local GPU behavior is excellent.

> **Key Takeaway:** The right optimization depends on which memory tier is the bottleneck. Do not treat all data movement as the same.

---

## 3A.3 HBM: The Scarce Resource Behind Modern AI

High Bandwidth Memory, or HBM, is one of the most important resources in AI systems.

HBM determines:

- Whether the model fits.
- How large the KV cache can be.
- How many concurrent requests can be served.
- How much tensor parallelism is required.
- Whether decode is bandwidth-limited.
- Whether long context is practical.

### Key HBM Reference Values

| Accelerator | HBM Capacity | HBM Bandwidth | Confidence |
|---|---:|---:|---|
| NVIDIA H100 SXM5 | 80 GB HBM3 | 3.35 TB/s | `[SHIPPED]` |
| NVIDIA H200 | 141 GB HBM3e | 4.8 TB/s | `[SHIPPED]` |
| AMD MI300X | 192 GB HBM3 | ≈5.3 TB/s | `[SHIPPED]` |

[SHIPPED] H100 SXM5 80 GB provides 80 GB of HBM3 with 3.35 TB/s peak memory bandwidth.

[SHIPPED] H200 provides 141 GB of HBM3e with 4.8 TB/s peak memory bandwidth.

[SHIPPED] AMD Instinct MI300X provides 192 GB of HBM3 and approximately 5.3 TB/s peak memory bandwidth.

### Why HBM Capacity Matters

For a 70B dense model:

```text
BF16 weight memory ≈ 70B parameters × 2 bytes = 140 GB
FP8 weight memory  ≈ 70B parameters × 1 byte  = 70 GB
INT4 weight memory ≈ 70B parameters × 0.5 byte = 35 GB
```

An 80 GB H100 can hold FP8 weights for a 70B model, but KV cache, activations, runtime buffers, and fragmentation still matter.

An H200 changes the serving economics because 141 GB leaves much more room for KV cache.

An MI300X changes a different dimension because 192 GB HBM can reduce or eliminate some model-sharding pressure for memory-heavy workloads.

> **Key Takeaway:** For LLM inference, HBM capacity can matter as much as peak TFLOPS. If weights and KV cache do not fit, the theoretical compute roof is irrelevant.

---

## 3A.4 Streaming Multiprocessors: Where GPU Execution Happens

The Streaming Multiprocessor, or SM, is the central execution unit of an NVIDIA GPU.

A kernel launch creates work. That work is divided into blocks and warps. Blocks are scheduled onto SMs. Warps execute on the SM’s functional units.

An SM contains resources such as:

- Warp schedulers
- CUDA cores
- Tensor Cores
- Load/store units
- Registers
- Shared memory / L1
- Special function units
- Instruction pipelines

[SHIPPED] The H100 SXM5 80 GB configuration exposes 132 SMs. Treat SM count as SKU-specific when comparing PCIe, NVL, SXM, or future derivatives.

[SHIPPED] H100 SXM5 includes 50 MB of unified L2 cache.

[SHIPPED] H100 provides a large per-SM shared-memory/L1 subsystem, with up to 228 KB of shared memory per SM depending on configuration. Effective kernel behavior depends on shared-memory allocation, cache configuration, access pattern, and compiler/kernel implementation.

---

## Figure Placeholder — Fig 3A.3

```markdown
![Fig 3A.3 — H100 Streaming Multiprocessor Internal Block Diagram](../assets/diagrams/png_300dpi/ch03a_fig_3a_3_h100_sm_internal_block.png)

**Fig 3A.3 — H100 Streaming Multiprocessor Internal Block Diagram.** The SM is the fundamental execution unit of an NVIDIA GPU. It contains warp schedulers, CUDA cores, Tensor Cores, load/store units, registers, and shared memory/L1 resources.
```

**Figure intro:**  
The SM is where GPU work actually executes. When a kernel launches, blocks are scheduled onto SMs, warps are issued by warp schedulers, and instructions execute on functional units such as CUDA cores, Tensor Cores, and load/store pipelines.

**Figure explanation:**  
A kernel can be limited by Tensor Core utilization, load/store throughput, register pressure, shared-memory usage, occupancy, instruction scheduling, or memory stalls. The SM is the bridge between code-level choices and hardware-level execution.

> **Key Takeaway:** The SM is the unit where GPU performance becomes real. Occupancy, warp scheduling, Tensor Core use, memory stalls, and register pressure all meet inside the SM.

---

## Table 3A.2 — Streaming Multiprocessor Resource Summary

| Resource | What It Does | Why It Matters |
|---|---|---|
| Warp schedulers | Issue instructions for resident warps | Determines ability to hide latency |
| CUDA cores | Execute scalar/vector FP32/INT-style instructions | Handles non-Tensor-Core math and control-heavy code |
| Tensor Cores | Execute matrix-multiply-accumulate operations | Main source of AI GEMM throughput |
| Load/store units | Move data between memory hierarchy and registers | Critical for memory-bound kernels |
| Registers | Fast per-thread storage | Too much register use can reduce occupancy |
| Shared memory / L1 | On-chip data sharing and tile reuse | Enables high-performance tiling |
| L2 cache | Shared cache across SMs | Reduces HBM traffic |
| Occupancy slots | Resident warps/blocks | Helps hide latency but is not the same as utilization |

When a kernel is slow, this table helps route the investigation.

- High register usage may limit occupancy.
- Poor memory coalescing may stress load/store units.
- A GEMM that does not use Tensor Cores may leave most AI throughput unused.
- A memory-bound kernel may not improve even if occupancy is high.

> **Key Takeaway:** SM performance is not one number. It is the interaction of schedulers, Tensor Cores, memory pipelines, registers, shared memory, and occupancy.

---

## 3A.5 Warps, SIMT, and Occupancy

A warp is the basic scheduling unit on NVIDIA GPUs.

On NVIDIA GPUs, a warp contains 32 threads.

GPU programming often looks like scalar programming:

```cpp
if (condition) {
    do_path_A();
} else {
    do_path_B();
}
```

But the hardware executes groups of lanes together. This is called SIMT: Single Instruction, Multiple Threads.

When all threads in a warp follow the same path, execution is efficient.

When different threads take different branches, the warp may execute both paths serially while masking off inactive lanes. This is called divergence.

---

## Figure Placeholder — Fig 3A.4

```markdown
![Fig 3A.4 — SIMT Warp Execution: Divergence and Occupancy](../assets/diagrams/png_300dpi/ch03a_fig_3a_4_simt_warp_execution.png)

**Fig 3A.4 — SIMT Warp Execution: Divergence and Occupancy.** A warp executes multiple threads in lockstep. When threads take different control-flow paths, execution serializes across branches, reducing effective throughput.
```

**Figure intro:**  
GPUs execute groups of threads called warps. The SIMT model lets the programmer write scalar-looking code while the hardware issues instructions across a warp of lanes.

**Figure explanation:**  
Warp execution is efficient when threads follow the same path and access memory coherently. Divergence causes the warp to execute multiple paths serially. Occupancy measures how many warps are resident, but high occupancy does not guarantee high useful throughput.

> **Key Takeaway:** Warp divergence wastes parallel lanes. Occupancy helps hide latency, but it is not the same as useful hardware utilization.

---

## 3A.6 Occupancy Is Not Utilization

This concept is important enough to isolate.

```text
Occupancy = how many warps are resident.
Utilization = how much useful work the hardware is doing.
```

High occupancy can help hide memory latency. But high occupancy is not always the goal.

A low-occupancy kernel can be excellent if it:

- Uses Tensor Cores efficiently.
- Reduces HBM traffic.
- Uses shared memory well.
- Avoids unnecessary synchronization.
- Saturates the true bottleneck.

A high-occupancy kernel can still be slow if it:

- Performs uncoalesced memory accesses.
- Diverges heavily.
- Spills registers to local memory.
- Uses the wrong datatype path.
- Misses Tensor Core acceleration.
- Stalls on memory or synchronization.

[REPRESENTATIVE] Occupancy is a diagnostic signal, not a performance goal by itself.

> **Key Takeaway:** Do not optimize occupancy blindly. First identify whether the bottleneck is compute, memory, latency hiding, synchronization, or instruction mix.

---

## 3A.7 CUDA Thread Hierarchy

CUDA exposes a hierarchy:

```text
Grid
  └── Blocks
        └── Warps
              └── Threads
```

A kernel launch creates a grid. The grid contains blocks. Blocks contain threads. Threads execute in warps.

The hardware schedules blocks onto SMs based on available resources:

- Registers per thread
- Shared memory per block
- Threads per block
- Warps per block
- Hardware limits per SM

This is why block size, register pressure, and shared-memory usage matter.

---

## Figure Placeholder — Fig 3A.5

```markdown
![Fig 3A.5 — CUDA Thread Hierarchy: Grid, Block, Warp, Thread](../assets/diagrams/png_300dpi/ch03a_fig_3a_5_cuda_thread_hierarchy.png)

**Fig 3A.5 — CUDA Thread Hierarchy: Grid, Block, Warp, Thread.** CUDA exposes a hierarchy of grids, blocks, warps, and threads. Blocks are scheduled onto SMs, and warps are the basic scheduling unit inside the SM.
```

**Figure intro:**  
To connect code to hardware, the reader needs the CUDA execution hierarchy. A kernel launch creates a grid. The grid contains blocks. Blocks contain threads. Threads are executed in warps. Blocks are assigned to SMs based on available resources.

**Figure explanation:**  
This hierarchy explains why block size, shared-memory use, register count, and occupancy matter. A kernel with too many registers or too much shared memory per block may reduce the number of blocks that fit on an SM. A kernel with poor thread mapping may produce uncoalesced memory access or divergence.

> **Key Takeaway:** CUDA programming choices become hardware scheduling choices. Grid, block, warp, and thread layout affect occupancy, memory access, and SM utilization.

---

## 3A.8 Tensor Cores: The Inner Loop of AI Compute

Tensor Cores are specialized matrix-multiply-accumulate engines.

They are one of the biggest reasons modern GPUs are effective for AI workloads.

Most transformer arithmetic is matrix multiplication:

```text
C = A × B + C
```

Tensor Cores accelerate this operation for supported datatypes and tile shapes.

Examples include:

- FP16
- BF16
- TF32
- FP8
- INT8
- FP4-related paths on newer Blackwell-class systems

But peak Tensor Core throughput is not automatic.

A workload must satisfy requirements such as:

- Supported datatype
- Compatible dimensions
- Efficient layout
- Proper alignment
- Good memory access pattern
- Kernel path that actually emits Tensor Core instructions
- Enough work to amortize overhead

[REPRESENTATIVE] Tensor Core efficiency depends on datatype, matrix dimensions, tile shape, memory layout, and kernel selection. Poorly aligned dimensions may trigger padding, a less efficient kernel path, or reduced Tensor Core utilization depending on framework, library, and hardware generation.

Do not say:

```text
Misaligned shapes always fall back to CUDA cores.
```

Safer:

```text
Misaligned shapes can reduce Tensor Core efficiency or cause framework/kernel-dependent padding, fallback, or slower kernel selection.
```

---

## Figure Placeholder — Fig 3A.6

```markdown
![Fig 3A.6 — Tensor Core / WGMMA Conceptual Data Path](../assets/diagrams/svg/ch03a_fig_3a_6_tensor_core_wgmma_concept.svg)

**Fig 3A.6 — Tensor Core / WGMMA Conceptual Data Path.** Tensor Cores accelerate matrix multiply-accumulate operations by operating on tiles of input matrices. High AI throughput depends on using supported datatypes, aligned dimensions, efficient tiling, and kernel paths that dispatch Tensor Core instructions.
```

**Figure intro:**  
Tensor Cores are the hardware reason modern GPUs can deliver enormous AI throughput. But peak Tensor Core throughput is only available when the workload uses compatible datatypes, shapes, layouts, and kernels.

**Figure explanation:**  
A GEMM that maps cleanly to Tensor Core tiles can approach high throughput. A GEMM with poor alignment, unsupported dtype, excessive layout conversion, or a fallback kernel may use much less of the theoretical peak. Hopper and later GPUs also introduce more advanced matrix instructions and scheduling patterns such as WGMMA, which should be introduced conceptually before deep kernel work.

> **Key Takeaway:** Tensor Core peak TFLOPS is not automatic. The kernel must use the right datatype, tile shape, alignment, memory layout, and instruction path.

---

## Table 3A.3 — Tensor Core Datatype Support by GPU Generation

| Generation | Example GPUs | Representative Tensor Core Formats | Production Note |
|---|---|---|---|
| Volta | V100 | FP16 | `[SHIPPED]` Introduced Tensor Cores for FP16 mixed-precision matrix math |
| Ampere | A100 | TF32, FP16, BF16, INT8, sparsity paths | `[SHIPPED]` Important generation for mixed-precision training and TF32 |
| Hopper | H100/H200 | FP8, BF16, FP16, TF32, INT8, Transformer Engine | `[SHIPPED]` Key generation for FP8 LLM training/inference |
| Blackwell | B100/B200/GB200 family | FP4/FP6/FP8 and newer Tensor Core paths as applicable | `[SHIPPED/ANNOUNCED]` Product-specific; verify exact SKU and release status |

The practical lesson is not to memorize this table forever. The practical lesson is that precision choice, quantization strategy, and hardware generation must be evaluated together.

> **Key Takeaway:** Precision is a hardware decision as much as a model decision. If the datatype does not map to fast matrix-engine paths, theoretical model efficiency may not become real system throughput.

---

## 3A.9 Dense vs Sparse TFLOPS: The H100 Example

This is one of the most common hardware-number mistakes.

Vendor tables often list Tensor Core throughput with sparsity enabled. That number is not the dense model peak.

For H100 SXM5 BF16:

[DERIVED FROM SHIPPED]

```text
Sparse BF16 peak ≈ 1,978.9 TFLOPS
Dense BF16 peak  ≈ 1,978.9 / 2
                 ≈ 989.4 TFLOPS
```

Use:

```text
H100 SXM5 dense/non-sparse BF16 Tensor Core peak ≈ 989.4 TFLOPS
```

Use separately:

```text
H100 SXM5 sparse BF16 Tensor Core peak ≈ 1,978.9 TFLOPS
```

Do not write:

```text
H100 BF16 peak = 1,979 TFLOPS
```

unless you clearly say:

```text
with structured sparsity
```

### Why This Matters

If you use sparse peak for dense model MFU, you understate your utilization by 2×.

Example:

```text
Measured dense BF16 throughput = 500 TFLOPS

Using dense peak:
MFU = 500 / 989.4 ≈ 50.5%

Using sparse peak incorrectly:
MFU = 500 / 1978.9 ≈ 25.3%
```

The same workload looks half as efficient because the wrong denominator was used.

> **Key Takeaway:** Always compare dense workloads to dense peak and sparse workloads to sparse peak.

---

## 3A.10 Hardware Reference Values Used in This Book

The values below are included for Chapter 3A reasoning. Treat Appendix A as the canonical hardware reference table.

| GPU / Accelerator | Memory | Bandwidth | Dense BF16 Peak | Sparse BF16 Peak | Label |
|---|---:|---:|---:|---:|---|
| H100 SXM5 | 80 GB HBM3 | 3.35 TB/s | ≈989.4 TFLOPS | ≈1,978.9 TFLOPS | `[SHIPPED]` / `[DERIVED FROM SHIPPED]` |
| H200 | 141 GB HBM3e | 4.8 TB/s | Hopper-generation compute; verify exact SKU table | Hopper-generation sparse/dense distinction applies | `[SHIPPED]` |
| MI300X | 192 GB HBM3 | ≈5.3 TB/s | ≈1,307.4 TFLOPS | ≈2,614.9 TFLOPS | `[SHIPPED]` |
| B200 / GB200 family | Product-specific | Product/system/rack-specific | Product-specific | Product-specific | `[SHIPPED/ANNOUNCED]` |

[SHIPPED] AMD Instinct MI300X peak theoretical dense BF16 performance is approximately 1,307.4 TFLOPS. With structured sparsity, AMD lists approximately 2,614.9 TFLOPS. Compare dense-to-dense and sparse-to-sparse values only.

[SHIPPED/ANNOUNCED] Blackwell values should be stated by product: B200 GPU, DGX B200 8-GPU system, GB200 Superchip, or GB200 NVL72 rack. Do not quote a single “B200 number” without saying which product and whether the number is per GPU, per system, or per rack.

---

## 3A.11 NVLink and NVSwitch: The Intra-Node Fabric

Single-GPU performance is not enough for large models.

When a model is partitioned across GPUs, the interconnect becomes part of the model execution path.

Tensor parallelism, pipeline parallelism, expert parallelism, context parallelism, KV movement, and collective operations all depend on communication.

NVLink and NVSwitch provide high-bandwidth GPU-to-GPU communication inside a node or local domain.

[SHIPPED] H100 SXM5 systems list up to 900 GB/s aggregate NVLink bandwidth per GPU. When comparing this value with PCIe or InfiniBand, always state whether the number is per-direction, bidirectional aggregate, or effective application bandwidth.

---

## Figure Placeholder — Fig 3A.7

```markdown
![Fig 3A.7 — NVLink / NVSwitch Domain](../assets/diagrams/png_300dpi/ch03a_fig_3a_7_nvlink_nvswitch_domain.png)

**Fig 3A.7 — NVLink / NVSwitch Domain.** High-bandwidth intra-node GPU-to-GPU fabric enables fast tensor-parallel and collective communication. The NVLink/NVSwitch domain is a key boundary for performance-sensitive multi-GPU workloads.
```

**Figure intro:**  
Single-GPU performance is not enough for large models. When a model is partitioned across GPUs, the interconnect becomes part of the model execution path. Tensor parallelism, pipeline parallelism, KV movement, and collective operations all depend on GPU-to-GPU bandwidth and latency.

**Figure explanation:**  
NVLink and NVSwitch provide much higher GPU-to-GPU bandwidth than PCIe-only paths. This is why HGX/SXM-class systems are often preferred for large multi-GPU training and tensor-parallel serving. However, topology guidance should remain workload-specific. Not every workload needs the highest-bandwidth fabric.

> **Key Takeaway:** Multi-GPU AI performance depends on topology. A model split across GPUs is only as fast as the compute, memory, and communication schedule together.

---

## 3A.12 PCIe, InfiniBand, and Directionality

Bandwidth numbers are often misleading because they mix:

- Per-direction bandwidth
- Bidirectional aggregate bandwidth
- Per-link bandwidth
- Per-GPU aggregate bandwidth
- Per-system aggregate bandwidth
- Line rate
- Effective application bandwidth

A principal architect always asks:

```text
What exactly does this bandwidth number mean?
```

[DERIVED FROM SHIPPED] PCIe 5.0 x16 is commonly treated as approximately 64 GB/s per direction, or approximately 128 GB/s aggregate bidirectional theoretical bandwidth. Sustained bandwidth depends on platform topology, NUMA placement, payload size, DMA path, and software stack.

[SHIPPED] NDR InfiniBand is commonly described as 400 Gb/s, or about 50 GB/s line rate per direction before protocol overhead. Effective NCCL throughput is lower and depends on message size, topology, congestion, routing, and overlap.

[SHIPPED/ANNOUNCED] 800 Gb/s InfiniBand or Ethernet-class products correspond to about 100 GB/s line rate per direction before overhead. Treat product availability and effective bandwidth as platform-specific.

---

## Table 3A.4 — GPU Interconnect Comparison

| Interconnect | Typical Scope | What It Connects | Production Use | Watchouts |
|---|---|---|---|---|
| PCIe | Host/device and sometimes GPU-GPU path | CPU ↔ GPU, GPU ↔ NIC, sometimes GPU ↔ GPU | General attachment, lower-cost systems, development | Lower bandwidth than NVLink; topology matters |
| NVLink | Intra-node GPU fabric | GPU ↔ GPU | Tensor parallelism, fast collectives, large model serving/training | SKU and generation specific |
| NVSwitch | Intra-node or rack-scale switch fabric | Many GPUs through switched NVLink | All-to-all GPU communication within a domain | System design and generation specific |
| InfiniBand / Ethernet fabric | Inter-node | GPU nodes ↔ GPU nodes | Distributed training, multi-node serving | Latency, congestion, collectives, overlap |
| GPUDirect RDMA | GPU ↔ NIC path | GPU memory ↔ network | Reduces CPU staging overhead | Requires platform and driver support |

The important distinction is scope.

PCIe is not the same role as NVLink. NVLink is not the same role as InfiniBand. A strong architecture decision maps each workload communication pattern to the correct interconnect tier.

> **Key Takeaway:** Always ask what the link connects and whether that link sits on the critical path of the workload.

---

## 3A.13 PCIe vs SXM/HGX vs OAM Form Factors

GPU form factor is not just packaging.

It changes:

- Power delivery
- Cooling design
- Physical density
- Local topology
- GPU-to-GPU bandwidth
- Serviceability
- Cost structure
- Platform availability

### PCIe GPUs

PCIe GPUs are flexible and widely deployable.

They are useful for:

- Development
- Single-GPU inference
- Smaller models
- Cost-sensitive serving
- Workloads that do not require tight GPU-to-GPU collectives
- Environments where standard server compatibility matters

### SXM / HGX Systems

SXM/HGX-class systems are designed for dense multi-GPU performance.

They are usually preferred when:

- The workload requires high-bandwidth GPU-to-GPU communication.
- Tensor parallelism is central.
- Training jobs require fast collectives.
- Power and cooling infrastructure can support high-density accelerators.
- NVLink/NVSwitch topology matters.

### OAM-Style Systems

AMD MI300X uses an OAM-style module in high-density accelerator platforms.

The production idea is similar: build a platform where power delivery, cooling, and scale-up fabric are designed around dense accelerator operation.

[REPRESENTATIVE] SXM/HGX-class and OAM-style platforms are usually preferred when dense multi-GPU communication, high power delivery, and high cooling capacity are required. PCIe systems remain valuable for development, single-GPU inference, smaller models, and workloads that do not depend on tight GPU-to-GPU collectives.

---

## Figure Placeholder — Fig 3A.8

```markdown
![Fig 3A.8 — PCIe vs SXM / OAM Form Factors](../assets/diagrams/svg/ch03a_fig_3a_8_pcie_vs_sxm_oam.svg)

**Fig 3A.8 — PCIe vs SXM / OAM Form Factors.** Form factor affects power delivery, cooling, memory bandwidth options, and GPU-to-GPU interconnect topology. PCIe cards are flexible and broadly deployable; SXM/HGX and OAM-style systems are optimized for high-bandwidth multi-GPU configurations.
```

**Figure intro:**  
GPU form factor is not just packaging. It determines power envelope, cooling approach, physical density, and the available communication fabric. This is why the same GPU family may behave differently in PCIe and SXM/HGX-class systems.

**Figure explanation:**  
PCIe systems can be excellent for development, smaller inference, single-GPU serving, or cost-sensitive workloads. SXM/HGX-class and OAM-style systems are usually preferred when the workload requires high-bandwidth GPU-to-GPU communication, high power delivery, and dense multi-GPU topology.

> **Key Takeaway:** Form factor is an architecture choice. It changes power, cooling, topology, and the communication budget available to the model.

---

## 3A.14 Topology Guidance Without Overclaiming

Avoid absolute rules.

Bad:

```text
Never run tensor parallelism across InfiniBand.
```

Better:

```text
[REPRESENTATIVE] Keep latency-sensitive tensor-parallel groups within the highest-bandwidth local GPU fabric whenever possible. Avoid extending TP across inter-node InfiniBand unless the model partitioning, collective schedule, overlap strategy, and network topology are explicitly designed and measured for it.
```

Why?

Because tensor parallelism often requires frequent collectives inside the model layer. If those collectives cross a slower or more variable network fabric, latency and synchronization can dominate.

But “never” is too strong. Some systems deliberately design around inter-node communication with careful overlap, partitioning, and topology-aware scheduling.

### Safer Topology Rules

| Unsafe Wording | Production Wording |
|---|---|
| “Never TP across IB.” | Avoid latency-sensitive TP across inter-node fabric unless validated. |
| “SXM is mandatory.” | SXM/HGX is usually preferred for high-performance multi-GPU training/serving. |
| “PCIe is too slow for AI.” | PCIe can be excellent when the workload is not dominated by tight GPU-to-GPU collectives. |
| “NVLink fixes scaling.” | NVLink helps local collectives, but algorithms, placement, and overlap still matter. |

> **Key Takeaway:** Topology rules are workload rules. Always validate them with communication traces.

---

## 3A.15 Alternative Accelerators: A Neutral Framework

The AI accelerator market includes multiple families:

- NVIDIA GPUs
- AMD Instinct accelerators
- Google TPUs
- Intel Gaudi-style accelerators
- Custom ASICs and cloud-specific accelerators

The goal is not to declare a universal winner.

The goal is to match the accelerator to the workload and organization.

[REPRESENTATIVE] Accelerator choice should be workload-driven. NVIDIA GPUs often provide the broadest CUDA/NVLink software ecosystem. AMD Instinct systems can offer strong memory capacity and bandwidth, with ROCm/HIP as the software path. TPUs are tightly integrated with cloud/provider software stacks. Gaudi-style systems emphasize Ethernet-oriented accelerator fabrics. The correct choice depends on model fit, precision support, interconnect, kernel maturity, framework support, team expertise, power/cooling, cost, and supply risk.

---

## Table 3A.5 — Accelerator Selection Matrix

| Accelerator Family | Common Strengths | Watchouts | Best-Fit Questions |
|---|---|---|---|
| NVIDIA GPU | Mature CUDA ecosystem, Tensor Cores, NVLink/NVSwitch, broad framework support | Cost, supply, power, vendor lock-in | Do we need strongest CUDA/software support and NVLink topology? |
| AMD CDNA / MI300X | Large HBM capacity, strong BF16/HBM specs, ROCm/HIP ecosystem, OAM/Infinity Fabric | Software maturity varies by workload and framework | Does memory capacity/bandwidth matter more than ecosystem friction? |
| TPU | Cloud-integrated systolic-array model, large-scale training/inference focus | Cloud/provider-specific stack and programming model | Are we aligned with the TPU software and cloud ecosystem? |
| Intel Gaudi-style accelerator | Ethernet-oriented accelerator fabric, cost-sensitive training/inference positioning | Framework/kernel maturity and ecosystem coverage | Does the workload fit the stack and network approach? |

A principal architect evaluates accelerator choice across multiple dimensions:

- Model fit in memory
- Arithmetic intensity
- HBM bandwidth
- Precision support
- Kernel availability
- Compiler maturity
- Interconnect
- Framework support
- Team expertise
- Power and cooling
- Supply and procurement risk
- Cost per token or cost per training run

> **Key Takeaway:** Choose accelerators by workload and system constraints. Peak TFLOPS alone is not an architecture decision.

---

## 3A.16 Reading a GPU Spec Sheet Like a Performance Architect

A spec sheet is not a shopping list.

It is a set of ceilings and constraints.

A weak reading asks:

```text
Which GPU has the biggest TFLOPS number?
```

A strong reading asks:

```text
Which resource limits my workload?
```

For LLM inference, that resource might be:

- HBM capacity
- HBM bandwidth
- KV-cache memory
- Decode latency
- Tensor Core throughput
- Scheduler overhead
- PCIe/NVLink transfer
- Power/cooling limit
- Cost per token

For distributed training, that resource might be:

- Tensor Core throughput
- HBM bandwidth
- Activation memory
- Optimizer state memory
- NVLink bandwidth
- InfiniBand bisection bandwidth
- Checkpoint I/O
- Failure recovery time
- Power envelope

---

## Table 3A.6 — How to Read a GPU Spec Sheet Like a Performance Architect

| Step | Question | Why It Matters |
|---:|---|---|
| 1 | Does the model and KV cache fit in HBM? | Capacity determines whether sharding, quantization, or paging is required |
| 2 | Is the workload compute-bound or memory-bound? | Determines whether TFLOPS or bandwidth matters more |
| 3 | What precision will the workload use? | Must map to supported fast Tensor Core or matrix-engine paths |
| 4 | Does the workload require tensor parallelism? | Determines need for high-bandwidth GPU-to-GPU links |
| 5 | What is the local topology? | NVLink/NVSwitch/PCIe affect communication bottlenecks |
| 6 | What is the node-to-node fabric? | Distributed training and large-scale serving depend on network design |
| 7 | Does the software stack support the workload well? | Kernels, compilers, serving frameworks, and drivers matter |
| 8 | What is the power/cooling/TCO impact? | Sustained performance depends on operating conditions and cost |
| 9 | What telemetry will prove success? | Choose metrics before deployment |
| 10 | What is the fallback plan? | Hardware choice affects operational risk |

The framework turns hardware comparison into an engineering process.

A memory-bound decode workload may care more about HBM capacity and bandwidth than peak dense TFLOPS. A tensor-parallel training workload may care more about NVLink topology and NCCL performance. A production service may care most about cost per token and failure behavior.

> **Key Takeaway:** A GPU spec sheet only becomes meaningful after you know the workload regime and the system bottleneck.

---

## 3A.17 Worked Example: Choosing Hardware for 70B Inference

Suppose you need to serve a 70B dense model.

Start with memory:

```text
BF16 weights ≈ 70B × 2 bytes = 140 GB
FP8 weights  ≈ 70B × 1 byte  = 70 GB
INT4 weights ≈ 70B × 0.5 byte = 35 GB
```

Now compare hardware:

| Hardware | Memory Impact | Serving Implication |
|---|---|---|
| H100 80 GB | FP8 70B weights may fit, but KV headroom is tight | Good for TP or quantized serving, but KV planning is critical |
| H200 141 GB | FP8 weights leave significant KV headroom | Stronger 70B serving fit due to larger HBM |
| MI300X 192 GB | Large memory footprint can reduce sharding pressure | Strong candidate for memory-heavy serving if software stack fits |
| B200/GB200 family | Product-specific larger/faster memory and newer precision support | Verify exact product, availability, software stack, and cost |

This example shows why “TFLOPS” is not enough. The serving system might be limited by memory capacity before it is limited by compute.

---

## 3A.18 Worked Example: Why Tensor Parallelism Needs Local Fabric

Tensor parallelism splits layers across GPUs.

That means some operations need communication inside every layer or every few operations.

For example:

```text
Layer GEMM shard on GPU 0
Layer GEMM shard on GPU 1
...
Combine partial results with collective communication
```

If that communication is frequent and latency-sensitive, it should stay on the fastest local fabric when possible.

[REPRESENTATIVE] Keep latency-sensitive tensor-parallel groups within NVLink/NVSwitch or equivalent high-bandwidth local fabric whenever possible.

If tensor parallelism crosses a slower inter-node fabric, the communication can dominate.

But do not overgeneralize. Some model schedules can hide communication. Some systems may be built specifically for inter-node parallelism. Always measure.

---

## 3A.19 How to Explain GPU Architecture in a Principal Interview

Do not answer like this:

```text
A GPU has many cores and is faster for parallel tasks.
```

That is true, but too shallow.

A principal-level answer sounds like this:

> I do not treat a GPU as a faster CPU. I treat it as a throughput engine built around SMs, warps, Tensor Cores, and a steep memory hierarchy. The first question is whether the workload is compute-bound, memory-bound, communication-bound, or overhead-bound. For AI workloads, Tensor Core alignment, HBM bandwidth, KV-cache capacity, and NVLink topology often matter more than raw peak TFLOPS alone.

### Scenario 1 — Why Is a GPU Not Just a Fast CPU?

Weak answer:

> A GPU has more cores.

Better answer:

> A CPU is optimized for low-latency execution of a small number of complex threads. A GPU is optimized for high-throughput execution of many simpler threads. The GPU hides latency with massive parallelism and relies on a memory hierarchy and matrix engines to keep throughput high.

### Scenario 2 — Why Do Tensor Cores Matter?

Weak answer:

> Tensor Cores make AI faster.

Better answer:

> Transformers are dominated by matrix multiplications in QKV projections, output projections, and FFN layers. Tensor Cores are specialized matrix-multiply engines that provide far higher throughput than general scalar cores when the workload uses supported datatypes, aligned shapes, and efficient kernels.

### Scenario 3 — Why Can High Occupancy Still Be Slow?

Weak answer:

> Higher occupancy is better.

Better answer:

> Occupancy only tells us how many warps are resident. It does not tell us whether useful hardware units are busy. A high-occupancy kernel can still be memory-bound, divergent, synchronization-heavy, or missing Tensor Core paths. I use occupancy as one diagnostic signal, not the objective.

### Scenario 4 — Why Does HBM Matter for LLM Inference?

Weak answer:

> HBM is fast memory.

Better answer:

> LLM inference, especially decode, repeatedly reads model weights and KV-cache state. If the workload is memory-bandwidth or memory-capacity constrained, HBM determines concurrency, context length, latency, and cost per token. HBM capacity can decide whether tensor parallelism or quantization is required.

### Scenario 5 — Why Does NVLink Matter?

Weak answer:

> NVLink is faster than PCIe.

Better answer:

> NVLink matters when GPU-to-GPU communication is on the critical path. Tensor-parallel workloads need frequent collectives or exchanges inside the model execution loop. A high-bandwidth local fabric reduces the communication penalty and makes multi-GPU partitioning more viable.

### Scenario 6 — When Is PCIe Good Enough?

Weak answer:

> PCIe is slow.

Better answer:

> PCIe can be good enough when the workload does not require frequent GPU-to-GPU collectives: development, single-GPU inference, smaller models, or cost-sensitive serving. PCIe becomes a problem when tight model-parallel communication or high-rate GPU-GPU exchange sits on the critical path.

### Scenario 7 — How Do You Choose H100 vs H200 vs MI300X?

Weak answer:

> Pick the fastest GPU.

Better answer:

> I first classify the workload. If the model or KV cache does not fit, memory capacity dominates. If decode is memory-bound, HBM bandwidth matters. If training requires tensor parallelism, topology matters. H100 has strong Hopper compute and NVLink ecosystem; H200 improves memory capacity and bandwidth; MI300X offers very large HBM capacity and bandwidth with a different software ecosystem. The right answer depends on model size, precision, serving/training regime, software stack, interconnect, power, and cost.

---

## 3A.20 Chapter Cheat Sheet

### CPU vs GPU

```text
CPU: latency-optimized, complex control, fewer powerful cores.
GPU: throughput-optimized, many parallel lanes, high memory bandwidth, matrix engines.
```

### GPU Execution

```text
Grid → Blocks → Warps → Threads
Blocks scheduled on SMs.
Warps execute inside SMs.
```

### Occupancy vs Utilization

```text
Occupancy = resident warps.
Utilization = useful hardware work.
```

### H100 SXM5 Reference

```text
[SHIPPED] H100 SXM5 HBM: 80 GB, 3.35 TB/s
[DERIVED FROM SHIPPED] Dense BF16 peak ≈ 989.4 TFLOPS
[SHIPPED] Sparse BF16 peak ≈ 1,978.9 TFLOPS
[SHIPPED] NVLink aggregate bandwidth per GPU: up to 900 GB/s
```

### H200 Reference

```text
[SHIPPED] H200 HBM3e: 141 GB, 4.8 TB/s
```

### MI300X Reference

```text
[SHIPPED] MI300X HBM3: 192 GB, ≈5.3 TB/s
[SHIPPED] MI300X dense BF16 peak: ≈1,307.4 TFLOPS
```

### PCIe 5.0 x16

```text
[DERIVED FROM SHIPPED] ≈64 GB/s per direction
[DERIVED FROM SHIPPED] ≈128 GB/s aggregate bidirectional theoretical
```

### NDR InfiniBand

```text
[SHIPPED] 400 Gb/s ≈ 50 GB/s line rate per direction before overhead
```

### Spec-Sheet Rule

```text
Do not ask: Which number is biggest?
Ask: Which resource limits my workload?
```

---

## 3A.21 Key Takeaways

1. A GPU is not a faster CPU. It is a throughput machine.
2. CPU architecture optimizes low-latency execution; GPU architecture optimizes parallel throughput.
3. The SM is where GPU execution becomes real: warps, schedulers, CUDA cores, Tensor Cores, registers, shared memory, and load/store units interact there.
4. Warps execute in SIMT style. Divergence wastes lanes.
5. Occupancy is not utilization. It is one diagnostic signal.
6. Tensor Cores are the main source of AI matrix throughput, but peak throughput requires supported datatypes, aligned shapes, efficient layouts, and correct kernel paths.
7. H100 dense BF16 and sparse BF16 peaks must not be mixed: dense ≈989.4 TFLOPS, sparse ≈1,978.9 TFLOPS.
8. HBM capacity and bandwidth often dominate LLM inference economics.
9. H200 improves the HBM capacity/bandwidth story relative to H100.
10. MI300X offers large HBM capacity and strong dense BF16/HBM specs, but the full system choice depends on software and workload fit.
11. NVLink/NVSwitch matter when GPU-to-GPU communication is on the critical path.
12. PCIe, NVLink, NVSwitch, and InfiniBand numbers must state directionality and scope.
13. SXM/HGX, PCIe, and OAM are form-factor and topology decisions, not just packaging.
14. Accelerator selection must be neutral and workload-driven.
15. A spec sheet is meaningful only after the workload regime is known.

---

## 3A.22 Review Questions

### Conceptual

1. Why is a GPU not simply a faster CPU?
2. What is the difference between latency optimization and throughput optimization?
3. What is an SM?
4. What is a warp?
5. What is SIMT execution?
6. Why does warp divergence hurt performance?
7. Why is occupancy not the same as utilization?
8. What role do Tensor Cores play in transformer workloads?
9. Why is Tensor Core peak throughput not automatic?
10. Why does HBM capacity matter for LLM serving?
11. Why does HBM bandwidth matter for decode?
12. Why does NVLink matter for tensor parallelism?
13. When might PCIe be good enough?
14. Why should accelerator comparisons avoid universal rankings?

### Calculation

1. If a 70B model is stored in BF16, approximately how much memory do the weights require?
2. If the same 70B model is stored in FP8, approximately how much memory do the weights require?
3. If H100 sparse BF16 peak is approximately 1,978.9 TFLOPS, what is the approximate dense BF16 peak?
4. If a kernel measures 500 TFLOPS dense BF16 on H100 SXM5, what is its MFU using the dense BF16 peak?
5. PCIe 5.0 x16 is approximately 64 GB/s per direction. What is the aggregate bidirectional theoretical bandwidth?
6. If NDR InfiniBand is 400 Gb/s, approximately how many GB/s is that as line rate before overhead?

### Principal-Level Interview Practice

1. Explain GPU architecture in two minutes to a hiring manager.
2. Explain why HBM can matter more than TFLOPS for LLM inference.
3. Explain dense vs sparse peak TFLOPS using H100 as the example.
4. Explain when you would prefer H200 over H100 for inference.
5. Explain when MI300X’s large memory footprint could be valuable.
6. Explain why tensor parallelism usually wants high-bandwidth local GPU fabric.
7. Explain why “never tensor-parallel across InfiniBand” is too absolute.
8. Explain how you would choose between NVIDIA, AMD, TPU, and Gaudi-style accelerators.
9. Explain what metrics you would collect to prove a GPU system is underutilized.
10. Explain how you would read a GPU spec sheet for a new accelerator generation.

---

## 3A.23 Production Notes for This Chapter

### Figure Assets Needed

| Figure | Status |
|---|---|
| Fig 3A.1 — GPU vs CPU Architecture Comparison | Existing; needs print export |
| Fig 3A.2 — GPU Memory Hierarchy | Existing; needs print export |
| Fig 3A.3 — H100 SM Internal Block Diagram | Existing; needs print export |
| Fig 3A.4 — SIMT Warp Execution | Existing; needs print export |
| Fig 3A.5 — CUDA Thread Hierarchy | Existing; needs print export |
| Fig 3A.6 — Tensor Core / WGMMA Concept | Must be created |
| Fig 3A.7 — NVLink / NVSwitch Domain | Existing; needs print export |
| Fig 3A.8 — PCIe vs SXM / OAM Form Factors | Must be created |

### Table Assets Included

| Table | Status |
|---|---|
| Table 3A.1 — GPU Memory Hierarchy Summary | Included |
| Table 3A.2 — Streaming Multiprocessor Resource Summary | Included |
| Table 3A.3 — Tensor Core Datatype Support by GPU Generation | Included |
| Table 3A.4 — GPU Interconnect Comparison | Included |
| Table 3A.5 — Accelerator Selection Matrix | Included |
| Table 3A.6 — GPU Spec Sheet Decision Framework | Included |

### Source Notes to Add in Final Book

Use official or primary sources for:

- NVIDIA H100 product specification and Hopper architecture whitepaper
- NVIDIA H200 product page / datasheet
- NVIDIA Blackwell / B200 / GB200 product pages
- AMD Instinct MI300X product page / datasheet
- PCI-SIG PCIe 5.0 specification
- NVIDIA/Mellanox InfiniBand product documentation
- CUDA Programming Guide
- cuBLAS / cuBLASLt documentation
- NCCL documentation

---

## 3A.24 Bridge to Chapter 3B

Chapter 3A built the core hardware mental model:

```text
SMs + warps + Tensor Cores + HBM + topology = accelerator performance system
```

Chapter 3B moves from fundamentals to roadmap.

The next question is:

> How do accelerator generations evolve, and which hardware changes actually matter for AI/ML infrastructure architecture?

That is where we compare Ampere, Hopper, Blackwell, Rubin, CDNA generations, HBM evolution, NVLink evolution, and the hardware roadmap signals that stay stable across product cycles.
