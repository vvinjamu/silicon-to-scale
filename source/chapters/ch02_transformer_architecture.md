# Chapter 2 — The Transformer Architecture: A Performance Engineer’s Deep Dive

> “The transformer architecture is not just the model you optimize. It is the workload specification that determines what your hardware must do.”

---

## Chapter Overview

Chapter 1 introduced the AI/ML performance mindset: classify the workload before optimizing it.

Chapter 2 applies that mindset to the transformer.

The transformer is often introduced as a machine-learning architecture. That is true, but incomplete for performance engineers. From an infrastructure perspective, a transformer is a structured workload made of:

- Large matrix multiplications
- Attention data movement
- Softmax and elementwise operations
- Normalization
- Residual connections
- Feed-forward projections
- Position encoding
- Stateful KV cache behavior during inference
- Communication and partitioning challenges during distributed training

This chapter explains the transformer from the point of view of the GPU, runtime, compiler, serving system, and distributed training stack.

By the end of this chapter, you should be able to:

- Explain a transformer block as a sequence of performance workloads.
- Track tensor shapes through Q, K, and V projections.
- Derive the scaled dot-product attention formula.
- Explain why attention scales as `O(S²)` during training/prefill.
- Explain why decode behaves differently from prefill.
- Compare MHA, MQA, GQA, and MLA from a KV-cache perspective.
- Explain why FFN/SwiGLU layers often dominate dense transformer FLOPs.
- Explain why low-FLOP operations can still matter for latency.
- Describe RMSNorm, LayerNorm, Pre-Norm, and Post-Norm in performance terms.
- Explain the transformer as a workload in a principal-level interview.

---

## 2.0 The Transformer as a Workload

A transformer block is not one operation.

It is a pipeline of operations with very different performance regimes.

Some operations are GEMM-heavy and use Tensor Cores well. Some are memory-bandwidth sensitive. Some create large intermediate tensors. Some are fusion targets. Some dominate training FLOPs. Some dominate inference memory pressure.

The performance engineer’s job is to map each part of the transformer to the hardware resource it stresses.

A simplified decoder-only transformer block looks like this:

```text
Input hidden states
    ↓
RMSNorm / LayerNorm
    ↓
QKV projection
    ↓
Attention
    ↓
Output projection
    ↓
Residual add
    ↓
RMSNorm / LayerNorm
    ↓
FFN / SwiGLU
    ↓
Residual add
    ↓
Output hidden states
```

From a performance point of view:

| Component | What It Looks Like to Hardware |
|---|---|
| QKV projection | GEMM / Tensor Core workload |
| Attention scores | Sequence-length-dependent compute and memory |
| Softmax | Memory/fusion-sensitive operation |
| Attention × V | GEMM-like attention operation |
| Output projection | GEMM |
| RMSNorm / LayerNorm | Memory-bandwidth-sensitive elementwise work |
| RoPE | Low-FLOP elementwise rotation, usually fusion-sensitive |
| FFN / SwiGLU | Large GEMMs, often dominant dense transformer FLOPs |
| Residual add | Memory-bandwidth-sensitive elementwise work |

The rest of this chapter decomposes this pipeline.

---

## Figure Placeholder — Fig 2.1

```markdown
![Fig 2.1 — Transformer Block as a Performance Workload](../assets/diagrams/png_300dpi/ch02_fig_2_1_transformer_block_performance_view.png)

**Fig 2.1 — Transformer Block as a Performance Workload.** A transformer block is a sequence of GEMMs, attention data movement, elementwise operations, normalization, residual adds, and FFN projections. The performance engineer’s job is to map each operation to compute, memory, or communication pressure.
```

**Figure intro:**  
A transformer is not a black box. From a performance point of view, it is a structured pipeline of matrix multiplications, attention operations, normalization, activation functions, and residual updates. Before deriving formulas, it is useful to see the full block as a workload that the GPU must execute.

**Figure explanation:**  
Q/K/V projections and FFN layers are GEMM-heavy and can use Tensor Cores. Normalization, RoPE, softmax, and residual adds are often memory-bandwidth or fusion-sensitive. Attention changes character depending on whether the system is in training, prefill, or decode.

> **Key Takeaway:** A transformer block is not one workload. It is a mix of GEMM-dominated compute, memory-bound elementwise operations, attention data movement, and stateful KV-cache behavior.

---

## 2.1 Notation: Symbols and Shapes

Before deriving attention, we need consistent notation.

The most important tensor in a transformer is the hidden-state tensor:

```text
X ∈ [B, S, H]
```

Where:

- `B` is batch size.
- `S` is sequence length.
- `H` is hidden dimension, also called `d_model`.

Most transformer operations preserve `[B, S, H]`, temporarily reshape it into head dimensions, or expand it inside the feed-forward network.

---

## Table 2.1 — Transformer Symbols and Shapes

| Symbol | Meaning | Typical Example |
|---|---|---|
| `B` | Batch size | Number of sequences processed together |
| `S` | Sequence length / context length | 2K, 4K, 8K, 32K tokens |
| `H` or `d_model` | Hidden dimension | 4096, 8192 |
| `L` | Number of transformer layers | 32, 80 |
| `n_h` | Number of query heads | 32, 64 |
| `n_kv` | Number of KV heads | 1, 8, 16, or equal to `n_h` |
| `d_head` | Per-head dimension | Often 64 or 128 |
| `d_ff` | FFN intermediate dimension | Often 2.7×–4× `H`, depending on architecture |
| `X` | Input hidden states | `[B, S, H]` |
| `Q`, `K`, `V` | Query, Key, Value tensors | Logical shape depends on MHA/MQA/GQA/MLA |

The exact memory layout may differ by framework. For example, one implementation may store heads before sequence length, while another stores sequence length before heads. What matters for performance reasoning is the logical shape and how much data must be computed, moved, cached, or communicated.

> **Key Takeaway:** Most transformer performance problems become easier once you can track how `[B, S, H]` changes through attention and FFN layers.

---

## 2.2 From Hidden States to Q, K, and V

The input to a transformer block is:

```text
X [B, S, H]
```

The model applies learned projections to create three tensors:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Where:

```text
W_Q ∈ [H, n_h × d_head]
W_K ∈ [H, n_kv × d_head]
W_V ∈ [H, n_kv × d_head]
```

For standard multi-head attention:

```text
n_kv = n_h
```

For grouped-query attention:

```text
n_kv < n_h
```

For multi-query attention:

```text
n_kv = 1
```

After projection, the logical shapes are often represented as:

```text
Q [B, n_h, S, d_head]
K [B, n_kv, S, d_head]
V [B, n_kv, S, d_head]
```

[REPRESENTATIVE] Actual tensor layout may vary by framework or kernel. A kernel may prefer `[B, S, n_h, d_head]`, packed QKV, or specialized memory layouts for FlashAttention or paged KV cache.

---

## Mental Shape Checkpoint 1

Suppose:

```text
B = 4
S = 2048
H = 8192
n_h = 64
n_kv = 8
d_head = 128
```

Then:

```text
X shape = [4, 2048, 8192]
Q shape = [4, 64, 2048, 128]
K shape = [4, 8, 2048, 128]
V shape = [4, 8, 2048, 128]
```

This is GQA: many query heads, fewer KV heads.

The KV cache stores K and V, not Q. That means reducing `n_kv` directly reduces KV-cache memory.

---

## Figure Placeholder — Fig 2.2

```markdown
![Fig 2.2 — Q/K/V Projection and Scaled Dot-Product Attention Shapes](../assets/diagrams/svg/ch02_fig_2_2_qkv_attention_shapes.svg)

**Fig 2.2 — Q/K/V Projection and Scaled Dot-Product Attention Shapes.** Input hidden states are projected into Q, K, and V tensors. Attention computes scores from `QKᵀ`, normalizes them with softmax, and applies the result to `V` to produce the attention output.
```

**Figure intro:**  
The scaled dot-product attention formula is compact, but the performance implications are hidden in the tensor shapes. The reader should first see how `[B, S, H]` becomes Q, K, and V, and how the sequence-length dimension creates the attention score matrix.

**Figure explanation:**  
The key performance issue is the `S × S` attention score matrix during training and prefill. As sequence length grows, the score matrix grows quadratically. In decode, the system processes one new token at a time but repeatedly attends to previous tokens through the KV cache.

> **Key Takeaway:** Attention complexity is not only about the formula. It is about tensor shape. The `S × S` score matrix drives prefill cost, while KV-cache reads drive decode cost.

---

## 2.3 Scaled Dot-Product Attention

[SHIPPED] The standard scaled dot-product attention formula is:

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V
```

Where:

- `Q` is the query tensor.
- `K` is the key tensor.
- `V` is the value tensor.
- `d_k` is the key/head dimension.

The scaling term:

```text
1 / sqrt(d_k)
```

prevents dot products from becoming too large as head dimension increases. Without scaling, the softmax can saturate, making gradients less useful during training.

### Step-by-Step

For one batch element and one attention head:

```text
Q [S, d_head]
K [S, d_head]
V [S, d_head]
```

Compute attention scores:

```text
Scores = QKᵀ
Scores shape = [S, S]
```

Scale and normalize:

```text
Weights = softmax(Scores / sqrt(d_head))
Weights shape = [S, S]
```

Apply to values:

```text
Output = Weights V
Output shape = [S, d_head]
```

For multiple heads, this happens across `n_h` query heads.

---

## 2.4 Why Attention Has `O(S²)` Behavior During Training and Prefill

[ESTIMATED] During training or prefill, each token can attend to many previous tokens. The attention score structure grows with sequence length:

```text
Scores shape = [B, n_h, S, S]
```

The `S × S` term is why attention is often described as quadratic in sequence length.

This does not mean attention is always the dominant cost. In many dense decoder-only transformers, FFN and projection GEMMs dominate total FLOPs. But attention becomes increasingly important as context length grows.

### Example

If sequence length doubles:

```text
S → 2S
```

Then attention-score size grows as:

```text
S² → (2S)² = 4S²
```

This quadratic scaling is one of the reasons long context is expensive.

[ESTIMATED] Exact attention FLOPs depend on batch size, number of heads, head dimension, causal masking, kernel implementation, and whether the system materializes intermediate attention matrices.

---

## 2.5 Why Decode Is Different from Prefill

A common mistake is to treat every token the same.

LLM inference has two distinct phases:

1. **Prefill** — process the full prompt.
2. **Decode** — generate one token at a time.

During prefill, the model processes many prompt tokens in parallel. During decode, each step generates one new token and attends to previously cached K/V tensors.

[REPRESENTATIVE] For many LLM serving workloads, decode is sensitive to HBM bandwidth, KV-cache layout, KV precision, batching, and scheduler behavior.

This is why transformer architecture choices such as GQA and MLA matter so much for serving infrastructure.

---

## Table 2.2 — Transformer Operations as Performance Workloads

| Operation | Main Shape | Approximate Cost | Common Regime | Performance Note |
|---|---|---:|---|---|
| QKV projection | `[B·S,H] × [H,projection]` | High | Compute-bound if large enough | GEMM / Tensor Core friendly |
| Attention scores | `QKᵀ → [S,S]` | Grows with `S²` | Compute/memory mixed | Prefill-sensitive |
| Softmax | `[B,n_h,S,S]` | Moderate | Memory/fusion-sensitive | Often fused in attention kernels |
| Attention × V | `[S,S] × [S,d_head]` | High for long `S` | Compute/memory mixed | Avoid materializing full attention |
| Output projection | `[B·S,H] × [H,H]` | High | Compute-bound if large enough | GEMM |
| RMSNorm / LayerNorm | `[B,S,H]` | Low FLOP, high bytes | Memory-bound | Fusion target |
| RoPE | Q/K elementwise rotation | Low | Memory/fusion-sensitive | Should be fused when possible |
| FFN / SwiGLU | `[B·S,H] × [H,d_ff]` and back | Very high | Compute-bound | Often dominant FLOPs |
| Residual add | `[B,S,H]` | Low | Memory-bound | Often fused |

The transformer is a mixed-regime workload. GEMMs, attention, normalization, RoPE, and residuals have different bottlenecks and require different optimizations.

> **Key Takeaway:** Do not optimize “the transformer.” Optimize the specific operation and regime that limits the product metric.

---

## 2.6 Multi-Head Attention, MQA, GQA, and MLA

Attention variants differ primarily in how many K/V heads are stored and reused.

This matters because decode reads K/V state repeatedly.

### Multi-Head Attention, MHA

In standard multi-head attention:

```text
n_kv = n_h
```

Every query head has its own K and V head.

This is the baseline design, but it creates large KV-cache memory requirements during inference.

### Multi-Query Attention, MQA

In multi-query attention:

```text
n_kv = 1
```

All query heads share one K/V head.

This gives the largest KV-cache reduction, but may introduce quality or training tradeoffs depending on the model.

### Grouped-Query Attention, GQA

In grouped-query attention:

```text
1 < n_kv < n_h
```

Multiple query heads share each K/V head.

GQA is a practical compromise: it reduces KV-cache memory and bandwidth pressure while preserving more capacity than MQA.

[SHIPPED] Meta’s Llama 3 70B model card describes the model as using Grouped-Query Attention and an 8K context length. Detailed calculations should be tied to the exact model configuration used.

### Multi-Head Latent Attention, MLA

[REPRESENTATIVE] Multi-head Latent Attention, used in DeepSeek-style architectures, reduces inference-time KV-cache pressure by caching a compressed latent representation rather than full per-head K/V tensors.

Do not treat MLA as a universal replacement for GQA. It is architecture-specific. The exact memory reduction depends on latent dimension, projection structure, and implementation.

---

## Figure Placeholder — Fig 2.4

```markdown
![Fig 2.4 — Attention Variants: MHA, MQA, GQA, and MLA](../assets/diagrams/png_300dpi/ch02_fig_2_4_attention_variants.png)

**Fig 2.4 — Attention Variants: MHA, MQA, GQA, and MLA.** Attention variants differ mainly in how many Key/Value heads are stored and reused. Reducing KV heads reduces KV-cache memory and bandwidth pressure, which directly affects serving capacity.
```

**Figure intro:**  
Multi-head attention is the baseline, but modern LLM serving is heavily shaped by how K and V tensors are stored. MQA, GQA, and MLA are not just model-quality choices; they are infrastructure choices because they change KV-cache size, memory bandwidth pressure, and maximum concurrency.

**Figure explanation:**  
MHA stores separate K/V heads for every query head. MQA shares one KV head across many query heads. GQA groups query heads into a smaller number of KV heads. MLA compresses KV state into a latent representation. These choices become especially important in long-context serving.

> **Key Takeaway:** Attention variants are serving-capacity decisions. Fewer KV heads usually mean less KV-cache memory and lower HBM bandwidth pressure during decode.

---

## Table 2.3 — Attention Variant KV Cache Comparison

| Variant | Query Heads | KV Heads | KV Cache Impact | Typical Tradeoff |
|---|---:|---:|---|---|
| MHA | `n_h` | `n_h` | Baseline KV memory | Strong baseline quality, high KV cost |
| MQA | `n_h` | 1 | Large KV reduction | Maximum sharing, possible quality tradeoff |
| GQA | `n_h` | `n_kv < n_h` | Moderate to large KV reduction | Practical compromise |
| MLA | `n_h` | Latent/compressed KV state | Architecture-specific reduction | More complex, model-specific design |

[ESTIMATED] For the same sequence length, layer count, head dimension, and datatype, KV-cache memory is approximately proportional to the number of KV heads.

For a model with 64 query heads and 8 KV heads:

```text
GQA KV reduction vs MHA = 64 / 8 = 8×
```

For MQA with 64 query heads and 1 KV head:

```text
MQA KV reduction vs MHA = 64 / 1 = 64×
```

The reduction math is straightforward. The model-quality impact is empirical.

> **Key Takeaway:** KV-cache memory scales with KV heads, not query heads. GQA and MLA matter because they reduce the amount of state that must be stored and read during decode.

---

## 2.7 RoPE: Rotary Position Embedding

A transformer needs positional information. Without it, the model sees a sequence as a bag of tokens.

[SHIPPED] Rotary Position Embedding, or RoPE, injects positional information by applying position-dependent rotations to query and key representations.

At a high level:

```text
Q_rotated = rotate(Q, position)
K_rotated = rotate(K, position)
```

The dot product between rotated Q and K then contains relative-position information.

RoPE is attractive because it works naturally inside attention and does not require adding a learned positional embedding table to token embeddings.

### Long-Context Caution

Long-context behavior depends on:

- Model configuration
- RoPE base or theta
- RoPE scaling method
- Training context length
- Fine-tuning recipe
- Extrapolation method
- Inference implementation

[REPRESENTATIVE] Many modern decoder-only LLMs use RoPE or RoPE variants. Do not assume that one theta value universally maps to one exact context length without referencing the model configuration.

> **Key Takeaway:** RoPE is cheap compared with GEMM, but long context is not cheap. Longer context increases attention and KV-cache pressure even if the positional encoding itself is efficient.

---

## 2.8 RMSNorm, LayerNorm, Pre-Norm, and Post-Norm

Normalization helps stabilize training.

Two common normalization choices are LayerNorm and RMSNorm.

### LayerNorm

LayerNorm subtracts the mean and divides by standard deviation-like statistics:

```text
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + epsilon) * gamma + beta
```

### RMSNorm

[SHIPPED] RMSNorm normalizes by the root mean square and does not subtract the mean:

```text
RMSNorm(x) = x / RMS(x) * gamma
```

Because it omits re-centering, RMSNorm is simpler than LayerNorm. It is commonly used in modern LLM architectures.

[ENV-SPECIFIC] Whether RMSNorm is faster in practice depends on kernel implementation, fusion, tensor shape, hardware, and framework.

### Pre-Norm vs Post-Norm

In Post-Norm, normalization is applied after the residual update.

In Pre-Norm, normalization is applied before the attention and FFN sublayers.

[REPRESENTATIVE] Many modern decoder-only LLMs use Pre-Norm layouts, where normalization is applied before attention and FFN sublayers. This is commonly associated with improved stability in deep transformer training, but exact architecture choices are model-specific.

---

## Figure Placeholder — Fig 2.3

```markdown
![Fig 2.3 — Transformer Decoder Block Architecture: Pre-Norm vs Post-Norm](../assets/diagrams/png_300dpi/ch02_fig_2_3_prenorm_postnorm_decoder_block.png)

**Fig 2.3 — Transformer Decoder Block Architecture: Pre-Norm vs Post-Norm.** Modern decoder-only LLMs commonly use normalization before attention and FFN sublayers, improving gradient flow and training stability compared with the original post-norm layout.
```

**Figure intro:**  
Normalization and residual connections are not cosmetic implementation details. They affect gradient flow, training stability, and how very deep transformer stacks can be optimized.

**Figure explanation:**  
In a pre-norm block, the residual path provides a cleaner gradient highway through the network. From a performance perspective, normalization itself is usually memory-bound and often a fusion target.

> **Key Takeaway:** Pre-Norm is a stability-oriented architecture choice. Normalization is also a performance concern because it is often memory-bound and repeated in every layer.

---

## 2.9 FFN and SwiGLU: Where Dense Transformer FLOPs Often Go

A transformer block has two major sublayers:

1. Attention
2. Feed-forward network, FFN

The FFN is often the dominant dense FLOP contributor because it uses large matrix multiplications applied to every token.

A classic FFN looks like:

```text
FFN(x) = W_2 activation(W_1 x)
```

A gated FFN, such as SwiGLU, often uses:

```text
FFN(x) = W_down( activation(W_gate x) ⊙ (W_up x) )
```

This uses:

- An up projection
- A gate projection
- A down projection

[ESTIMATED] SwiGLU-style FFNs can improve model quality, but FLOP and parameter comparisons depend on the chosen intermediate dimension. A naive same-width gated FFN would add projection work, but many production architectures adjust intermediate width to manage total parameter and FLOP budget.

### Why FFN Matters for Hardware

FFN layers are usually large GEMMs.

That means:

- They can use Tensor Cores effectively.
- They are often compute-bound when shapes are large enough.
- They are important for tensor parallelism.
- They strongly influence training FLOP budget.
- They can dominate dense transformer arithmetic even when attention dominates memory pressure at long sequence length.

---

## Figure Placeholder — Fig 2.5

```markdown
![Fig 2.5 — FLOP Distribution Inside a Transformer Layer](../assets/diagrams/svg/ch02_fig_2_5_transformer_flop_distribution.svg)

**Fig 2.5 — FLOP Distribution Inside a Transformer Layer.** Most dense decoder-only transformer FLOPs come from large matrix multiplications, especially the FFN and projection layers. Smaller elementwise operations are often lower-FLOP but may still matter because of memory bandwidth and fusion behavior.
```

**Figure intro:**  
A transformer contains many operations, but they do not contribute equally to FLOPs. Large GEMMs dominate the arithmetic count, while normalization, activation, RoPE, and residual operations may dominate memory traffic or kernel-launch overhead.

**Figure explanation:**  
Low-FLOP operations are not always irrelevant. They may not dominate total FLOPs, but they can hurt performance if they introduce extra memory traffic, graph breaks, or many small kernels.

> **Key Takeaway:** FLOP dominance and performance dominance are not always the same. FFN may dominate math, while memory-bound operations can still limit latency.

---

## 2.10 Dense Transformer FLOPs per Token

Two approximations are useful for dense decoder-only transformers.

[ESTIMATED]

```text
Inference FLOPs/token ≈ 2N
Training FLOPs/token ≈ 6N
```

Where:

```text
N = non-embedding parameter count
```

### Inference Approximation

A dense forward pass through model weights performs roughly one multiply and one add per parameter:

```text
Inference FLOPs/token ≈ 2N
```

This is useful for estimating decode compute, especially when discussing model weight reads and dense matrix operations.

### Training Approximation

Training includes:

1. Forward pass
2. Backward pass through activations
3. Backward pass for weights

A common approximation is:

```text
Training FLOPs/token ≈ 3 × forward pass
                          ≈ 3 × 2N
                          ≈ 6N
```

### Assumptions

These are approximations. Exact FLOPs vary with:

- Architecture
- Embeddings
- Vocabulary projection
- Sequence length
- Attention implementation
- Activation recomputation
- MoE sparsity
- Tensor parallelism
- Precision
- Kernel implementation

Use `2N` and `6N` to start a discussion, not to finish one.

---

## 2.11 Prefill vs Decode: Same Transformer, Different Bottleneck

The same transformer block behaves differently in prefill and decode.

## Table 2.4 — Prefill vs Decode Transformer Workload

| Dimension | Prefill | Decode |
|---|---|---|
| Input shape | Many prompt tokens | One new token per step |
| Parallelism | High across sequence | Limited across time |
| Dominant work | GEMMs and attention over prompt | Weight reads and KV-cache reads |
| Attention behavior | Can be compute-heavy at long sequence | Often memory/KV bandwidth sensitive |
| Main metric | TTFT, prompt throughput | TPOT, tokens/sec, P99 latency |
| Typical bottleneck | Compute or attention memory depending on length | HBM bandwidth, KV layout, batching |
| Optimization examples | FlashAttention, chunked prefill, batching | KV quantization, GQA/MLA, continuous batching |

[REPRESENTATIVE] These regimes vary by model, hardware, precision, context length, batch size, and serving framework.

This distinction is the bridge from Chapter 2 to Chapter 6 and Chapter 11. Chapter 6 covers serving architecture. Chapter 11 covers KV-cache math and memory management.

> **Key Takeaway:** Prefill and decode are different performance regimes. Do not optimize LLM serving as if every token behaves the same way.

---

## 2.12 FlashAttention: Why IO-Aware Attention Matters

[SHIPPED] FlashAttention is an IO-aware exact attention algorithm. Its core idea is to reduce reads and writes between HBM and on-chip memory by tiling attention computation.

Traditional attention may materialize large intermediate attention matrices in HBM:

```text
Scores [B, n_h, S, S]
Softmax output [B, n_h, S, S]
```

For long sequences, this becomes expensive.

FlashAttention avoids materializing the full attention matrix in HBM. Instead, it computes attention in blocks, using on-chip memory more efficiently.

[ENV-SPECIFIC] Speedup depends on sequence length, head dimension, GPU architecture, kernel implementation, precision, masking, and workload shape.

Do not say:

```text
FlashAttention is mandatory above 2K.
```

Safer:

```text
For long sequences, FlashAttention-style kernels are often critical because standard attention can become dominated by memory traffic and attention-matrix materialization.
```

---

## 2.13 Transformer Performance Engineering Map

The transformer is a bridge between model architecture and infrastructure architecture.

## Table 2.5 — Transformer Performance Engineering Map

| Component | Stressed Resource | Common Regime | Optimization Families | Related Chapters |
|---|---|---|---|---|
| QKV projection | Tensor Cores, HBM | Compute-bound if large enough | GEMM tuning, fusion, compiler | Ch07, Ch09 |
| Attention scores | Compute + memory | Mixed | FlashAttention, tiling, sequence partitioning | Ch07 |
| Softmax | HBM, cache | Memory/fusion-sensitive | Fused attention kernels | Ch07, Ch09 |
| KV cache | HBM capacity and bandwidth | Memory-bound in decode | GQA, MLA, quantization, paging | Ch11 |
| FFN / SwiGLU | Tensor Cores | Compute-bound | GEMM efficiency, tensor parallelism | Ch07, Ch10 |
| Norms | HBM bandwidth | Memory-bound | Fusion, persistent kernels | Ch07, Ch09 |
| RoPE | Elementwise memory traffic | Fusion-sensitive | Fuse with Q/K projection or attention | Ch07, Ch09 |
| Residual adds | HBM bandwidth | Memory-bound | Fusion | Ch09 |
| Long context | HBM, attention memory | Memory and scheduling sensitive | FlashAttention, chunked prefill, KV policies | Ch06, Ch11 |
| Distributed transformer | Network, topology | Communication-bound | TP/PP/DP/CP, overlap, NCCL tuning | Ch10, Ch14 |

The table routes each bottleneck to the rest of the book.

If the bottleneck is KV-cache bandwidth, Chapter 11 matters.  
If the bottleneck is FFN GEMM efficiency, Chapter 7 matters.  
If the bottleneck appears only at scale, Chapter 10 and Chapter 14 matter.  
If the bottleneck is hidden in production, Chapter 17 matters.

> **Key Takeaway:** A principal engineer maps model components to system resources. That mapping determines which optimization path is worth pursuing.

---

## 2.14 How to Explain the Transformer as a Workload in a Principal Interview

Do not describe the transformer only as “attention plus feed-forward layers.”

That is correct but not principal-level.

A strong answer sounds like this:

> I do not treat the transformer as a black-box ML model. I decompose it into matrix multiplications, attention data movement, elementwise operations, normalization, and KV-cache state. In training and prefill, large GEMMs dominate and can become compute-bound. In decode, repeated weight and KV-cache reads push the workload toward memory bandwidth limits. Architecture choices like GQA, MLA, SwiGLU, and RoPE directly change memory footprint, FLOP distribution, and serving capacity.

### Scenario 1 — Why Is Transformer Training GPU-Friendly?

Weak answer:

> Transformers use attention and GPUs are good at ML.

Better answer:

> Transformer training contains many large matrix multiplications in QKV projections, output projection, and FFN layers. Those GEMMs expose massive parallelism and can use Tensor Cores efficiently. That is why transformers map well to GPUs during training and prefill.

### Scenario 2 — Why Is Decode Different from Prefill?

Weak answer:

> Decode is slower because it generates one token at a time.

Better answer:

> Decode is autoregressive. Each step processes one new token while reading model weights and prior K/V cache state. That reduces temporal parallelism and often shifts the bottleneck toward memory bandwidth, KV-cache layout, batching, and scheduler behavior.

### Scenario 3 — Why Does GQA Matter?

Weak answer:

> GQA is a better attention variant.

Better answer:

> GQA reduces the number of KV heads relative to query heads. Since KV-cache memory scales with KV heads, GQA can reduce serving memory and decode bandwidth pressure while preserving more modeling capacity than MQA.

### Scenario 4 — Why Does FlashAttention Help?

Weak answer:

> FlashAttention is faster attention.

Better answer:

> FlashAttention is IO-aware. It avoids materializing the full `S × S` attention matrix in HBM and computes attention in tiles using on-chip memory more efficiently. The performance win comes from reducing memory traffic, not changing the mathematical result.

### Scenario 5 — Why Does FFN Dominate FLOPs but Decode Still Feels Memory-Bound?

Weak answer:

> FFN is bigger, so it should dominate everything.

Better answer:

> FFN often dominates arithmetic FLOPs in dense transformer layers, but decode performance depends heavily on reading weights and KV-cache state for each generated token. FLOP dominance and latency dominance are not always the same.

---

## 2.15 Chapter Cheat Sheet

### Core Formula

[SHIPPED]

```text
Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k)) V
```

### Core Shapes

```text
X [B, S, H]
Q [B, n_h, S, d_head]
K [B, n_kv, S, d_head]
V [B, n_kv, S, d_head]
```

### Attention Scaling

[ESTIMATED]

```text
Prefill/training attention score shape ∝ [S, S]
Sequence scaling ∝ O(S²)
```

### Dense Transformer FLOP Approximations

[ESTIMATED]

```text
Inference FLOPs/token ≈ 2N
Training FLOPs/token ≈ 6N
```

### KV-Head Reduction

[ESTIMATED]

```text
KV reduction vs MHA ≈ n_h / n_kv
```

Example:

```text
64 query heads / 8 KV heads = 8× KV-head reduction
```

### Performance Map

| If the bottleneck is... | Start with... |
|---|---|
| GEMM throughput | Ch07 kernels, Ch09 compiler |
| Attention memory traffic | FlashAttention, Ch07 |
| KV-cache memory | Ch11 |
| Decode latency | Ch06 and Ch11 |
| Distributed scaling | Ch10 and Ch14 |
| Production regressions | Ch12 and Ch17 |

---

## 2.16 Key Takeaways

1. A transformer is a performance workload, not just a model architecture.
2. The hidden-state tensor `[B,S,H]` is the starting point for shape reasoning.
3. Q/K/V projections convert hidden states into attention heads.
4. Scaled dot-product attention uses `softmax(QKᵀ / sqrt(d_k)) V`.
5. Prefill and training attention scale quadratically with sequence length through the `S × S` attention structure.
6. Decode behaves differently because it generates one token at a time and repeatedly reads KV-cache state.
7. MHA, MQA, GQA, and MLA are infrastructure decisions because they change KV-cache memory and bandwidth pressure.
8. RoPE is efficient positional encoding, but long context still increases attention and KV-cache pressure.
9. RMSNorm and Pre-Norm are common choices in modern decoder-only LLMs, but implementation details matter.
10. FFN/SwiGLU layers often dominate dense transformer FLOPs, but memory-bound operations can still dominate latency.
11. FlashAttention helps by reducing HBM traffic and avoiding full attention-matrix materialization.
12. Principal engineers map transformer components to system resources before choosing optimization strategies.

---

## 2.17 Review Questions

### Conceptual

1. Why should a performance engineer treat the transformer as a workload specification?
2. What does the tensor shape `[B,S,H]` represent?
3. What are Q, K, and V?
4. Why does scaled dot-product attention divide by `sqrt(d_k)`?
5. Why does prefill attention scale as `O(S²)` with sequence length?
6. Why is decode different from prefill?
7. What is the difference between MHA, MQA, GQA, and MLA?
8. Why does reducing KV heads reduce serving memory pressure?
9. Why can FFN dominate FLOPs while decode remains memory-sensitive?
10. Why is FlashAttention described as IO-aware?

### Calculation

1. If `B=2`, `S=4096`, `H=8192`, what is the shape of `X`?
2. If `n_h=64` and `d_head=128`, what is `n_h × d_head`?
3. If `n_h=64` and `n_kv=8`, what is the KV-head reduction compared with MHA?
4. If a dense model has `N=70B` non-embedding parameters, estimate inference FLOPs/token using `2N`.
5. Using the same model, estimate training FLOPs/token using `6N`.
6. If sequence length doubles, how does the size of the attention-score matrix change?

### Principal-Level Interview Practice

1. Explain the transformer as a workload to a hiring manager in two minutes.
2. A team says the transformer is slow. What are the first five questions you ask?
3. A decode workload is memory-bound. Which transformer architecture choices could help?
4. A training workload has high GEMM throughput but poor scaling. Where do you look next?
5. How would you explain the performance impact of GQA to a systems team?

---

## 2.18 Production Notes for This Chapter

### Figure Assets Needed

| Figure | Status |
|---|---|
| Fig 2.1 — Transformer Block Performance View | Existing; needs print export |
| Fig 2.2 — Q/K/V Projection and Attention Shapes | Must be created |
| Fig 2.3 — Pre-Norm vs Post-Norm Decoder Block | Existing; needs print export |
| Fig 2.4 — MHA vs MQA vs GQA vs MLA | Existing; may need MQA supplement |
| Fig 2.5 — FLOP Distribution by Transformer Operation | Must be created |

### Table Assets Included

| Table | Status |
|---|---|
| Table 2.1 — Transformer Symbols and Shapes | Included |
| Table 2.2 — Transformer Operations as Performance Workloads | Included |
| Table 2.3 — Attention Variant KV Cache Comparison | Included |
| Table 2.4 — Prefill vs Decode Transformer Workload | Included |
| Table 2.5 — Transformer Performance Engineering Map | Included |

### Confidence Labels Used

| Label | Use in Chapter 2 |
|---|---|
| [SHIPPED] | Published architecture methods or released model-card claims |
| [ESTIMATED] | FLOP approximations, complexity models, and KV-head reduction derivations |
| [REPRESENTATIVE] | Common workload behavior and implementation patterns |
| [ENV-SPECIFIC] | Measured performance, speedups, bottlenecks, and runtime behavior |
| [DERIVED FROM SHIPPED] | Calculations from released model or hardware values, if used |

### Source Notes to Add in Final Book

Use official or primary sources for:

- Original Transformer paper
- FlashAttention paper
- Grouped-Query Attention paper
- DeepSeek MLA technical report
- RoPE / RoFormer paper
- RMSNorm paper
- SwiGLU / GLU variants paper
- Llama 3 model card and config
- Transformer FLOP-count references

---

## 2.19 Bridge to Chapter 3A

Chapter 2 decomposed the transformer into operations, shapes, FLOPs, memory traffic, and serving state.

Chapter 3A moves one layer down.

The next question is:

> What does the GPU actually look like underneath these workloads, and why do SMs, Tensor Cores, HBM, NVLink, PCIe, and memory hierarchy determine which transformer operations run efficiently?

That is where model architecture becomes hardware architecture.
