# Chapter 2 Technical Validation Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Ch02 — *The Transformer Architecture: A Performance Engineer’s Deep Dive*  
**Target file:** `publishing/validation/ch02_technical_validation.md`  
**Production status:** Draft validation plan for `production-v1.0`  
**Last reviewed:** 2026-04-30  
**Purpose:** Validate the numerical claims, formulas, architecture assumptions, and confidence labels used in Chapter 2 before production rewriting.

---

## 0. Executive Summary

Chapter 2 is technically strong, but it must avoid overclaiming. The safest production strategy is:

1. Treat the transformer as a performance workload, not only an ML architecture.
2. Use standard formulas for attention, Q/K/V shapes, FLOPs, and KV-cache reductions.
3. Label model-specific examples clearly.
4. Avoid universal claims such as “all modern models use GQA” or “GQA has zero quality loss.”
5. Keep LLaMA-3 70B details tied to a source and avoid assuming undocumented internal architecture values unless verified from configuration or technical reports.
6. Treat MLA as DeepSeek-specific or architecture-specific, not as a universal replacement for GQA.
7. Treat long-context RoPE settings as model-specific, not universally valid.

High-priority production corrections:

| Topic | Production Rule |
|---|---|
| Attention formula | Use the standard scaled dot-product attention formula |
| Q/K/V shapes | Define notation before formulas |
| MHA/MQA/GQA memory math | Derive from number of KV heads |
| MLA | Describe as latent compression of KV state, architecture-specific |
| RoPE | Explain mechanism and long-context adaptations carefully |
| RMSNorm | Explain it omits re-centering and uses RMS scaling |
| Pre-Norm | Say “commonly used in modern LLMs,” not “all frontier models” |
| SwiGLU | Explain 3-projection FFN pattern and label FLOP estimates |
| FLOPs/token | Use 2N inference and 6N training as `[ESTIMATED]` dense-transformer approximations |
| GEMM dominance | Use softer language: “dominates dense transformer FLOPs in many common settings” |

---

# 1. Validation Table

---

## 1.1 Scaled Dot-Product Attention Formula

| Field | Validation |
|---|---|
| Claim | Transformer attention uses scaled dot-product attention. |
| Current value or formula | `Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k)) V` |
| Validation status | **Valid.** This is the standard formula from *Attention Is All You Need*. |
| Corrected value or safer wording | None needed, but define dimensions before using the formula. |
| Confidence label | `[SHIPPED]` for historical architecture; formula itself is standard. |
| Source type needed | Original Transformer paper, *Attention Is All You Need* by Vaswani et al. |
| Recommended final wording | `[SHIPPED] The transformer uses scaled dot-product attention: Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k)) V. The scaling factor prevents dot products from growing too large as head dimension increases, which would push softmax into regions with small gradients.` |
| Priority | **P0** |

### Production note

Use this formula only after introducing notation:

```text
Q = query tensor
K = key tensor
V = value tensor
d_k = key/head dimension
```

---

## 1.2 Q/K/V Tensor Shapes

| Field | Validation |
|---|---|
| Claim | Input hidden states are projected into Q, K, and V tensors. |
| Current value or formula | `X [B,S,H] → Q [B,n_h,S,d_head], K/V [B,n_kv,S,d_head]` |
| Validation status | **Valid as a production-friendly convention.** |
| Corrected value or safer wording | Clarify that tensor layout can differ by framework; the logical dimensions are what matter. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | Transformer implementation docs, framework model configs, Llama-style configs. |
| Recommended final wording | `[REPRESENTATIVE] Starting from hidden states X with logical shape [B,S,H], the model projects X into Q, K, and V. Query heads typically have shape [B,n_h,S,d_head]. In MHA, K and V also use n_h heads; in GQA or MQA, K and V use fewer KV heads, n_kv.` |
| Priority | **P0** |

### Production note

Use `n_h` for query heads and `n_kv` for KV heads consistently.

---

## 1.3 Attention Complexity During Prefill

| Field | Validation |
|---|---|
| Claim | Attention during prefill has quadratic sequence-length behavior. |
| Current value or formula | Time/memory pressure includes `S × S` attention scores; complexity commonly described as `O(S²)` with respect to sequence length. |
| Validation status | **Valid.** |
| Corrected value or safer wording | Clarify that exact FLOPs also depend on `B`, `n_h`, and `d_head`; the `O(S²)` statement isolates sequence scaling. |
| Confidence label | `[ESTIMATED]` |
| Source type needed | Transformer paper; FlashAttention paper for memory traffic discussion. |
| Recommended final wording | `[ESTIMATED] During prefill or training, self-attention must compare tokens against other tokens in the prompt. This creates an attention-score structure that scales as O(S²) with sequence length, although exact FLOPs also depend on batch size, number of heads, head dimension, and implementation.` |
| Priority | **P0** |

---

## 1.4 Decode Attention and KV-Cache Bandwidth Wording

| Field | Validation |
|---|---|
| Claim | Decode attention is often bandwidth-sensitive because each generated token attends to the KV cache. |
| Current value or formula | Decode processes one new token and reads previous K/V state. |
| Validation status | **Valid as a representative performance characterization.** |
| Corrected value or safer wording | Avoid saying decode is always memory-bound. It is often KV-cache / weight-read / scheduling sensitive depending on model, batch size, context length, quantization, and implementation. |
| Confidence label | `[REPRESENTATIVE]` or `[ENV-SPECIFIC]` for measured claims |
| Source type needed | vLLM/PagedAttention paper, LLM serving papers, profiler results. |
| Recommended final wording | `[REPRESENTATIVE] During autoregressive decode, each new token reuses prior K/V state. For many LLM serving workloads, this makes decode sensitive to HBM bandwidth, KV-cache layout, KV precision, batching, and scheduler behavior. The exact bottleneck is environment-specific and should be verified with profiling.` |
| Priority | **P0** |

---

## 1.5 MHA/MQA/GQA Memory Reduction Math

| Field | Validation |
|---|---|
| Claim | Reducing KV heads reduces KV-cache memory roughly proportional to the KV-head count. |
| Current value or formula | `KV reduction vs MHA = n_h / n_kv` for GQA-like layouts. |
| Validation status | **Valid as a first-order KV-cache memory derivation.** |
| Corrected value or safer wording | Clarify that this applies to K/V tensor storage, assuming same `d_head`, layers, sequence length, and datatype. |
| Confidence label | `[ESTIMATED]` |
| Source type needed | GQA paper; model config; serving framework documentation. |
| Recommended final wording | `[ESTIMATED] For the same sequence length, layer count, head dimension, and datatype, KV-cache memory is approximately proportional to the number of KV heads. Compared with MHA using n_h KV heads, GQA with n_kv KV heads reduces K/V storage by roughly n_h / n_kv.` |
| Priority | **P0** |

### Example

```text
MHA: 64 query heads, 64 KV heads
GQA: 64 query heads, 8 KV heads

KV reduction = 64 / 8 = 8×
```

```text
MQA: 64 query heads, 1 KV head

KV reduction = 64 / 1 = 64×
```

### Production caution

Do not claim a fixed quality loss percentage unless tied to a specific paper/model/evaluation.

---

## 1.6 MLA Claims and DeepSeek-Specific Wording

| Field | Validation |
|---|---|
| Claim | MLA reduces KV-cache cost by compressing key/value state into a latent representation. |
| Current value or formula | Multi-head Latent Attention stores compressed latent KV-like state rather than full K/V tensors. |
| Validation status | **Valid for DeepSeek-style MLA descriptions.** |
| Corrected value or safer wording | Describe MLA as architecture-specific, not a universal standard. Avoid giving generic 4×–8× reduction unless tied to a model/config. |
| Confidence label | `[REPRESENTATIVE]`; model-specific values can be `[SHIPPED]` if from released config or `[ESTIMATED]` if derived. |
| Source type needed | DeepSeek-V2 / DeepSeek-V3 technical reports. |
| Recommended final wording | `[REPRESENTATIVE] Multi-head Latent Attention (MLA), used in DeepSeek-style architectures, reduces inference-time KV-cache pressure by caching a compressed latent representation rather than full per-head K/V tensors. The exact memory reduction is model-configuration-specific and should not be generalized without showing the dimensions.` |
| Priority | **P1** |

### Production caution

Do not write:

```text
MLA always reduces KV cache by 8×.
```

Write:

```text
MLA can substantially reduce KV-cache pressure, but the exact ratio depends on the latent dimension and model implementation.
```

---

## 1.7 RoPE and Long-Context Claims

| Field | Validation |
|---|---|
| Claim | RoPE encodes position by rotating query/key representations and supports relative-position behavior. |
| Current value or formula | Rotary Position Embedding applies position-dependent rotations to Q/K dimensions. |
| Validation status | **Valid.** |
| Corrected value or safer wording | Avoid mapping specific theta values to exact context lengths unless sourced from a model configuration. |
| Confidence label | `[SHIPPED]` for RoPE as a published method; `[ENV-SPECIFIC]` or `[REPRESENTATIVE]` for long-context adaptations. |
| Source type needed | RoFormer/RoPE paper; model configs; long-context extension papers such as YaRN if used. |
| Recommended final wording | `[SHIPPED] Rotary Position Embedding (RoPE) injects positional information by applying position-dependent rotations to query and key representations. Many modern decoder-only LLMs use RoPE or RoPE variants. Long-context behavior depends on model configuration, scaling method, training recipe, and extrapolation strategy.` |
| Priority | **P1** |

### Production caution

Avoid unsourced claims like:

```text
theta=10000 means 4K context
theta=500000 means 128K context
```

unless tied to a specific model configuration.

---

## 1.8 RMSNorm vs LayerNorm Claims

| Field | Validation |
|---|---|
| Claim | RMSNorm is a simpler alternative to LayerNorm that uses root-mean-square scaling and omits mean re-centering. |
| Current value or formula | `RMSNorm(x) = x / RMS(x) * gamma` |
| Validation status | **Valid.** |
| Corrected value or safer wording | Avoid saying RMSNorm is always faster; say it can reduce normalization work and is commonly used in LLMs. |
| Confidence label | `[SHIPPED]` for method; `[ENV-SPECIFIC]` for speed claims. |
| Source type needed | RMSNorm paper; framework docs such as PyTorch RMSNorm. |
| Recommended final wording | `[SHIPPED] RMSNorm normalizes by the root mean square of the activation vector and does not subtract the mean as LayerNorm does. This makes it simpler than LayerNorm and commonly attractive in LLM architectures, though actual performance depends on kernel implementation and fusion.` |
| Priority | **P1** |

---

## 1.9 Pre-Norm vs Post-Norm Claims

| Field | Validation |
|---|---|
| Claim | Pre-Norm improves training stability for deep transformer stacks compared with Post-Norm. |
| Current value or formula | Pre-Norm places normalization before attention/FFN sublayers. |
| Validation status | **Directionally valid, but avoid universal wording.** |
| Corrected value or safer wording | Say “commonly used in modern decoder-only LLMs” rather than “all frontier models use Pre-Norm.” |
| Confidence label | `[REPRESENTATIVE]` or `[ENV-SPECIFIC]` |
| Source type needed | Transformer architecture papers, model technical reports, Llama-style architecture docs. |
| Recommended final wording | `[REPRESENTATIVE] Many modern decoder-only LLMs use a Pre-Norm layout, where normalization is applied before attention and FFN sublayers. This is commonly associated with improved stability in deep transformer training, but exact architecture choices remain model-specific.` |
| Priority | **P1** |

---

## 1.10 SwiGLU / FFN Parameter and FLOP Estimates

| Field | Validation |
|---|---|
| Claim | SwiGLU-style FFNs use gated projections and often involve three linear matrices rather than the classic two-matrix FFN. |
| Current value or formula | Classic FFN: `W1`, activation, `W2`; SwiGLU: gate/up projections plus down projection. |
| Validation status | **Valid.** |
| Corrected value or safer wording | Make clear that dimensions are often adjusted so parameter count/FLOPs are comparable to a traditional FFN. |
| Confidence label | `[SHIPPED]` for method; `[ESTIMATED]` for FLOP/parameter estimates. |
| Source type needed | GLU Variants Improve Transformer; Llama architecture references. |
| Recommended final wording | `[ESTIMATED] SwiGLU replaces the standard activation FFN with a gated structure, typically using an up projection, a gate projection, and a down projection. This can improve model quality, but FLOP and parameter comparisons depend on the chosen intermediate dimension.` |
| Priority | **P0** |

### Production caution

Avoid saying:

```text
SwiGLU always costs 50% more FLOPs.
```

Instead say:

```text
A naive same-width SwiGLU would use more projection work, but model architectures often adjust intermediate width to control total parameter and FLOP budget.
```

---

## 1.11 Dense Transformer Inference FLOPs/Token Approximation

| Field | Validation |
|---|---|
| Claim | Dense decoder-only transformer inference costs approximately `2N` FLOPs per generated token. |
| Current value or formula | `Inference FLOPs/token ≈ 2N` |
| Validation status | **Valid as a rough dense forward-pass approximation.** |
| Corrected value or safer wording | Clarify assumptions and exclusions. |
| Confidence label | `[ESTIMATED]` |
| Source type needed | Transformer FLOP-count references, scaling-law papers, model performance notes. |
| Recommended final wording | `[ESTIMATED] For dense decoder-only transformers, a common mental model is inference FLOPs/token ≈ 2N, where N is the non-embedding parameter count. This approximates the dense forward pass through model weights; exact values vary with architecture, attention implementation, context length, vocabulary projection, sparsity, and implementation details.` |
| Priority | **P0** |

---

## 1.12 Dense Transformer Training FLOPs/Token Approximation

| Field | Validation |
|---|---|
| Claim | Dense decoder-only transformer training costs approximately `6N` FLOPs per token. |
| Current value or formula | `Training FLOPs/token ≈ 6N` |
| Validation status | **Valid as a rough training approximation.** |
| Corrected value or safer wording | Clarify that recomputation and attention/context effects can change the total. |
| Confidence label | `[ESTIMATED]` |
| Source type needed | Transformer training compute references, scaling-law papers. |
| Recommended final wording | `[ESTIMATED] For dense decoder-only transformers, a common rule of thumb is training FLOPs/token ≈ 6N, where N is the non-embedding parameter count. This approximates one forward pass plus backward computation. Activation recomputation, attention terms, sequence length, embeddings, and implementation details can change the exact value.` |
| Priority | **P0** |

---

## 1.13 LLaMA-3 70B Architecture Assumptions if Referenced

| Field | Validation |
|---|---|
| Claim | LLaMA-3 70B is a decoder-only transformer using GQA and 8K context. |
| Current value or formula | 70B parameters, GQA, 8K context, text input/output. |
| Validation status | **Valid when using Meta-published model card level claims.** |
| Corrected value or safer wording | If using detailed internal values such as layers, hidden size, query heads, and KV heads, validate against model config or technical report before publication. |
| Confidence label | `[SHIPPED]` for released model-card claims; `[ESTIMATED]` for derived memory/FLOP calculations. |
| Source type needed | Meta Llama 3 model card, Hugging Face config, Meta technical report. |
| Recommended final wording | `[SHIPPED] Meta Llama 3 70B is a released decoder-only autoregressive transformer model with 70B parameters, 8K context length, and Grouped-Query Attention. Detailed calculations that depend on hidden size, layer count, and head count should reference the specific model configuration used.` |
| Priority | **P0** |

### Production caution

If the chapter uses:

```text
L = 80
H = 8192
n_h = 64
n_kv = 8
d_head = 128
```

verify against the exact released configuration before final publication.

---

## 1.14 “GEMM Dominates Transformer FLOPs” Claim

| Field | Validation |
|---|---|
| Claim | GEMM dominates transformer FLOPs. |
| Current value or formula | Sometimes stated as ~99% of FLOPs. |
| Validation status | **Directionally valid for dense transformer layers, but exact percentage is model- and sequence-dependent.** |
| Corrected value or safer wording | Avoid exact universal percentages. |
| Confidence label | `[ESTIMATED]` or `[REPRESENTATIVE]` |
| Source type needed | FLOP derivation by model configuration; transformer compute analysis. |
| Recommended final wording | `[ESTIMATED] In dense decoder-only transformers, large matrix multiplications in projections and FFN layers usually dominate the FLOP count, especially at typical training and prefill shapes. However, memory-bound operations such as normalization, RoPE, softmax, and KV-cache reads can still dominate latency or bandwidth pressure in specific regimes.` |
| Priority | **P0** |

### Production caution

Replace:

```text
GEMM is 99% of transformer FLOPs.
```

with:

```text
GEMM usually dominates dense transformer FLOPs, but the exact percentage depends on architecture, sequence length, and whether attention, embeddings, and vocabulary projection are included.
```

---

## 1.15 FlashAttention and Memory-Traffic Claims

| Field | Validation |
|---|---|
| Claim | FlashAttention reduces HBM traffic by avoiding materialization of the full attention matrix in HBM. |
| Current value or formula | IO-aware exact attention using tiling between HBM and SRAM/shared memory. |
| Validation status | **Valid.** |
| Corrected value or safer wording | Avoid saying FlashAttention is always mandatory above a fixed sequence length. |
| Confidence label | `[SHIPPED]` for published method; `[ENV-SPECIFIC]` for performance gains. |
| Source type needed | FlashAttention paper; implementation docs; benchmark data. |
| Recommended final wording | `[SHIPPED] FlashAttention is an IO-aware exact attention algorithm that tiles attention computation to reduce reads and writes between HBM and on-chip memory, avoiding materialization of the full attention matrix in HBM. The speedup depends on sequence length, head dimension, hardware, implementation, and workload shape.` |
| Priority | **P0** |

### Production caution

Avoid:

```text
FlashAttention is mandatory above 2K.
```

Safer:

```text
For long sequences, FlashAttention-style kernels are often critical because standard attention can become dominated by memory traffic and attention-matrix materialization.
```

---

## 1.16 Modern Architecture Comparison Table Claims

| Field | Validation |
|---|---|
| Claim | Modern architectures can be compared by attention variant, normalization, activation, RoPE, and context length. |
| Current value or formula | Architecture comparison table includes model families and design choices. |
| Validation status | **Valid conceptually, but each model row must be validated independently.** |
| Corrected value or safer wording | Keep the table conservative and source every row. Avoid stale or speculative claims. |
| Confidence label | `[SHIPPED]` for released model specs; `[ANNOUNCED]` for vendor-announced unreleased features; `[ESTIMATED]` for inferred values. |
| Source type needed | Official model cards, technical reports, config files, vendor papers. |
| Recommended final wording | `[SHIPPED/ANNOUNCED/ESTIMATED as applicable] Architecture comparison tables should be treated as snapshots. Each row must be tied to a model card, configuration file, or technical report, because attention variants, context length, normalization, activation functions, and tokenizer details vary by model release.` |
| Priority | **P0** |

---

## 1.17 Any Claim That Needs Confidence Labels

| Claim Type | Recommended Label | Example |
|---|---|---|
| Released model-card facts | `[SHIPPED]` | Llama 3 70B uses GQA and 8K context |
| Formula derivations | `[ESTIMATED]` | FLOPs/token ≈ 2N or 6N |
| KV-head memory reduction | `[ESTIMATED]` | GQA 64Q/8KV gives 8× KV-head reduction |
| Representative performance regime | `[REPRESENTATIVE]` | Decode is often KV-cache bandwidth-sensitive |
| Measured speedups | `[ENV-SPECIFIC]` | FlashAttention speedup on a specific GPU/model |
| Vendor/model roadmap claims | `[ANNOUNCED]` | Future architecture features |
| Published algorithm names | `[SHIPPED]` if method is published/released | RoPE, RMSNorm, FlashAttention |

### Production note

The confidence label should attach to the claim, not the paragraph.

Example:

```text
[ESTIMATED] For dense decoder-only transformers, inference FLOPs/token ≈ 2N.
```

Better than:

```text
Inference is 2N.
```

---

# 2. Corrected / Safer Wording Blocks

## 2.1 Transformer as a Workload

```markdown
A transformer block is best understood as a mixed-regime workload. QKV projections, output projection, and FFN layers are dominated by large matrix multiplications. Attention introduces sequence-length-dependent data movement. Normalization, RoPE, softmax, and residual operations are lower-FLOP but often memory- or fusion-sensitive. In decode, KV-cache reads become a first-order serving constraint.
```

## 2.2 Attention Formula

```markdown
[SHIPPED] Scaled dot-product attention is:

Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k)) V

where d_k is the key/head dimension. The scaling factor controls the magnitude of the dot products before softmax.
```

## 2.3 Attention Variant Memory

```markdown
[ESTIMATED] For the same sequence length, layer count, head dimension, and datatype, KV-cache memory is approximately proportional to the number of KV heads. If a model moves from MHA with 64 KV heads to GQA with 8 KV heads, the K/V storage term is reduced by roughly 64 / 8 = 8×.
```

## 2.4 MLA

```markdown
[REPRESENTATIVE] Multi-head Latent Attention (MLA), used in DeepSeek-style architectures, reduces inference-time KV-cache pressure by caching a compressed latent representation rather than full per-head K/V tensors. The exact memory reduction is model-configuration-specific.
```

## 2.5 GEMM Dominance

```markdown
[ESTIMATED] In dense decoder-only transformers, large matrix multiplications usually dominate total FLOPs, especially in projection and FFN layers. But performance is not determined by FLOPs alone: normalization, RoPE, softmax, residuals, and KV-cache reads can dominate memory traffic, latency, or fusion behavior in specific regimes.
```

---

# 3. P0 / P1 / P2 Validation Action List

## P0 — Must Fix Before Chapter 2 Production Source

| Task | Action |
|---|---|
| Validate and present scaled dot-product attention formula | Add source-backed formula |
| Define all tensor-shape symbols before formulas | Add Table 2.1 |
| Validate Q/K/V shape notation | Use logical shapes and clarify framework layout variation |
| Validate MHA/MQA/GQA KV reduction math | Add Table 2.3 |
| Avoid universal quality-loss claims for GQA/MQA | Use softer wording |
| Validate LLaMA-3 70B assumptions | Use model card/config references |
| Label 2N/6N FLOPs approximations `[ESTIMATED]` | Add assumptions |
| Soften “GEMM is 99% of FLOPs” | Use model/sequence-dependent wording |
| Validate FlashAttention memory claim | Use FlashAttention paper |
| Add confidence labels to all numeric/model claims | Update source chapter |

## P1 — Strongly Recommended

| Task | Action |
|---|---|
| Add MLA details as DeepSeek-specific | Avoid generalizing |
| Validate RoPE long-context claims | Avoid fixed theta/context mapping without source |
| Add RMSNorm vs LayerNorm wording | Avoid universal speed claims |
| Add Pre-Norm vs Post-Norm wording | Avoid “all models” claims |
| Validate FFN/SwiGLU FLOP estimates | Show assumptions |
| Cross-reference Ch06, Ch07, Ch10, Ch11 | Improve continuity |
| Add architecture comparison source notes | Prevent stale model table |

## P2 — Nice to Have

| Task | Action |
|---|---|
| Add appendix note on shape notation | Helps beginners |
| Add model-config checklist | Useful for future updates |
| Add optional “derive FLOPs by hand” worksheet | Learning asset |
| Add interactive tensor-shape calculator later | Web enhancement |

---

# 4. Source Notes to Add to Chapter 2 or Appendix

Use these source categories in the final book:

| Source Category | Use |
|---|---|
| Original Transformer paper | Scaled dot-product attention formula and Transformer architecture |
| FlashAttention paper | IO-aware attention and reduced HBM reads/writes |
| GQA paper | MQA/GQA quality and memory tradeoff |
| DeepSeek-V2/V3 reports | MLA claims and DeepSeek-specific architecture |
| RoPE / RoFormer paper | Rotary positional embedding mechanism |
| RMSNorm paper / framework docs | RMSNorm definition and comparison to LayerNorm |
| GLU Variants Improve Transformer | SwiGLU/GEGLU/ReGLU FFN variants |
| Meta Llama 3 model card / config | Llama 3 70B GQA/context/model-family claims |
| Model config files | Hidden size, layers, heads, KV heads, intermediate size |
| Scaling-law / FLOP-count references | 2N and 6N FLOPs/token approximations |

---

# 5. Commit Instructions

Save this file as:

```text
publishing/validation/ch02_technical_validation.md
```

Then run:

```powershell
git add publishing\validation\ch02_technical_validation.md
git commit -m "Add Chapter 2 technical validation plan"
git push origin production-v1.0
```

---

# 6. Next Production Step

After committing this validation file, the next production task is:

```text
Create source/chapters/ch02_transformer_architecture.md
```

That source chapter should incorporate:

1. Validated attention formula.
2. Consistent tensor-shape notation.
3. Figure placeholders from the Chapter 2 figure integration plan.
4. MHA/MQA/GQA/MLA comparison table.
5. Safer MLA wording.
6. Safer RoPE long-context wording.
7. MFU/roofline continuity from Chapter 1.
8. Operation-to-regime performance map.
9. Principal interview explanation section.
10. Key takeaways and review questions.
