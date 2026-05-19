# Chapter 2 Figure Integration Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Chapter 2 — *The Transformer Architecture: A Performance Engineer’s Deep Dive*  
**Target file:** `publishing/figure_plans/ch02_figure_integration_plan.md`  
**Production status:** Draft integration plan for `production-v1.0`  
**Primary goal:** Make Chapter 2 visually teach the transformer as a performance workload, not merely as an ML architecture.

---

## 0. Integration Strategy

Chapter 2 should visually answer one question:

> What does a transformer force the hardware and infrastructure to do?

The chapter should guide the reader from a full transformer block to the specific operations that dominate AI infrastructure performance:

1. Token embeddings and tensor shapes
2. Q/K/V projections
3. Scaled dot-product attention
4. Multi-head attention variants
5. Normalization and residual flow
6. FFN/SwiGLU compute dominance
7. Prefill vs decode behavior
8. KV cache implications
9. Operation-level performance regimes

The chapter already has strong diagram assets in the repository:

- `diagrams/diagram_03_transformer_pipeline.html`
- `diagrams/diagrams_batch3.html#d23`
- `diagrams/diagrams_batch3.html#d24`

The main production task is to place these figures correctly, create missing shape/FLOP tables, add captions, and prepare print-safe exports.

---

# 1. Proposed Chapter 2 Visual Sequence

Recommended flow:

| Order | Figure/Table | Purpose |
|---:|---|---|
| 1 | Fig 2.1 — Transformer Block Performance View | Establish the transformer as a workload pipeline |
| 2 | Table 2.1 — Transformer Symbols and Shapes | Give readers notation before formulas |
| 3 | Fig 2.2 — Q/K/V Projection and Scaled Dot-Product Attention Shapes | Make attention math visual |
| 4 | Table 2.2 — Operation → Shape → FLOPs → Regime | Convert architecture into performance behavior |
| 5 | Fig 2.3 — Pre-Norm vs Post-Norm Decoder Block | Explain modern decoder block ordering |
| 6 | Fig 2.4 — MHA vs MQA vs GQA vs MLA | Explain KV-cache and serving impact |
| 7 | Table 2.3 — Attention Variant KV Cache Comparison | Quantify MHA/MQA/GQA/MLA memory differences |
| 8 | Fig 2.5 — FLOP Distribution by Transformer Operation | Show FFN/GEMM dominance |
| 9 | Table 2.4 — Prefill vs Decode Transformer Workload | Clarify training/prefill/decode regime differences |
| 10 | Table 2.5 — Transformer Performance Engineering Map | Final synthesis table for architect-level reasoning |

---

# 2. Detailed Figure and Table Plan

---

## Fig 2.1 — Transformer Block Performance View

**Type:** Existing figure  
**Existing source file:** `diagrams/diagram_03_transformer_pipeline.html`  
**Status:** Exists but must be integrated into Chapter 2  
**Recommended source mapping:** `diagrams/diagram_03_transformer_pipeline.html`  
**Recommended web asset:** Existing HTML/SVG diagram  
**Recommended print export:** `assets/diagrams/png_300dpi/ch02_fig_2_1_transformer_block_performance_view.png`  
**Exact section placement:** Immediately after the Chapter 2 overview, before the first formula-heavy section.

### Caption

**Fig 2.1 — Transformer Block as a Performance Workload.**  
A transformer block is a sequence of GEMMs, attention data movement, elementwise operations, normalization, residual adds, and FFN projections. The performance engineer’s job is to map each operation to compute, memory, or communication pressure.

### Intro paragraph before figure

A transformer is not a black box. From a performance point of view, it is a structured pipeline of matrix multiplications, attention operations, normalization, activation functions, and residual updates. Before deriving formulas, it is useful to see the full block as a workload that the GPU must execute.

### Explanation paragraph after figure

The diagram should be used as the chapter’s map. Q/K/V projections and FFN layers are GEMM-heavy and can use Tensor Cores. Normalization, RoPE, softmax, and residual adds are often memory-bandwidth or fusion-sensitive. Attention changes character depending on whether the system is in prefill, training, or decode. The rest of the chapter decomposes this diagram into shapes, FLOPs, memory traffic, and optimization regimes.

### Key takeaway box

> **Key Takeaway:** A transformer block is not one workload. It is a mix of GEMM-dominated compute, memory-bound elementwise operations, attention data movement, and stateful KV-cache behavior.

### Web-readiness status

**Mostly ready.** Existing HTML/SVG diagram is browser-ready.

### Print-readiness status

**Not ready.** Needs 300-DPI PNG or vector PDF export and label-size validation.

### Required fixes before production

- Export to `assets/diagrams/png_300dpi/ch02_fig_2_1_transformer_block_performance_view.png`.
- Verify all labels are readable at 7×10 trim size.
- Add alt text: “Diagram showing a transformer block from input embeddings through QKV projection, attention, output projection, normalization, FFN, and residual connections.”
- Ensure figure caption appears near the figure in PDF and HTML.
- Keep the figure near the opening, not buried later in the chapter.

---

## Table 2.1 — Transformer Symbols and Shapes

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Early in the chapter, before the attention derivation and before any formula using `B`, `S`, `H`, `n_h`, or `d_head`.

### Caption

**Table 2.1 — Transformer Symbols and Shapes.**  
This notation table defines the tensor dimensions used throughout the chapter.

### Proposed table content

| Symbol | Meaning | Typical Example |
|---|---|---|
| `B` | Batch size | Number of sequences processed together |
| `S` | Sequence length / context length | 2K, 4K, 8K, 32K tokens |
| `H` or `d_model` | Hidden dimension | 4096, 8192 |
| `L` | Number of transformer layers | 32, 80 |
| `n_h` | Number of query heads | 32, 64 |
| `n_kv` | Number of KV heads | 1, 8, 16, or equal to `n_h` |
| `d_head` | Per-head dimension | Often 64 or 128 |
| `d_ff` | FFN intermediate dimension | Often 2.7×–4× `H` depending on architecture |
| `X` | Input hidden states | `[B, S, H]` |
| `Q`, `K`, `V` | Query, Key, Value tensors | `[B, n_h, S, d_head]` or KV-head variant |

### Intro paragraph before table

Before deriving attention, the reader needs a stable notation system. The same symbols will appear in shape walkthroughs, FLOP estimates, KV-cache formulas, and distributed parallelism discussions.

### Explanation paragraph after table

The most important shape to remember is the hidden-state tensor: `[B, S, H]`. Most transformer operations either preserve this shape, project it into head dimensions, or expand it temporarily inside the feed-forward network. When performance engineers debug transformer workloads, they usually start by identifying which dimension is growing: batch size, sequence length, hidden dimension, layers, or heads.

### Key takeaway box

> **Key Takeaway:** Most transformer performance problems become easier once you can track how `[B, S, H]` changes through attention and FFN layers.

### Web-readiness status

**Ready after table is authored.** Needs responsive table styling.

### Print-readiness status

**Low to medium risk.** The table is compact but should be checked at 7×10 trim.

### Required fixes before production

- Keep the table concise.
- Use consistent notation across Chapter 2, Chapter 6, Chapter 10, and Chapter 11.
- Avoid mixing `D`, `H`, and `d_model` without explanation.
- Cross-reference Appendix C glossary if possible.

---

## Fig 2.2 — Q/K/V Projection and Scaled Dot-Product Attention Shapes

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch02_fig_2_2_qkv_attention_shapes.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch02_fig_2_2_qkv_attention_shapes.png`  
**Exact section placement:** In the scaled dot-product attention section, immediately before the attention formula.

### Caption

**Fig 2.2 — Q/K/V Projection and Scaled Dot-Product Attention Shapes.**  
Input hidden states are projected into Q, K, and V tensors. Attention computes scores from `QKᵀ`, normalizes them with softmax, and applies the result to `V` to produce the attention output.

### Intro paragraph before figure

The scaled dot-product attention formula is compact, but the performance implications are hidden in the tensor shapes. The reader should first see how `[B, S, H]` becomes Q, K, and V, and how the sequence-length dimension creates the attention score matrix.

### Explanation paragraph after figure

The key performance issue is the `S × S` attention score matrix during training and prefill. As sequence length grows, the score matrix grows quadratically. In decode, the system processes one new token at a time but repeatedly attends to previous tokens through the KV cache. That is why the same attention mechanism can be compute-heavy in prefill and memory-heavy in decode.

### Key takeaway box

> **Key Takeaway:** Attention complexity is not only about the formula. It is about tensor shape. The `S × S` score matrix drives prefill cost, while KV-cache reads drive decode cost.

### Web-readiness status

**Not ready.** Needs SVG or HTML diagram.

### Print-readiness status

**Not ready.** Needs print-safe 300-DPI PNG/vector export.

### Required fixes before production

- Create a clean shape-flow diagram:
  - `X [B,S,H]`
  - `W_Q`, `W_K`, `W_V`
  - `Q [B,n_h,S,d_head]`
  - `K [B,n_kv,S,d_head]`
  - `V [B,n_kv,S,d_head]`
  - `QKᵀ [B,n_h,S,S]`
  - `softmax`
  - output `[B,S,H]`
- Include separate labels for MHA/GQA where KV heads differ from query heads.
- Use short labels and large fonts for print.
- Add alt text.

---

## Table 2.2 — Operation → Shape → FLOPs → Regime

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** After the transformer block and attention shape explanation, before detailed operation-by-operation discussion.

### Caption

**Table 2.2 — Transformer Operations as Performance Workloads.**  
Each transformer operation has a characteristic shape, cost, and bottleneck regime.

### Proposed table content

| Operation | Main Shape | Approximate Cost | Common Regime | Performance Note |
|---|---|---:|---|---|
| QKV projection | `[B·S,H] × [H,3H]` or variant | High | Compute-bound if large enough | GEMM / Tensor Core friendly |
| Attention scores | `QKᵀ` → `[S,S]` | Grows with `S²` | Compute/memory mixed | Prefill-sensitive |
| Softmax | `[B,n_h,S,S]` | Moderate | Memory/fusion-sensitive | Often fused in attention kernels |
| Attention × V | `[S,S] × [S,d_head]` | High for long `S` | Compute/memory mixed | Avoid materializing full attention |
| Output projection | `[B·S,H] × [H,H]` | High | Compute-bound if large enough | GEMM |
| RMSNorm / LayerNorm | `[B,S,H]` | Low FLOP, high bytes | Memory-bound | Fusion target |
| RoPE | Q/K elementwise rotation | Low | Memory/fusion-sensitive | Should be fused when possible |
| FFN / SwiGLU | `[B·S,H] × [H,d_ff]` and back | Very high | Compute-bound | Often dominant FLOPs |
| Residual add | `[B,S,H]` | Low | Memory-bound | Often fused |

### Intro paragraph before table

The transformer becomes easier to optimize when each operation is mapped to a performance regime. Some operations want Tensor Cores. Others want reduced memory traffic or fusion. Others become bottlenecks only at long context or during decode.

### Explanation paragraph after table

This table is the bridge from model architecture to performance engineering. It shows why one profiler trace may contain both compute-bound GEMMs and memory-bound elementwise kernels. It also explains why a single optimization strategy cannot cover the whole transformer.

### Key takeaway box

> **Key Takeaway:** The transformer is a mixed-regime workload. GEMMs, attention, normalization, RoPE, and residuals have different bottlenecks and require different optimizations.

### Web-readiness status

**Ready after table is authored.** Needs horizontal scroll on mobile.

### Print-readiness status

**Medium to high risk.** The table is wide and may need splitting for 7×10 print.

### Required fixes before production

- Consider splitting into two tables:
  1. Operation → Shape → Cost
  2. Operation → Regime → Optimization
- Keep table cell text short.
- Avoid overly long formulas in cells.
- Use print-safe table formatting.

---

## Fig 2.3 — Pre-Norm vs Post-Norm Decoder Block

**Type:** Existing figure  
**Existing source file:** `diagrams/diagrams_batch3.html#d23`  
**Status:** Exists but must be integrated into Chapter 2  
**Recommended print export:** `assets/diagrams/png_300dpi/ch02_fig_2_3_prenorm_postnorm_decoder_block.png`  
**Exact section placement:** In the normalization and residual connections section, after explaining RMSNorm/LayerNorm and before discussing modern decoder-only architecture.

### Caption

**Fig 2.3 — Transformer Decoder Block Architecture: Pre-Norm vs Post-Norm.**  
Modern decoder-only LLMs commonly use normalization before attention and FFN sublayers, improving gradient flow and training stability compared with the original post-norm layout.

### Intro paragraph before figure

Normalization and residual connections are not cosmetic implementation details. They affect gradient flow, training stability, and how very deep transformer stacks can be optimized. The distinction between pre-norm and post-norm is therefore architectural, not merely stylistic.

### Explanation paragraph after figure

In a pre-norm block, the residual path provides a cleaner gradient highway through the network. This helps stabilize deep models. From a performance perspective, normalization itself is usually memory-bound and often a fusion target. The order of operations also affects compiler fusion opportunities and graph structure.

### Key takeaway box

> **Key Takeaway:** Pre-norm is a stability-oriented architecture choice. Normalization is also a performance concern because it is often memory-bound and repeated in every layer.

### Web-readiness status

**Mostly ready.** Existing HTML/SVG diagram is browser-ready.

### Print-readiness status

**Not ready.** Needs print-safe export and readability test.

### Required fixes before production

- Export from batch diagram to high-resolution PNG/vector.
- Ensure text labels are readable in print.
- Add alt text.
- Avoid saying “no modern frontier model uses post-norm” as an absolute statement unless validated.
- Label normalization performance claims `[REPRESENTATIVE]` or `[ENV-SPECIFIC]` where appropriate.

---

## Fig 2.4 — MHA vs MQA vs GQA vs MLA

**Type:** Existing figure, with possible extension  
**Existing source file:** `diagrams/diagrams_batch3.html#d24`  
**Status:** Exists but should be extended or supplemented because the current diagram title emphasizes MHA vs GQA vs MLA; MQA may need explicit table coverage.  
**Recommended print export:** `assets/diagrams/png_300dpi/ch02_fig_2_4_attention_variants.png`  
**Exact section placement:** In the attention variants section, after baseline MHA explanation and before KV-cache memory examples.

### Caption

**Fig 2.4 — Attention Variants: MHA, MQA, GQA, and MLA.**  
Attention variants differ mainly in how many Key/Value heads are stored and reused. Reducing KV heads reduces KV-cache memory and bandwidth pressure, which directly affects serving capacity.

### Intro paragraph before figure

Multi-head attention is the baseline, but modern LLM serving is heavily shaped by how K and V tensors are stored. MQA, GQA, and MLA are not just model-quality choices; they are infrastructure choices because they change KV-cache size, memory bandwidth pressure, and maximum concurrency.

### Explanation paragraph after figure

MHA stores separate K/V heads for every query head. MQA shares one KV head across many query heads. GQA groups query heads into a smaller number of KV heads, creating a practical balance between quality and memory savings. MLA compresses KV state into a latent representation, further changing the memory/computation tradeoff. These choices become especially important in long-context serving.

### Key takeaway box

> **Key Takeaway:** Attention variants are serving-capacity decisions. Fewer KV heads usually mean less KV-cache memory and lower HBM bandwidth pressure during decode.

### Web-readiness status

**Mostly ready.** Existing diagram is browser-ready.

### Print-readiness status

**Not ready.** Needs export and possible simplification.

### Required fixes before production

- Export to 300-DPI PNG or vector.
- Confirm whether MQA is visually represented; if not, cover MQA in Table 2.3.
- Add alt text.
- Label memory-reduction examples as `[DERIVED FROM MODEL CONFIG]` or `[ESTIMATED]`.
- Avoid universal quality claims unless cited.

---

## Table 2.3 — Attention Variant KV Cache Comparison

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Immediately after Fig 2.4.

### Caption

**Table 2.3 — KV Cache Impact of Attention Variants.**  
MHA, MQA, GQA, and MLA differ in the number of KV heads they store, which changes KV-cache capacity and decode bandwidth pressure.

### Proposed table content

| Variant | Query Heads | KV Heads | KV Cache Impact | Typical Tradeoff |
|---|---:|---:|---|---|
| MHA | `n_h` | `n_h` | Baseline KV memory | Strong baseline quality, high KV cost |
| MQA | `n_h` | 1 | Large KV reduction | Maximum sharing, possible quality tradeoff |
| GQA | `n_h` | `n_kv < n_h` | Moderate to large KV reduction | Practical compromise |
| MLA | `n_h` | Latent/compressed KV state | Architecture-specific reduction | More complex, model-specific design |

### Example row note

For a model with 64 query heads and 8 KV heads:

```text
GQA KV reduction vs MHA = 64 / 8 = 8×
```

For MQA with 64 query heads and 1 KV head:

```text
MQA KV reduction vs MHA = 64 / 1 = 64×
```

### Intro paragraph before table

The diagram explains the structure. The table quantifies the serving impact. KV-cache memory is proportional to the number of KV heads, so reducing KV heads can directly increase maximum concurrency or context length.

### Explanation paragraph after table

The exact quality and performance tradeoff depends on the model, training recipe, dataset, context length, and inference stack. The reduction math is straightforward, but the model-quality impact is empirical. Treat quality claims as environment- and model-specific unless backed by a published evaluation.

### Key takeaway box

> **Key Takeaway:** KV-cache memory scales with KV heads, not query heads. GQA and MLA matter because they reduce the amount of state that must be stored and read during decode.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Low to medium risk.** Keep the table compact.

### Required fixes before production

- Validate model-specific examples.
- Label derived KV reductions.
- Avoid precise quality-loss percentages unless sourced.
- Cross-reference Chapter 11 for full KV-cache math.

---

## Fig 2.5 — FLOP Distribution by Transformer Operation

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch02_fig_2_5_transformer_flop_distribution.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch02_fig_2_5_transformer_flop_distribution.png`  
**Exact section placement:** In the FLOP-budget section, before or during the LLaMA-3 70B worked example.

### Caption

**Fig 2.5 — FLOP Distribution Inside a Transformer Layer.**  
Most dense decoder-only transformer FLOPs come from large matrix multiplications, especially the FFN and projection layers. Smaller elementwise operations are often lower-FLOP but may still matter because of memory bandwidth and fusion behavior.

### Intro paragraph before figure

A transformer contains many operations, but they do not contribute equally to FLOPs. Large GEMMs dominate the arithmetic count, while normalization, activation, RoPE, and residual operations may dominate memory traffic or kernel-launch overhead.

### Explanation paragraph after figure

The figure should prevent a common misunderstanding: low-FLOP operations are not always irrelevant. They may not dominate total FLOPs, but they can still hurt performance if they introduce extra memory traffic, graph breaks, or many small kernels. FFN may dominate compute, while norms and RoPE may dominate fusion opportunities.

### Key takeaway box

> **Key Takeaway:** FLOP dominance and performance dominance are not always the same. FFN may dominate math, while memory-bound operations can still limit latency.

### Web-readiness status

**Not ready.** Needs new chart or figure.

### Print-readiness status

**Not ready.** Needs print-safe chart.

### Required fixes before production

- Decide whether to use a pie chart, stacked bar, or proportional block diagram.
- Label values as `[ESTIMATED]` unless calculated from a specific model configuration.
- Avoid universal statements like “FFN is always X%.”
- Use a representative model example, such as dense decoder-only architecture.
- Add alt text.

---

## Table 2.4 — Prefill vs Decode Transformer Workload

**Type:** New table  
**Existing source file:** Can cross-reference Ch06 concepts, but table should exist in Ch02  
**Status:** Must be created  
**Exact section placement:** After attention and KV-cache implications, before closing performance map.

### Caption

**Table 2.4 — Prefill vs Decode: Same Transformer, Different Bottleneck.**  
The transformer behaves differently during prefill and decode. Prefill processes many tokens in parallel; decode generates one token at a time and repeatedly reads model weights and KV cache state.

### Proposed table content

| Dimension | Prefill | Decode |
|---|---|---|
| Input shape | Many prompt tokens | One new token per step |
| Parallelism | High across sequence | Limited across time |
| Dominant work | GEMMs and attention over prompt | Weight reads and KV-cache reads |
| Attention behavior | Can be compute-heavy at long sequence | Often memory/KV bandwidth sensitive |
| Main metric | TTFT, prompt throughput | TPOT, tokens/sec, P99 latency |
| Typical bottleneck | Compute or attention memory depending on length | HBM bandwidth, KV layout, batching |
| Optimization examples | FlashAttention, chunked prefill, batching | KV quantization, GQA/MLA, continuous batching |

### Intro paragraph before table

A transformer block does not have one fixed performance profile. The same architecture behaves differently depending on whether the system is processing a prompt or generating tokens autoregressively.

### Explanation paragraph after table

This distinction is the bridge from Chapter 2 to Chapter 6 and Chapter 11. In prefill, the system can use more parallelism across tokens. In decode, each step depends on previous tokens, and KV-cache state becomes a dominant resource. This is why transformer architecture choices directly affect serving infrastructure.

### Key takeaway box

> **Key Takeaway:** Prefill and decode are different performance regimes. Do not optimize LLM serving as if every token behaves the same way.

### Web-readiness status

**Ready after table is authored.**

### Print-readiness status

**Medium risk.** Table is moderately wide.

### Required fixes before production

- Keep cells concise.
- Cross-reference Chapter 6 for inference scheduling.
- Cross-reference Chapter 11 for KV-cache math.
- Label bottleneck statements `[REPRESENTATIVE]` or `[ENV-SPECIFIC]`.

---

## Table 2.5 — Transformer Performance Engineering Map

**Type:** New / rebuilt synthesis table  
**Existing source file:** Existing content concept in Chapter 2, but should be rebuilt cleanly  
**Status:** Must be created  
**Exact section placement:** Near the end of Chapter 2, before key takeaways and review questions.

### Caption

**Table 2.5 — Transformer Performance Engineering Map.**  
This table maps transformer components to the hardware resource they stress and the optimization families that apply.

### Proposed table content

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

### Intro paragraph before table

This table is the chapter’s synthesis. It connects transformer architecture to the rest of the book: kernels, compiler stack, inference serving, KV cache, distributed training, networking, and observability.

### Explanation paragraph after table

The point is not to memorize every optimization. The point is to route the problem correctly. If the bottleneck is KV-cache bandwidth, Chapter 11 matters. If the bottleneck is FFN GEMM efficiency, Chapter 7 matters. If the bottleneck appears only at scale, Chapter 10 and Chapter 14 matter.

### Key takeaway box

> **Key Takeaway:** A principal engineer maps model components to system resources. That mapping determines which optimization path is worth pursuing.

### Web-readiness status

**Ready after table is authored.** Needs responsive table scrolling.

### Print-readiness status

**High risk.** Wide synthesis table may need landscape layout or split version.

### Required fixes before production

- Consider splitting into two tables for print:
  1. Component → Resource → Regime
  2. Component → Optimization → Related Chapters
- Use short labels.
- Cross-link chapters in HTML version.
- Add print-safe formatting.

---

# 3. Final Figure Numbering Recommendation

Use the following final figure numbering for Chapter 2:

| Number | Asset |
|---|---|
| Fig 2.1 | Transformer Block Performance View |
| Fig 2.2 | Q/K/V Projection and Scaled Dot-Product Attention Shapes |
| Fig 2.3 | Pre-Norm vs Post-Norm Decoder Block |
| Fig 2.4 | MHA vs MQA vs GQA vs MLA |
| Fig 2.5 | FLOP Distribution by Transformer Operation |

Optional:

| Number | Asset |
|---|---|
| Fig 2.6 | Prefill vs Decode Transformer Workload |

Recommendation: implement prefill vs decode as **Table 2.4** first. Add Fig 2.6 later only if Chapter 2 becomes too text-heavy.

---

# 4. Final Table Numbering Recommendation

Use the following table numbering:

| Number | Table |
|---|---|
| Table 2.1 | Transformer Symbols and Shapes |
| Table 2.2 | Operation → Shape → FLOPs → Regime |
| Table 2.3 | Attention Variant KV Cache Comparison |
| Table 2.4 | Prefill vs Decode Transformer Workload |
| Table 2.5 | Transformer Performance Engineering Map |

---

# 5. Required Updates to `publishing/figure_inventory.md`

Add or update these rows:

```markdown
| Figure | Title | Current Asset | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|---|
| Fig 2.1 | Transformer Block Performance View | diagrams/diagram_03_transformer_pipeline.html | Ch02 | Yes | No | Export 300-DPI/vector and caption |
| Fig 2.2 | Q/K/V Projection and Scaled Dot-Product Attention Shapes | TBD | Ch02 | No | No | Create SVG + print export |
| Fig 2.3 | Pre-Norm vs Post-Norm Decoder Block | diagrams/diagrams_batch3.html#d23 | Ch02 | Yes | No | Export 300-DPI/vector and validate labels |
| Fig 2.4 | MHA vs MQA vs GQA vs MLA | diagrams/diagrams_batch3.html#d24 | Ch02 | Yes | No | Export and supplement MQA if needed |
| Fig 2.5 | FLOP Distribution by Transformer Operation | TBD | Ch02 | No | No | Create chart and validate assumptions |
```

Add table tracking if desired:

```markdown
| Table | Title | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|
| Table 2.1 | Transformer Symbols and Shapes | Ch02 | Yes | Needs check | Create in source |
| Table 2.2 | Operation → Shape → FLOPs → Regime | Ch02 | Yes | High risk | Split if too wide |
| Table 2.3 | Attention Variant KV Cache Comparison | Ch02 | Yes | Needs check | Add derived labels |
| Table 2.4 | Prefill vs Decode Transformer Workload | Ch02 | Yes | Needs check | Cross-reference Ch06/Ch11 |
| Table 2.5 | Transformer Performance Engineering Map | Ch02 | Yes | High risk | Consider split print version |
```

---

# 6. Chapter 2 Visual Production Checklist

## Web Checklist

- [ ] Embed Fig 2.1 transformer block diagram near the chapter opening.
- [ ] Add Fig 2.2 Q/K/V shape diagram.
- [ ] Embed Fig 2.3 pre-norm/post-norm diagram.
- [ ] Embed Fig 2.4 attention variants diagram.
- [ ] Add all five tables with responsive scrolling.
- [ ] Add alt text for each figure.
- [ ] Add anchors for figures and tables.
- [ ] Add cross-links to Ch06, Ch07, Ch10, Ch11, and Ch14.
- [ ] Verify mobile readability.

## Print Checklist

- [ ] Export all existing HTML/SVG diagrams as 300-DPI PNG or vector PDF.
- [ ] Validate all labels at 7×10 trim.
- [ ] Keep figures with captions.
- [ ] Prevent figure/caption separation across pages.
- [ ] Split wide tables if needed.
- [ ] Avoid long code/formula lines.
- [ ] Test formulas in PDF layout.

## Technical Validation Checklist

- [ ] Validate attention formula.
- [ ] Validate Q/K/V shape notation.
- [ ] Validate MHA/MQA/GQA KV-head reduction math.
- [ ] Validate MLA wording.
- [ ] Validate RoPE claims.
- [ ] Validate FFN/SwiGLU FLOP assumptions.
- [ ] Validate LLaMA-3 70B example values if used.
- [ ] Label representative claims correctly.
- [ ] Avoid absolute claims without sources.

---

# 7. Recommended Next Commit

After saving this file as:

```text
publishing/figure_plans/ch02_figure_integration_plan.md
```

Run:

```powershell
git add publishing\figure_plans\ch02_figure_integration_plan.md
git commit -m "Add Chapter 2 figure integration plan"
git push origin production-v1.0
```

Then update the master figure inventory:

```powershell
git add publishing\figure_inventory.md
git commit -m "Update figure inventory for Chapter 2"
git push origin production-v1.0
```

---

# 8. Next Production Step After This Plan

The next task should be:

```text
Chapter 2 Technical Validation Plan
```

Recommended file:

```text
publishing/validation/ch02_technical_validation.md
```

The validation should cover:

1. Attention formula
2. Q/K/V shapes
3. `O(S²)` prefill attention complexity
4. Decode attention and KV-cache bandwidth wording
5. MHA/MQA/GQA/MLA memory reduction math
6. MLA claims and DeepSeek-specific wording
7. RoPE/long-context claims
8. FFN/SwiGLU parameter and FLOP estimates
9. LLaMA-3 70B assumptions
10. Modern architecture comparison table
11. Performance-regime labels
12. Confidence labels
