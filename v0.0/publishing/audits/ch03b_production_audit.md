# Chapter 3B Production Audit Report

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Ch03B — *GPU Architecture Roadmap: NVIDIA and AMD Generations*  
**Audit status:** Baseline production review  
**Overall readiness:** **Good, not yet Production Ready**  
**Recommended next file:** `publishing/audits/ch03b_production_audit.md`  
**Last reviewed:** 2026-04-30

---

## 0. Executive Summary

Chapter 3B is the roadmap companion to Chapter 3A.

Chapter 3A teaches the hardware fundamentals: SMs, Tensor Cores, HBM, NVLink, PCIe, SXM/HGX, OAM, and accelerator-selection logic.

Chapter 3B should teach the reader how to evaluate GPU generations over time.

The current Ch03B structure is strong because it does not merely compare products. It tries to explain what each generation changed:

- NVIDIA Hopper: H100/H200 as the LLM-era training and inference baseline
- NVIDIA Blackwell: B200/GB200/NVL72 as rack-scale AI factory architecture
- NVIDIA Rubin: future-generation roadmap signal
- AMD CDNA 3: MI300X and the importance of large HBM capacity
- AMD CDNA 3+ / CDNA 4: MI325X, MI350X, MI355X and the move toward larger memory and FP4/MXFP formats
- Interconnect evolution: NVLink 4 → NVLink 5, Infinity Fabric, rack-scale fabrics
- Memory evolution: HBM2e → HBM3 → HBM3e → HBM4
- Hardware selection: choose by workload regime, not by marketing number

The chapter is **high value** but also **high risk** because roadmap chapters age quickly. It must use confidence labels aggressively and must distinguish:

```text
shipping product spec
vendor-announced roadmap
inference from roadmap direction
benchmark claim
marketing claim
production engineering guidance
```

The most important production rule for Chapter 3B:

> This chapter should not predict the future. It should teach the reader how to evaluate roadmap claims safely.

Final readiness score: **Good Draft / Production Candidate**.

---

# 1. What Is Strong

## 1.1 Strong Strategic Positioning

Chapter 3B fills a real gap. Most AI infrastructure engineers know the names A100, H100, H200, B200, GB200, MI300X, MI325X, and MI350X, but they often do not know how to interpret generation changes.

The chapter can teach readers to ask:

```text
Did this generation improve compute?
Did it improve memory capacity?
Did it improve memory bandwidth?
Did it improve interconnect?
Did it introduce a new precision format?
Did it improve software-visible capability?
Did it change the system design boundary from GPU to node to rack?
```

That is a principal-level roadmap mindset.

## 1.2 Strong “Four Lenses” Framework

The current chapter concept includes a useful framework for evaluating GPU generations:

1. Compute density
2. Memory architecture
3. Interconnect / scale-up fabric
4. Software ecosystem and deployment maturity

This should be preserved and made the foundation of the chapter.

Recommended improved version:

| Lens | Question |
|---|---|
| Compute density | What math formats and TFLOPS/W changed? |
| Memory architecture | What capacity and bandwidth changed? |
| Interconnect | What communication domain changed: GPU, node, rack, cluster? |
| Software maturity | Can production frameworks actually use the new hardware paths? |

## 1.3 Strong NVIDIA Roadmap Coverage

The current Ch03B draft covers NVIDIA Hopper, Blackwell, and Rubin. This is the correct sequence.

The strongest points are:

- H100 as the LLM-era baseline accelerator
- H200 as a memory refresh rather than a new compute architecture
- Blackwell as a system/rack-scale transition, not only a faster GPU
- GB200/NVL72 as a move toward large NVLink domains
- Rubin as a roadmap topic that must be labeled `[ANNOUNCED]`

## 1.4 Strong AMD Roadmap Coverage

The AMD section is valuable because it teaches a different architectural story:

- MI300X: large HBM capacity and chiplet architecture
- MI325X: memory-capacity/bandwidth refresh
- MI350/MI355: CDNA 4 generation with 288 GB HBM3e, 8 TB/s class memory bandwidth, and MXFP formats
- MI400: roadmap / CDNA Next / CDNA 5 style discussion

The strongest practical idea is:

> AMD’s memory-capacity story matters most when model fit and KV-cache capacity dominate the workload.

That is a useful engineering lens.

## 1.5 Strong Hardware Selection Angle

The chapter is strongest when it helps the reader make decisions:

- H100 vs H200 for inference
- H200 vs B200 for memory-bound workloads
- MI300X/MI325X/MI350X for large-memory serving
- B200/GB200 for rack-scale, NVLink-domain, FP4/FP8-oriented systems
- Future Rubin/MI400 claims as planning signals, not procurement facts

This decision framework should be the final synthesis of the chapter.

---

# 2. What Is Weak or Confusing

## 2.1 Current Chapter May Sound Like Vendor Commentary

Some wording risks sounding like a vendor argument:

```text
NVIDIA’s closed optimized stack vs emerging open ecosystem
```

This is provocative, but it may make the chapter sound like a market-opinion piece rather than a technical production book.

Safer production framing:

```text
The AI accelerator roadmap reflects two different system strategies: vertically integrated GPU/network/software platforms and increasingly open accelerator ecosystems. A performance architect should evaluate both by workload fit, software maturity, memory capacity, interconnect, operational risk, and cost.
```

## 2.2 Shipped vs Announced vs Roadmap Claims Need Separation

Chapter 3B should not mix:

- H100 shipping specs
- H200 shipping specs
- B200 / DGX B200 product specs
- GB200 NVL72 rack-scale specs
- GB300 or Blackwell Ultra updates
- Vera Rubin roadmap
- Rubin Ultra roadmap
- MI325X shipping / product specs
- MI350/MI355 launched specs
- MI400 roadmap
- MI400 inferred architecture details

These must be separated with labels:

| Status | Label |
|---|---|
| Product shipping and listed in official product specs | `[SHIPPED]` |
| Vendor-announced product or platform | `[ANNOUNCED]` |
| Derived from official table | `[DERIVED FROM SHIPPED]` |
| Engineering estimate or example | `[ESTIMATED]` |
| Directional architecture interpretation | `[REPRESENTATIVE]` |
| Cluster-specific benchmark or deployment result | `[ENV-SPECIFIC]` |

## 2.3 Some Values May Already Be Stale

Roadmap chapters become outdated faster than fundamentals. The chapter should avoid making 2024/2025 phrasing sound permanent.

Examples needing refresh or validation:

- “Blackwell is next generation” may now need product-specific status.
- “MI350/MI355 announced” may now be shipping or product-page listed.
- “Rubin next generation” may need updated wording based on latest official NVIDIA roadmap.
- “MI400 2026” should remain roadmap unless specific product pages exist.
- “B200 number” must distinguish B200 GPU, DGX B200, GB200 Superchip, GB200 NVL72, GB300, and Blackwell Ultra.

## 2.4 NVIDIA Blackwell Section Can Confuse Product Levels

Blackwell appears at multiple system levels:

| Product Level | Example |
|---|---|
| GPU | B200 |
| Node / system | DGX B200, HGX B200 |
| Superchip | GB200 |
| Rack-scale | GB200 NVL72 |
| Updated generation | GB300 / Blackwell Ultra where applicable |

The chapter must not quote a single “Blackwell number” without saying the level.

Wrong:

```text
Blackwell has 30 TB HBM.
```

Better:

```text
[SHIPPED] GB200 NVL72 is a 72-GPU rack-scale platform with rack-level HBM capacity and NVLink-domain characteristics. Per-GPU, per-system, and per-rack values must not be mixed.
```

## 2.5 AMD CDNA Section Needs Cleaner Product-Level Separation

The AMD side also needs clean layering:

| Product | Role |
|---|---|
| MI300X | CDNA 3 large-HBM accelerator |
| MI325X | CDNA 3+ / HBM3e refresh |
| MI350X / MI355X | CDNA 4 / MI350 series generation |
| MI400 | Next-generation roadmap / CDNA Next or CDNA 5 depending official wording |
| Helios rack | Rack-scale AMD platform roadmap |

Do not mix MI350 Series platform specs with one exact MI355X-only value unless the product page supports it.

## 2.6 Benchmarks and Comparative Claims Need Guardrails

Claims like:

```text
MI355X is 97–119% of B200 performance.
AMD wins latency-sensitive scenario.
Blackwell is 30x faster.
H200 gives 1.4x decode throughput.
MI300X wins 70B serving.
```

may be useful, but they need strict labels and context.

Recommended treatment:

- Official vendor benchmark: `[ANNOUNCED]` or `[ENV-SPECIFIC]`
- MLPerf result: `[SHIPPED]` for submitted result status, `[ENV-SPECIFIC]` for production expectation
- Derived from HBM bandwidth: `[ESTIMATED]`
- Marketing multiplier: use cautiously and explain benchmark/workload

## 2.7 The Chapter Needs a “Roadmap Reading Rules” Section

Before product discussion, add a clear rules section:

```text
Rule 1: Never compare sparse peak to dense peak.
Rule 2: Never compare GPU-level number to rack-level number.
Rule 3: Never compare line rate to application throughput.
Rule 4: Never assume announced roadmap equals shipping availability.
Rule 5: Never choose hardware from TFLOPS alone.
```

This will make the chapter safer and more useful.

---

# 3. Missing Diagrams or Tables

## 3.1 Existing Diagrams to Use

| Diagram | Existing Source | Status | Recommended Use |
|---|---|---|---|
| AMD MI300X Die Stack — 6 XCDs + HBM3 | `diagrams/diagrams_batch2.html#d15` | Exists | AMD CDNA 3 / MI300X section |
| NVLink Domain — DGX H100 + NVSwitch | `diagrams/diagrams_batch2.html#d14` | Exists | Interconnect evolution section |
| HBM3e Die Stacking Cross-Section vs GDDR6X | `diagrams/diagrams_batch2.html#d13` | Exists | HBM evolution section |
| GPU vs CPU Architecture Comparison | `diagrams/diagrams_batch3.html#d25` | Exists, but mostly Ch03A | Optional cross-reference only |
| H100 SM Internal Block | `diagrams/diagrams_batch1.html#d3` | Exists, but Ch03A | Do not repeat unless needed |

## 3.2 Missing Recommended Figures

| Figure | Status | Recommendation |
|---|---|---|
| Fig 3B.1 — GPU Generation Evaluation Framework | Must create | Four-lens roadmap evaluation framework |
| Fig 3B.2 — NVIDIA Roadmap Timeline: Ampere → Hopper → Blackwell → Rubin | Must create | Central roadmap visual |
| Fig 3B.3 — AMD Roadmap Timeline: CDNA 3 → CDNA 3+ → CDNA 4 → MI400 | Must create | AMD roadmap visual |
| Fig 3B.4 — Memory Evolution: HBM2e → HBM3 → HBM3e → HBM4 | Existing partial / needs dedicated figure | Use existing HBM3e diagram and create roadmap table |
| Fig 3B.5 — NVLink / Infinity Fabric Evolution | Must create or adapt | Compare communication domains |
| Fig 3B.6 — MI300X Die Stack | Exists | Use existing Pack 2 diagram |
| Fig 3B.7 — GPU vs Node vs Rack Numbering Levels | Must create | Prevent Blackwell/GB200 value confusion |
| Fig 3B.8 — Hardware Selection Decision Tree | Must create | Final synthesis figure |

## 3.3 Missing Recommended Tables

| Table | Status | Recommendation |
|---|---|---|
| Table 3B.1 — Roadmap Confidence Labels | Must create | Shipped / announced / estimated / representative |
| Table 3B.2 — NVIDIA Generation Comparison | Must create/validate | A100, H100, H200, B200, GB200/NVL72, Rubin |
| Table 3B.3 — AMD Generation Comparison | Must create/validate | MI300X, MI325X, MI350X, MI355X, MI400 |
| Table 3B.4 — Memory Evolution | Must create | HBM2e, HBM3, HBM3e, HBM4 |
| Table 3B.5 — Interconnect Evolution | Must create | NVLink 3/4/5, Infinity Fabric, rack-scale fabrics |
| Table 3B.6 — Product-Level Numbering Guardrail | Must create | GPU vs system vs rack |
| Table 3B.7 — Hardware Selection Matrix | Must create | Workload → better hardware attributes |
| Table 3B.8 — Claims Requiring Annual Refresh | Must create | Current-as-of tracking |

---

# 4. Where Existing Diagrams Should Be Placed

| Placement | Figure/Table | Source | Purpose |
|---|---|---|---|
| After chapter overview | Fig 3B.1 — Four-Lens GPU Generation Evaluation Framework | Create | Establish analytical framework |
| NVIDIA roadmap section | Fig 3B.2 — NVIDIA Roadmap Timeline | Create | Show Ampere → Hopper → Blackwell → Rubin |
| AMD roadmap section | Fig 3B.3 — AMD Roadmap Timeline | Create | Show CDNA 3 → CDNA 3+ → CDNA 4 → MI400 |
| HBM evolution section | Fig 3B.4 — HBM Evolution | `diagrams/diagrams_batch2.html#d13` + new table | Explain memory generation trend |
| Interconnect evolution section | Fig 3B.5 — NVLink / Infinity Fabric Evolution | Create/adapt | Explain scale-up fabric evolution |
| MI300X architecture section | Fig 3B.6 — MI300X Die Stack | `diagrams/diagrams_batch2.html#d15` | Explain AMD chiplet/HBM strategy |
| Blackwell/GB200 section | Fig 3B.7 — Product-Level Numbering Guardrail | Create | Prevent GPU/node/rack spec confusion |
| Final selection section | Fig 3B.8 — Hardware Selection Decision Tree | Create | Convert roadmap into buying/design logic |

---

# 5. Technical Claims That Need Validation

## 5.1 NVIDIA Claims

| Claim | Risk | Recommended Label |
|---|---|---|
| A100 80GB HBM2e ≈ 2.0 TB/s, BF16 ≈ 312 TFLOPS | Must distinguish dense/sparse and SXM vs PCIe | `[SHIPPED]` |
| H100 SXM5 80GB HBM3, 3.35 TB/s, dense BF16 ≈ 989.4 TFLOPS | Already validated in Ch01/Ch03A | `[SHIPPED]` / `[DERIVED FROM SHIPPED]` |
| H200 141GB HBM3e, 4.8 TB/s | Validate official product page | `[SHIPPED]` |
| H200 is memory refresh, not a new architecture | Valid but interpretive | `[REPRESENTATIVE]` |
| B200 per-GPU memory / bandwidth / FP4 / FP8 | Product-specific; validate exact SKU | `[SHIPPED]` or `[ANNOUNCED]` |
| DGX B200 has 8 Blackwell GPUs and 1,440 GB total HBM | System-level; do not use as per-GPU value | `[SHIPPED]` |
| GB200 NVL72 has 72 GPUs and rack-scale NVLink domain | Rack-level; must not be mixed with per-GPU | `[SHIPPED]` |
| NVLink 5 provides 1.8 TB/s GPU-to-GPU interconnect | Directionality and product-specific | `[SHIPPED]` |
| GB200 NVL72 130 TB/s GPU bandwidth / NVLink domain | Rack-level aggregate | `[SHIPPED]` |
| Rubin / Vera Rubin platform availability | Roadmap / announced | `[ANNOUNCED]` |
| Feynman / future architectures | Roadmap | `[ANNOUNCED]` only if official |

## 5.2 AMD Claims

| Claim | Risk | Recommended Label |
|---|---|---|
| MI300X 192GB HBM3, 5.3 TB/s, dense BF16 ≈ 1,307.4 TFLOPS | Validate against AMD specs | `[SHIPPED]` |
| MI300X die count: 6 XCD + 4 MCD = 10 dies | Important and should be validated | `[SHIPPED]` |
| MI300X “single-GPU 70B BF16 serving” | True for weights-only memory, but KV/runtime overhead matters | `[ESTIMATED]` / `[REPRESENTATIVE]` |
| MI325X 256GB HBM3e, 6.0 TB/s, same compute class | Validate AMD product data | `[SHIPPED]` if product listed |
| MI350 / MI355 288GB HBM3e, 8 TB/s | Validate AMD MI350 Series page | `[SHIPPED]` |
| MI350/MI355 CDNA 4, MXFP4/MXFP6 | Validate AMD product page / CDNA4 whitepaper | `[SHIPPED]` |
| MI400 / Helios rack | Roadmap, not final spec | `[ANNOUNCED]` |
| ROCm parity claims | Workload-dependent and version-dependent | `[REPRESENTATIVE]` / `[ENV-SPECIFIC]` |
| AMD vs NVIDIA benchmark comparison | Benchmark-specific | `[ENV-SPECIFIC]` |

## 5.3 Memory / Interconnect Claims

| Claim | Risk | Recommended Label |
|---|---|---|
| HBM2e → HBM3 → HBM3e bandwidth trend | Valid trend, exact values product-specific | `[REPRESENTATIVE]` |
| HBM4 for future Rubin/MI400 | Roadmap if official; otherwise estimate | `[ANNOUNCED]` / `[ESTIMATED]` |
| NVLink 4 = 900 GB/s H100 | Aggregate/per-GPU directionality needed | `[SHIPPED]` |
| NVLink 5 = 1.8 TB/s Blackwell | Directionality needed | `[SHIPPED]` |
| Infinity Fabric ≈ 896GB/s / >1TB/s | Product-specific and directionality needed | `[SHIPPED]` |
| Rack-scale NVLink domains | Must distinguish product/rack | `[SHIPPED]` / `[ANNOUNCED]` |

---

# 6. Reader-Experience Improvements

## 6.1 Add “Roadmap Reading Rules” at the Beginning

Readers need rules before numbers.

Recommended callout:

```text
Roadmap Reading Rules:
1. Compare dense to dense and sparse to sparse.
2. Compare GPU-level numbers to GPU-level numbers.
3. Compare system-level numbers to system-level numbers.
4. Compare rack-level numbers to rack-level numbers.
5. Separate shipping specs from announced roadmap.
6. Treat benchmark wins as workload-specific, not universal.
7. Do not choose hardware from TFLOPS alone.
```

## 6.2 Add “Product Level” Visual

This is crucial for Blackwell and GB200:

```text
GPU → Board / Module → Node / System → Rack → Cluster
```

Example:

```text
B200 GPU != DGX B200 != GB200 Superchip != GB200 NVL72 rack
```

This should become a visual figure and a table.

## 6.3 Use “What Changed?” Boxes for Each Generation

For every generation, use the same pattern:

```text
What changed?
- Compute
- Memory
- Interconnect
- Precision
- Software
- System design boundary

What did not change?
- Still bounded by arithmetic intensity
- Still limited by HBM for decode
- Still needs workload-aware topology
```

This makes the chapter easier to scan.

## 6.4 Add “Roadmap Signal vs Procurement Fact” Callouts

Example:

```text
Rubin is a roadmap signal, not a procurement-ready spec unless the product SKU and delivery date are verified.
```

Example:

```text
MI400 should be discussed as AMD roadmap direction until final product specifications are published.
```

## 6.5 Reduce Vendor-Opinion Tone

Use neutral wording:

Instead of:

```text
NVIDIA’s closed stack vs AMD’s open ecosystem.
```

Use:

```text
NVIDIA emphasizes vertically integrated GPU, networking, software, and rack-scale systems. AMD emphasizes high-memory accelerators, open software stack momentum, and rack-scale roadmap systems. Both must be evaluated through workload fit and operational maturity.
```

---

# 7. Principal-Level Interview Improvements

Add a section:

```text
How to Discuss GPU Roadmaps in a Principal Interview
```

Suggested answer:

```text
I do not evaluate accelerator roadmaps by peak TFLOPS alone. I look at four lenses: compute density, memory capacity/bandwidth, interconnect domain, and software maturity. H100 was the LLM baseline because it combined Hopper Tensor Cores, HBM3, FP8, and NVLink 4. H200 was mainly a memory refresh that improved inference capacity. Blackwell moves the system boundary toward rack-scale NVLink domains and newer precision formats. AMD’s MI300X and MI350 line emphasize very large HBM capacity and bandwidth, which can be compelling for memory-bound inference and large-context workloads if the software stack fits. For future Rubin or MI400 claims, I separate vendor-announced roadmap from shipping specs.
```

## Interview Scenarios to Add

| Scenario | Principal-Level Answer |
|---|---|
| Why was H100 an inflection point? | FP8/Hopper Tensor Cores, HBM3, NVLink 4, software maturity |
| Why did H200 matter if compute was similar? | Memory capacity and bandwidth changed inference economics |
| Why is Blackwell more than “faster H100”? | New precision paths, NVLink 5, rack-scale NVL domains |
| Why does GB200/NVL72 change system design? | The scale-up domain moves from node-level to rack-level |
| Why is MI300X important? | 192 GB HBM changes model-fit and KV-cache economics |
| Why does MI325X matter? | Memory refresh for capacity/bandwidth-constrained workloads |
| Why does MI350/MI355 matter? | CDNA 4, 288 GB HBM3e, 8 TB/s class bandwidth, MX formats |
| How do you evaluate Rubin or MI400? | Use only vendor-announced facts; label inference clearly |
| How do you choose between NVIDIA and AMD? | Workload fit, memory, interconnect, software stack, operational risk, TCO |
| What is the biggest mistake in roadmap comparison? | Mixing per-GPU, per-system, and per-rack numbers |

---

# 8. Print-Readiness Risks

| Risk | Severity | Fix |
|---|---|---|
| Long generation comparison tables | High | Split NVIDIA and AMD into separate tables |
| Wide roadmap timelines | High | Use landscape appendix or stacked timeline |
| Product-level confusion | High | Add product-level guardrail figure |
| Too many raw specs in prose | High | Move detailed specs to Appendix A |
| Roadmap claims may become stale | High | Add “current as of” and confidence labels |
| Vendor comparison may sound biased | Medium | Use neutral workload-fit language |
| HBM/NVLink values may wrap badly | Medium | Use compact tables |
| Dense benchmark claims | Medium | Move to validation notes or sidebars |
| Figure text may be too small | Medium | Export high-resolution/vector |
| Footnotes may overload print | Low | Use compact source notes |

---

# 9. Web-Readiness Risks

| Risk | Severity | Fix |
|---|---|---|
| Ch03B currently PDF-first | High | Create `chapters/ch03b_gpu_roadmap.html` |
| Tables too wide on mobile | High | Responsive table wrappers |
| Roadmap timelines may overflow | High | Use stacked mobile layout |
| Need per-section anchors | Medium | Generate sidebar TOC |
| Need source refresh date | Medium | Add “Current as of 2026” callout |
| Need chart/figure placeholders | Medium | Use figure cards until final diagrams are exported |
| Need link to Ch03A and Ch04 | Medium | Previous/next navigation |
| Vendor claims require source notes | Medium | Link to Appendix A or validation file |
| Needs visual distinction for labels | Medium | CSS badges for `[SHIPPED]`, `[ANNOUNCED]`, `[ESTIMATED]` |

---

# 10. Final Readiness Score

**Score:** **Good — Not Yet Production Ready**

| Category | Score |
|---|---:|
| Strategic value | 9/10 |
| Technical relevance | 9/10 |
| Chapter structure | 8/10 |
| Reader clarity | 6.5/10 |
| Visual integration | 5/10 |
| Technical validation readiness | 4.5/10 |
| Print readiness | 4.5/10 |
| Web readiness | 5/10 |
| Interview usefulness | 8.5/10 |
| Production readiness | 6/10 |

## Readiness Label

**Good Draft / High-Value Roadmap Chapter**

Chapter 3B has strong content and excellent market relevance, but it is one of the highest-risk chapters because roadmap material changes quickly. It needs strict validation, confidence labels, product-level separation, and neutral wording before final publication.

---

# 11. P0 / P1 / P2 Action List

## P0 — Must Fix Before Production

| Task | Output |
|---|---|
| Add roadmap reading rules | Opening callout |
| Separate shipped vs announced vs inferred claims | Confidence labels throughout |
| Validate A100/H100/H200/B200/GB200 values | `publishing/validation/ch03b_technical_validation.md` |
| Validate MI300X/MI325X/MI350X/MI355X values | Technical validation file |
| Treat Rubin and MI400 as roadmap unless official specs exist | Safer wording |
| Add product-level guardrail: GPU vs system vs rack | Fig 3B.7 and Table 3B.6 |
| Split NVIDIA and AMD comparison tables | Tables 3B.2 and 3B.3 |
| Remove vendor-opinion language | Neutral architecture framing |
| Validate all interconnect values and directionality | Technical validation file |
| Create production Markdown source | `source/chapters/ch03b_gpu_roadmap.md` |

## P1 — Strongly Recommended

| Task | Output |
|---|---|
| Add NVIDIA roadmap timeline | Fig 3B.2 |
| Add AMD roadmap timeline | Fig 3B.3 |
| Add HBM evolution table/figure | Fig 3B.4 / Table 3B.4 |
| Add NVLink/Infinity Fabric evolution table | Table 3B.5 |
| Add hardware selection decision tree | Fig 3B.8 |
| Add principal interview explanation section | New section |
| Cross-reference Ch03A, Ch04, Ch05, Ch10, Ch14, Appendix A | Source navigation |
| Add “current as of” date | Currency control |
| Add source refresh checklist | Maintenance asset |

## P2 — Nice to Have

| Task | Output |
|---|---|
| Add LinkedIn visual from roadmap timeline | Marketing asset |
| Add one-page GPU roadmap cheat sheet | Reader aid |
| Add “how not to compare hardware” sidebar | Practical warning |
| Add scenario examples: 70B serving, 405B inference, training cluster | Applied decision examples |
| Add future-edition refresh template | Publishing maintenance |

---

# 12. Recommended Next Commit

Save this file as:

```text
publishing/audits/ch03b_production_audit.md
```

Then run:

```powershell
git add publishing\audits\ch03b_production_audit.md
git commit -m "Add Chapter 3B production audit"
git push origin production-v1.0
```

---

# 13. Next Production Step

After committing this audit, the next task should be:

```text
Create Chapter 3B figure integration plan
```

Recommended file:

```text
publishing/figure_plans/ch03b_figure_integration_plan.md
```

That plan should cover:

1. Four-lens GPU generation evaluation framework
2. NVIDIA roadmap timeline
3. AMD roadmap timeline
4. HBM evolution
5. NVLink / Infinity Fabric evolution
6. MI300X die stack
7. Product-level numbering guardrail
8. Hardware selection decision tree
9. NVIDIA generation comparison table
10. AMD generation comparison table
