# Chapter 5 Figure Integration Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Ch05 — *Power, Thermal, and AI Data Center Infrastructure*  
**Target file:** `publishing/figure_plans/ch05_figure_integration_plan.md`  
**Production status:** Production Planning Pack  
**Last reviewed:** 2026-04-30

---

## 0. Visual Strategy

Chapter 5 should make the physical system visible:

```text
power in → compute work → heat out → sustained performance
```

The visuals should teach that AI infrastructure is a chain:

```text
grid/substation → switchgear → UPS → PDU → rack → server → GPU → cooling loop → heat rejection
```

The chapter needs mostly new figures because power and thermal are not yet visually covered in earlier chapters.

---

# 1. Proposed Figure/Table Sequence

| Order | Figure/Table | Purpose |
|---:|---|---|
| 1 | Table 5.1 — Power Terminology | Prevent TDP/TBP/system/rack confusion |
| 2 | Fig 5.1 — AI Data Center Power Delivery Chain | Show power path |
| 3 | Fig 5.2 — GPU Power-to-Heat Flow | Explain power becomes heat |
| 4 | Table 5.2 — Accelerator/System Power Reference | Validated power numbers |
| 5 | Fig 5.3 — Air Cooling vs Direct Liquid Cooling | Cooling methods |
| 6 | Table 5.3 — Cooling Method Comparison | Practical pros/cons |
| 7 | Fig 5.4 — Rack Power Density Evolution | Show density trend |
| 8 | Table 5.4 — PUE and Efficiency Metrics | Separate facility efficiency from model efficiency |
| 9 | Fig 5.5 — Power Cap vs Performance Curve | Performance/W tradeoff |
| 10 | Fig 5.6 — Thermal Throttling Feedback Loop | Explain sustained-performance loss |
| 11 | Table 5.5 — Power/Thermal Bottleneck Signals | Diagnostic table |
| 12 | Table 5.6 — Wrong Fix vs Right First Question | Principal mindset |
| 13 | Fig 5.7 — Power/Thermal Observability Stack | Monitoring pipeline |
| 14 | Fig 5.8 — Power-Aware Scheduling Decision Flow | Final synthesis |
| 15 | Table 5.7 — Production Infrastructure Checklist | Operational checklist |

---

# 2. Detailed Figure and Table Plan

---

## Table 5.1 — Power Terminology

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Opening section after “Power in One Page.”

### Caption

**Table 5.1 — Power Terminology for AI Infrastructure.**  
GPU power, board power, system power, rack power, and facility power are different measurement boundaries and must not be mixed.

### Proposed table content

| Term | Boundary | Meaning | Common Mistake |
|---|---|---|---|
| TDP | Device/design thermal boundary | Thermal design target, vendor-specific meaning | Treating it as exact runtime power |
| TBP | Board/module | Board-level power, often used by AMD | Treating it as full server power |
| GPU power | Accelerator | GPU/device power telemetry | Ignoring CPU, fans, NICs, PSUs |
| System power | Server/DGX/HGX node | Full server draw | Treating as per-GPU power |
| Rack power | Rack | Sum of systems, switches, power shelves | Comparing to one server |
| Facility power | Data center | IT + cooling + power overhead | Confusing with IT load |
| PUE | Facility efficiency | Total facility energy / IT equipment energy | Treating it as model efficiency |

### Intro paragraph before table

Before discussing power, define the boundary. A 750 W accelerator, a 14.3 kW AI server, and a 120 kW rack are all valid numbers, but they describe different layers of the infrastructure.

### Explanation paragraph after table

This table prevents the most common power-analysis mistake: comparing values measured at different boundaries. Always ask whether the number is per GPU, per board, per system, per rack, or facility-wide.

### Key takeaway box

> **Key Takeaway:** Power numbers are meaningless without the measurement boundary.

### Web-readiness status

Ready after authoring.

### Print-readiness status

Medium risk; keep compact.

### Required production fixes

- Keep terminology vendor-neutral.
- Avoid implying TDP/TBP definitions are identical across vendors.
- Add confidence-label note.

---

## Fig 5.1 — AI Data Center Power Delivery Chain

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch05_fig_5_1_power_delivery_chain.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch05_fig_5_1_power_delivery_chain.png`  
**Exact section placement:** After Table 5.1.

### Caption

**Fig 5.1 — AI Data Center Power Delivery Chain.**  
Power flows from utility/grid or on-site generation through transformers, switchgear, UPS, PDUs, rack power shelves, server power supplies, voltage regulators, and finally GPUs, CPUs, memory, and networking.

### Intro paragraph before figure

A GPU cluster does not begin at the GPU. It begins at the power source. Every stage in the power chain must be designed for sustained load, redundancy, fault isolation, maintenance, and safety.

### Explanation paragraph after figure

If any link in the chain is undersized or unstable, the result may be throttling, node failures, breaker trips, job interruptions, or reduced cluster availability. AI infrastructure must be planned as an end-to-end power delivery system.

### Key takeaway box

> **Key Takeaway:** A GPU is only as reliable as the power chain that feeds it.

### Web-readiness status

Not ready; new SVG needed.

### Print-readiness status

Not ready; needs print export.

### Required production fixes

- Create simple left-to-right chain.
- Include redundancy/UPS as optional branch.
- Avoid excessive electrical detail.
- Add alt text.

---

## Fig 5.2 — GPU Power-to-Heat Flow

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch05_fig_5_2_power_to_heat_flow.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch05_fig_5_2_power_to_heat_flow.png`  
**Exact section placement:** Section explaining why power and cooling are inseparable.

### Caption

**Fig 5.2 — GPU Power-to-Heat Flow.**  
Almost all electrical power consumed by compute equipment ultimately becomes heat that must be removed by the cooling system.

### Intro paragraph before figure

In a data center, power and thermal are two sides of the same problem. A rack drawing 100 kW is also creating roughly 100 kW of heat that must be removed continuously.

### Explanation paragraph after figure

This is why AI infrastructure design must consider both electrical capacity and heat rejection. A system can be electrically powered but still fail to sustain performance if cooling cannot keep device temperatures within operating limits.

### Key takeaway box

> **Key Takeaway:** Every watt consumed by IT equipment becomes heat the facility must remove.

### Web-readiness status

Not ready.

### Print-readiness status

Not ready.

### Required production fixes

- Create power-in/heat-out Sankey-style diagram.
- Label electrical power, useful compute, heat, and cooling loop.
- Avoid thermodynamics over-detail.
- Add alt text.

---

## Table 5.2 — Accelerator and System Power Reference

**Type:** New validation-sensitive table  
**Existing source file:** None  
**Status:** Must be created after validation  
**Exact section placement:** After Fig 5.2.

### Caption

**Table 5.2 — Accelerator and System Power Reference.**  
Power values must be compared at the same product level: GPU, board, system, rack, or facility.

### Proposed table content

| Product | Level | Power Claim | Confidence |
|---|---|---:|---|
| H100 SXM5 | GPU/module | ~700 W class TDP/TGP depending source/SKU | `[SHIPPED]` |
| H200 SXM | GPU/module | ~700 W class depending SKU | `[SHIPPED]` |
| MI300X | OAM accelerator | 750 W peak TBP | `[SHIPPED]` |
| DGX B200 | System | ~14.3 kW max system power | `[SHIPPED]` |
| GB200 NVL72 / DGX GB200 rack | Rack-scale | ~120 kW rack power consumption | `[SHIPPED]` if official user guide |
| Future Rubin / MI400 rack systems | Rack-scale roadmap | TBD / announced | `[ANNOUNCED]` |

### Intro paragraph before table

This table intentionally mixes product layers only when the layer is explicitly shown. That prevents a GPU-level number from being compared to a full system or rack-level number.

### Explanation paragraph after table

Use the table to teach measurement boundaries, not to rank products. A 750 W OAM accelerator and a 14.3 kW system power number are both valid, but they are not comparable without normalization.

### Key takeaway box

> **Key Takeaway:** Compare power only after identifying whether the value is per GPU, per board, per system, or per rack.

### Web-readiness status

Ready after validation.

### Print-readiness status

High risk; split if necessary.

### Required production fixes

- Validate each number in `ch05_technical_validation.md`.
- Add “current as of 2026 edition.”
- Avoid B200 per-GPU power if not officially sourced.
- Use product-level labels.

---

## Fig 5.3 — Air Cooling vs Direct Liquid Cooling

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch05_fig_5_3_air_vs_liquid_cooling.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch05_fig_5_3_air_vs_liquid_cooling.png`  
**Exact section placement:** Cooling architecture section.

### Caption

**Fig 5.3 — Air Cooling vs Direct Liquid Cooling.**  
Air cooling removes heat through airflow and heat sinks; direct liquid cooling moves heat through cold plates, coolant loops, CDUs, and facility heat rejection systems.

### Intro paragraph before figure

As rack density rises, airflow alone becomes increasingly difficult. Direct liquid cooling moves heat more efficiently from high-power devices but adds plumbing, service, monitoring, and facility integration requirements.

### Explanation paragraph after figure

The figure should show tradeoffs rather than declare one method universally superior. Air cooling can remain suitable for lower-density systems. Liquid cooling becomes increasingly important for dense AI racks and rack-scale systems.

### Key takeaway box

> **Key Takeaway:** Cooling strategy follows rack density, service model, facility design, and reliability requirements.

### Web-readiness status

Not ready.

### Print-readiness status

Not ready.

### Required production fixes

- Show both paths side by side.
- Include cold plate, CDU, facility water loop, rear-door option if simple.
- Avoid overclaiming air cooling limits.
- Add alt text.

---

## Table 5.3 — Cooling Method Comparison

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** After Fig 5.3.

### Caption

**Table 5.3 — Cooling Method Comparison for AI Infrastructure.**  
Cooling approaches differ in density support, complexity, facility requirements, serviceability, and risk.

### Proposed table content

| Cooling Method | Best Fit | Strengths | Watchouts |
|---|---|---|---|
| Air cooling | Lower/mid density racks | Simpler, mature, service-friendly | Airflow limits at high density |
| Rear-door heat exchanger | Transitional high-density racks | Can extend air-cooled facilities | Still facility-dependent |
| Direct-to-chip liquid | Dense GPU servers/racks | Efficient heat removal near devices | Plumbing, leak detection, service model |
| Immersion cooling | Specialized high-density environments | High thermal capacity | Ecosystem, maintenance, compatibility |
| Hybrid liquid/air | Most dense AI platforms | Handles device heat and residual air load | Requires integrated design |

### Intro paragraph before table

Cooling is not a binary decision. Modern AI data centers often use hybrid designs where direct liquid cooling removes most device heat and air cooling handles residual components.

### Explanation paragraph after table

The correct cooling strategy depends on rack density, water availability, maintenance model, reliability requirements, cost, and facility readiness.

### Key takeaway box

> **Key Takeaway:** Cooling is an infrastructure architecture decision, not an afterthought.

### Web-readiness status

Ready after authoring.

### Print-readiness status

Medium risk.

### Required production fixes

- Label thresholds as representative, not universal.
- Avoid claiming immersion is required.
- Add facility caveat.

---

## Fig 5.4 — Rack Power Density Evolution

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch05_fig_5_4_rack_power_density_evolution.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch05_fig_5_4_rack_power_density_evolution.png`  
**Exact section placement:** Rack-scale AI infrastructure section.

### Caption

**Fig 5.4 — Rack Power Density Evolution.**  
AI infrastructure has pushed rack power density from traditional enterprise ranges into tens and, for rack-scale AI systems, around 100 kW or more.

### Intro paragraph before figure

Traditional data center rack power budgets are not enough for the densest AI systems. Rack-scale GPU platforms require a different level of power delivery, cooling capacity, commissioning, and monitoring.

### Explanation paragraph after figure

The figure should use representative ranges, not universal thresholds. It should visually show why 40–60 kW racks and 100 kW+ racks require different cooling and power planning than traditional enterprise racks.

### Key takeaway box

> **Key Takeaway:** Rack density changes the data center design problem.

### Web-readiness status

Not ready.

### Print-readiness status

Not ready.

### Required production fixes

- Use representative ranges.
- Label GB200 NVL72 class as rack-scale example.
- Add current-as-of note.
- Add alt text.

---

## Table 5.4 — PUE and Efficiency Metrics

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Efficiency section.

### Caption

**Table 5.4 — PUE and AI Efficiency Metrics.**  
PUE measures facility overhead, while tokens per watt, samples per watt, and training throughput per watt measure workload efficiency.

### Proposed table content

| Metric | Formula | Measures | Does Not Measure |
|---|---|---|---|
| PUE | Total facility energy / IT equipment energy | Facility infrastructure efficiency | Model efficiency |
| IT power | Server/network/storage power | Compute load | Cooling overhead |
| Tokens/sec/W | Token throughput / power | Inference efficiency | Quality or latency alone |
| Samples/sec/W | Samples throughput / power | Training/inference efficiency | Job completion alone |
| TFLOPS/W | FLOPs / power | Math efficiency | Memory/communication efficiency |
| Cost/token | Cost / output token | Business efficiency | Hardware utilization alone |

### Intro paragraph before table

PUE is useful but often misunderstood. It tells you about facility overhead, not whether a model serving stack is efficient.

### Explanation paragraph after table

A data center can have a good PUE and still run an inefficient model stack. Conversely, a model can achieve good tokens per watt while being deployed in a facility with poor overhead. Use the right metric for the question.

### Key takeaway box

> **Key Takeaway:** PUE is not tokens per watt. Facility efficiency and workload efficiency are different layers.

### Web-readiness status

Ready after authoring.

### Print-readiness status

Medium risk.

### Required production fixes

- Validate PUE formula.
- Add caution about measurement boundary.
- Use workload-specific examples.

---

## Fig 5.5 — Power Cap vs Performance Curve

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch05_fig_5_5_power_cap_performance_curve.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch05_fig_5_5_power_cap_performance_curve.png`  
**Exact section placement:** Performance-per-watt / power capping section.

### Caption

**Fig 5.5 — Power Cap vs Performance Curve.**  
Performance often increases sublinearly with power. A lower power cap may improve performance per watt even if maximum throughput drops slightly.

### Intro paragraph before figure

Maximum power is not always maximum efficiency. GPUs often have voltage/frequency regions where additional watts produce diminishing performance gains.

### Explanation paragraph after figure

Power caps can improve fleet efficiency, reduce thermal stress, and increase reliability, but the optimal cap is workload-specific. Memory-bound workloads may lose little throughput under power caps, while compute-bound workloads may be more sensitive.

### Key takeaway box

> **Key Takeaway:** The best operating point may be the best sustained performance per watt, not the highest possible clock.

### Web-readiness status

Not ready.

### Print-readiness status

Not ready.

### Required production fixes

- Use representative curve, not product-specific claim.
- Label `[REPRESENTATIVE]`.
- Include performance/W curve if space allows.
- Add alt text.

---

## Fig 5.6 — Thermal Throttling Feedback Loop

**Type:** New figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch05_fig_5_6_thermal_throttling_loop.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch05_fig_5_6_thermal_throttling_loop.png`  
**Exact section placement:** Thermal throttling section.

### Caption

**Fig 5.6 — Thermal Throttling Feedback Loop.**  
When temperature rises beyond safe operating limits, the system may reduce frequency or power, lowering throughput and increasing job time.

### Intro paragraph before figure

A GPU can start a job at high performance and then slow down as thermal limits are reached. Sustained performance is what matters for training jobs and production serving.

### Explanation paragraph after figure

The loop connects workload intensity, power draw, heat, cooling capacity, device temperature, clock behavior, and throughput. Monitoring only average utilization can miss this feedback loop.

### Key takeaway box

> **Key Takeaway:** Thermal throttling converts a cooling problem into an application-performance problem.

### Web-readiness status

Not ready.

### Print-readiness status

Not ready.

### Required production fixes

- Create loop diagram.
- Show metrics to monitor: temperature, clocks, power, throttle reason, throughput.
- Add alt text.

---

## Table 5.5 — Power/Thermal Bottleneck Signals

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** After Fig 5.6.

### Caption

**Table 5.5 — Power and Thermal Bottleneck Signals.**  
Power and thermal problems have observable signatures in telemetry and application behavior.

### Proposed table content

| Symptom | Possible Cause | What to Check | Possible Fix |
|---|---|---|---|
| Throughput drops after warm-up | Thermal throttling | GPU temp, clocks, throttle flags | Improve cooling, adjust power cap |
| Frequent node resets | Power instability | PSU logs, BMC, rack power events | Rebalance load, inspect power chain |
| High tokens/W variance | Scheduling or thermal variation | Power, temperature, batch mix | Power-aware scheduling |
| Lower-than-expected clocks | Power cap or thermals | Clock telemetry, power limit | Tune cap/cooling |
| Rack breaker trips | Rack budget exceeded | Rack PDU telemetry | Reduce density, phase balance |
| Fan/CDU alarms | Cooling issue | Fan speed, CDU flow/temp | Service cooling loop |

### Intro paragraph before table

Power and thermal problems should be diagnosed with telemetry, not guesswork. The symptoms often appear as performance variation or reliability issues before they are recognized as infrastructure problems.

### Explanation paragraph after table

This table helps connect low-level telemetry to application-level impact. A training job slowdown may be caused by cooling instability, not model code.

### Key takeaway box

> **Key Takeaway:** Power and thermal events are performance events.

### Web-readiness status

Ready after authoring.

### Print-readiness status

Medium risk.

### Required production fixes

- Keep vendor-neutral.
- Add DCGM/rocm-smi/BMC examples in prose.
- Cross-reference observability chapter.

---

## Table 5.6 — Wrong Fix vs Right First Question

**Type:** New table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** Before principal interview section.

### Caption

**Table 5.6 — Wrong Fix vs Right First Question for Power/Thermal Problems.**  
Power and thermal issues are often misdiagnosed as software performance issues.

### Proposed table content

| Symptom | Wrong Fix | Right First Question |
|---|---|---|
| Throughput drops after minutes | Tune kernels | Are clocks dropping due to thermals? |
| Rack trips power | Replace GPUs | Is rack power budget exceeded? |
| Poor performance/W | Max out clocks | Are we past the efficient voltage/frequency region? |
| Random job failures | Retry jobs | Are failures correlated with power or thermal events? |
| Slow cluster during heat wave | Blame scheduler | Did cooling capacity or inlet temperature change? |
| Good benchmark, poor production | Tune model only | Is sustained power/cooling different from benchmark environment? |

### Intro paragraph before table

A principal engineer avoids jumping to software fixes when the limiting resource may be power delivery or thermal headroom.

### Explanation paragraph after table

The right first question is often a measurement-boundary question: where was power measured, where was temperature measured, and what changed at the rack or facility level?

### Key takeaway box

> **Key Takeaway:** If performance changes over time under constant workload, check power and thermals.

### Web-readiness status

Ready after authoring.

### Print-readiness status

Medium risk.

### Required production fixes

- Keep concise.
- Use in interview section.
- Add telemetry examples.

---

## Fig 5.7 — Power/Thermal Observability Stack

**Type:** New or adapted figure  
**Existing source file:** `diagram_08_observability_stack.html` can be adapted  
**Status:** Existing observability concept; Ch05-specific version should be created  
**Recommended asset path:** `assets/diagrams/svg/ch05_fig_5_7_power_thermal_observability_stack.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch05_fig_5_7_power_thermal_observability_stack.png`  
**Exact section placement:** Monitoring and reliability section.

### Caption

**Fig 5.7 — Power/Thermal Observability Stack.**  
A production AI cluster should collect GPU, server, rack, cooling, and facility telemetry and correlate it with application throughput and failures.

### Intro paragraph before figure

Power and thermal events are not isolated facilities events. They affect application throughput, job completion time, failure rate, and cost.

### Explanation paragraph after figure

The observability stack should collect metrics from GPUs, BMCs, PDUs, CDUs, facility systems, schedulers, and applications. Correlation is the key: power and thermal signals must be tied to throughput and reliability.

### Key takeaway box

> **Key Takeaway:** You cannot optimize what you cannot observe across hardware, facility, and workload layers.

### Web-readiness status

Partial; base observability diagram exists.

### Print-readiness status

Not ready.

### Required production fixes

- Adapt existing observability stack to power/thermal.
- Add metrics: power, temp, clocks, throttle reason, CDU flow, PDU load, failures.
- Add alt text.
- Cross-reference later observability chapter.

---

## Fig 5.8 — Power-Aware Scheduling Decision Flow

**Type:** New synthesis figure  
**Existing source file:** None  
**Status:** Must be created  
**Recommended asset path:** `assets/diagrams/svg/ch05_fig_5_8_power_aware_scheduling_flow.svg`  
**Recommended print export:** `assets/diagrams/png_300dpi/ch05_fig_5_8_power_aware_scheduling_flow.png`  
**Exact section placement:** End of chapter before key takeaways.

### Caption

**Fig 5.8 — Power-Aware Scheduling Decision Flow.**  
A scheduler can improve stability and efficiency by considering rack power budget, thermal headroom, workload intensity, and performance-per-watt targets.

### Intro paragraph before figure

At scale, power and thermal constraints become scheduling constraints. Not every job should be placed only by free GPU count.

### Explanation paragraph after figure

A power-aware scheduler can avoid placing too many high-power jobs in the same rack, can respect facility constraints, and can use power caps where performance loss is small but efficiency gain is high.

### Key takeaway box

> **Key Takeaway:** In AI data centers, scheduling is also power and thermal management.

### Web-readiness status

Not ready.

### Print-readiness status

Not ready.

### Required production fixes

- Create simple decision flow.
- Include rack power budget and cooling capacity.
- Label representative guidance.
- Add alt text.

---

## Table 5.7 — Production Infrastructure Checklist

**Type:** New checklist table  
**Existing source file:** None  
**Status:** Must be created  
**Exact section placement:** End of chapter / production checklist section.

### Caption

**Table 5.7 — Production AI Infrastructure Readiness Checklist.**  
Before deploying dense GPU systems, validate power, cooling, observability, reliability, safety, and operational runbooks.

### Proposed table content

| Category | Questions |
|---|---|
| Power | Is rack/system power budget validated under sustained workload? |
| Cooling | Is thermal capacity validated at peak and steady state? |
| Redundancy | What fails when one PSU, pump, fan, or CDU path fails? |
| Telemetry | Are GPU, PDU, BMC, CDU, and app metrics correlated? |
| Scheduling | Can scheduler avoid rack-level power/thermal hotspots? |
| Reliability | Are thermal/power events linked to job failures? |
| Safety | Are liquid cooling and electrical procedures documented? |
| Cost | Are tokens/W, jobs/day/MW, and cost/token tracked? |

### Intro paragraph before table

The chapter should end with a practical operational checklist. Dense AI systems require readiness across facilities, hardware, software, and operations.

### Explanation paragraph after table

The checklist helps principal engineers connect architecture decisions to deployment readiness. A powerful rack is not production-ready until it is power-stable, thermally stable, observable, and operationally safe.

### Key takeaway box

> **Key Takeaway:** Production AI infrastructure is a system of systems.

### Web-readiness status

Ready after authoring.

### Print-readiness status

Medium risk.

### Required production fixes

- Keep checklist concise.
- Consider moving expanded version to appendix.
- Cross-reference observability/reliability chapters.

---

# 3. Final Figure Numbering Recommendation

| Number | Title |
|---|---|
| Fig 5.1 | AI Data Center Power Delivery Chain |
| Fig 5.2 | GPU Power-to-Heat Flow |
| Fig 5.3 | Air Cooling vs Direct Liquid Cooling |
| Fig 5.4 | Rack Power Density Evolution |
| Fig 5.5 | Power Cap vs Performance Curve |
| Fig 5.6 | Thermal Throttling Feedback Loop |
| Fig 5.7 | Power/Thermal Observability Stack |
| Fig 5.8 | Power-Aware Scheduling Decision Flow |

---

# 4. Final Table Numbering Recommendation

| Number | Title |
|---|---|
| Table 5.1 | Power Terminology |
| Table 5.2 | Accelerator and System Power Reference |
| Table 5.3 | Cooling Method Comparison |
| Table 5.4 | PUE and AI Efficiency Metrics |
| Table 5.5 | Power/Thermal Bottleneck Signals |
| Table 5.6 | Wrong Fix vs Right First Question |
| Table 5.7 | Production Infrastructure Checklist |

---

# 5. Figure Inventory Updates

```markdown
| Figure | Title | Current Asset | Chapter | Web Ready | Print Ready | Required Action |
|---|---|---|---|---|---|---|
| Fig 5.1 | AI Data Center Power Delivery Chain | TBD | Ch05 | No | No | Create SVG + print export |
| Fig 5.2 | GPU Power-to-Heat Flow | TBD | Ch05 | No | No | Create SVG + print export |
| Fig 5.3 | Air Cooling vs Direct Liquid Cooling | TBD | Ch05 | No | No | Create SVG + print export |
| Fig 5.4 | Rack Power Density Evolution | TBD | Ch05 | No | No | Create SVG + print export |
| Fig 5.5 | Power Cap vs Performance Curve | TBD | Ch05 | No | No | Create SVG + print export |
| Fig 5.6 | Thermal Throttling Feedback Loop | TBD | Ch05 | No | No | Create SVG + print export |
| Fig 5.7 | Power/Thermal Observability Stack | diagram_08_observability_stack.html adaptation | Ch05 | Partial | No | Create Ch05-specific adaptation |
| Fig 5.8 | Power-Aware Scheduling Decision Flow | TBD | Ch05 | No | No | Create SVG + print export |
```

---

# 6. Recommended Commit

Save this file as:

```text
publishing/figure_plans/ch05_figure_integration_plan.md
```

Then run:

```powershell
git add publishing\figure_plans\ch05_figure_integration_plan.md
git commit -m "Add Chapter 5 figure integration plan"
git push origin production-v1.0
```
