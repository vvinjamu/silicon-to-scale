# Chapter 5 — Power, Thermal, and AI Data Center Infrastructure

> “A GPU does not run in a vacuum. It runs inside a power chain, a cooling system, a rack, a data center, and an operating model.”

---

## Chapter Overview

Chapter 4 explained why memory capacity, memory bandwidth, and data movement often dominate AI performance.

Chapter 5 moves one layer deeper into physical infrastructure.

Modern AI systems are constrained not only by:

```text
FLOPs
HBM
NVLink
PCIe
InfiniBand
software kernels
```

but also by:

```text
power delivery
cooling capacity
rack density
thermal stability
power capping
voltage/frequency behavior
facility efficiency
observability
reliability
operational safety
```

This chapter answers:

- Why is power a performance constraint?
- Why does thermal design affect sustained throughput?
- What is the difference between GPU power, board power, system power, rack power, and facility power?
- Why do dense AI racks often require liquid cooling?
- What is PUE, and what does it not measure?
- How do power caps affect performance per watt?
- Why can thermal throttling look like a software performance problem?
- How should a principal engineer discuss power and thermal infrastructure?

> **Current as of 2026 edition:** Power, thermal, and rack-level values are product-specific and time-sensitive. Verify final procurement decisions against current vendor product pages, deployment guides, facility design constraints, and safety requirements.

---

## 5.0 Power in One Page

Power is not just a facilities topic.

Power determines how much hardware can be installed, how hard it can run, how stable it remains, and how much it costs to operate.

A simple model:

```text
Power in → useful compute + losses → heat out
```

In a data center, nearly all IT electrical power eventually becomes heat that the facility must remove.

Core truths:

1. Every watt consumed by IT equipment becomes heat.
2. Power delivery limits determine how much hardware can run.
3. Cooling determines whether performance can be sustained.
4. Thermal limits determine whether clocks stay high.
5. PUE measures facility overhead, not model efficiency.
6. Performance per watt must be measured for a specific workload.
7. Rack-level power can become the limiting resource before GPU count does.

> **Key Takeaway:** Power and cooling are performance constraints, not side details.

---

## 5.1 Product-Level Power Guardrail

Before using any power number, ask:

```text
What is the measurement boundary?
```

A 750 W accelerator, a 14.3 kW system, and a 120 kW rack are all valid numbers.

They are not the same type of number.

```text
GPU/device power != board/module power != system power != rack power != facility power
```

If you mix these boundaries, every comparison becomes misleading.

---

## Table 5.1 — Power Terminology for AI Infrastructure

| Term | Boundary | Meaning | Common Mistake |
|---|---|---|---|
| TDP | Device/design thermal boundary | Thermal design target, vendor-specific meaning | Treating it as exact runtime power |
| TBP | Board/module | Board-level power, often used by AMD | Treating it as full server power |
| GPU power | Accelerator | GPU/device power telemetry | Ignoring CPU, fans, NICs, PSUs |
| System power | Server/DGX/HGX node | Full server draw | Treating as per-GPU power |
| Rack power | Rack | Sum of systems, switches, power shelves, pumps/fans where applicable | Comparing to one server |
| Facility power | Data center | IT + cooling + power overhead | Confusing with IT load |
| PUE | Facility efficiency | Total facility energy / IT equipment energy | Treating it as model efficiency |

[REPRESENTATIVE] TDP, TBP, GPU power, system power, rack power, and facility power describe different measurement boundaries. Do not compare them unless the boundary is normalized.

> **Key Takeaway:** Power numbers are meaningless without the measurement boundary.

---

## 5.2 AI Data Center Power Delivery Chain

A GPU cluster does not begin at the GPU.

It begins at the power source.

Power must flow through multiple layers before it reaches the accelerator:

```text
Utility / generation
  → transformer / substation
  → switchgear
  → UPS or backup path
  → PDU / busway
  → rack power shelf
  → server PSU
  → voltage regulators
  → GPU / CPU / memory / NICs
```

Every layer must be designed for sustained load, redundancy, serviceability, fault isolation, and safety.

---

## Figure Placeholder — Fig 5.1

```markdown
![Fig 5.1 — AI Data Center Power Delivery Chain](../assets/diagrams/svg/ch05_fig_5_1_power_delivery_chain.svg)

**Fig 5.1 — AI Data Center Power Delivery Chain.** Power flows from utility/grid or on-site generation through transformers, switchgear, UPS, PDUs, rack power shelves, server power supplies, voltage regulators, and finally GPUs, CPUs, memory, and networking.
```

**Figure intro:**  
A GPU cluster does not begin at the GPU. It begins at the power source. Every stage in the power chain must be designed for sustained load, redundancy, fault isolation, maintenance, and safety.

**Figure explanation:**  
If any link in the chain is undersized or unstable, the result may be throttling, node failures, breaker trips, job interruptions, or reduced cluster availability. AI infrastructure must be planned as an end-to-end power delivery system.

> **Key Takeaway:** A GPU is only as reliable as the power chain that feeds it.

---

## 5.3 Power Becomes Heat

In digital systems, almost all consumed electrical power eventually becomes heat.

A rack drawing 100 kW is also creating roughly 100 kW of heat that must be removed continuously.

That means power and thermal are the same problem viewed from two sides:

```text
Electrical side:
  Can we deliver the power safely and reliably?

Thermal side:
  Can we remove the heat continuously?

Performance side:
  Can the GPUs sustain clocks without throttling?

Operations side:
  Can we monitor, service, and recover the system?
```

---

## Figure Placeholder — Fig 5.2

```markdown
![Fig 5.2 — GPU Power-to-Heat Flow](../assets/diagrams/svg/ch05_fig_5_2_power_to_heat_flow.svg)

**Fig 5.2 — GPU Power-to-Heat Flow.** Almost all electrical power consumed by compute equipment ultimately becomes heat that must be removed by the cooling system.
```

**Figure intro:**  
In a data center, power and thermal are two sides of the same problem. A rack drawing 100 kW is also creating roughly 100 kW of heat that must be removed continuously.

**Figure explanation:**  
This is why AI infrastructure design must consider both electrical capacity and heat rejection. A system can be electrically powered but still fail to sustain performance if cooling cannot keep device temperatures within operating limits.

> **Key Takeaway:** Every watt consumed by IT equipment becomes heat the facility must remove.

---

## 5.4 Validated Power Reference Values

Power values should be read with product level visible.

Do not compare:

```text
MI300X 750 W TBP
```

directly against:

```text
DGX B200 14.3 kW system power
```

without normalizing the boundary.

One is board/module level.  
The other is system level.

Do not compare:

```text
DGX B200 system power
```

directly against:

```text
GB200 NVL72 rack power
```

without saying one is a server/system and the other is a rack-scale platform.

---

## Table 5.2 — Accelerator and System Power Reference

| Product | Level | Power Claim | Confidence |
|---|---|---:|---|
| H100 SXM5 | GPU/module | 700 W class depending SKU/source | `[SHIPPED]` when official SKU is used |
| H200 SXM | GPU/module | 700 W class depending SKU/source | `[SHIPPED]` when official SKU is used |
| MI300X | OAM accelerator | 750 W peak TBP | `[SHIPPED]` |
| DGX B200 | System | ~14.3 kW max system power | `[SHIPPED]` |
| GB200 NVL72 / DGX GB200 rack | Rack-scale | ~120 kW rack power consumption | `[SHIPPED]` |
| Future Rubin / MI400 rack systems | Rack-scale roadmap | TBD / announced | `[ANNOUNCED]` |

[SHIPPED] AMD Instinct MI300X lists 750 W peak TBP. This is an accelerator/module-level value, not a full server or rack power value.

[SHIPPED] NVIDIA DGX B200 lists approximately 14.3 kW maximum system power usage. This is a system-level value, not per-GPU power.

[SHIPPED] NVIDIA DGX GB200 rack-scale documentation lists approximately 120 kW rack power consumption for an NVL72 rack. This is a rack-level value and must not be compared directly to a single GPU or server.

> **Key Takeaway:** Compare power only after identifying whether the value is per GPU, per board, per system, or per rack.

---

## 5.5 Rack Power Density

Traditional enterprise racks were often planned around much lower power densities than today’s densest AI systems.

AI infrastructure changes the problem.

A rack can move from:

```text
single-digit kW
  → tens of kW
  → 100 kW+ rack-scale AI systems
```

[REPRESENTATIVE] AI infrastructure has pushed rack power density from traditional enterprise ranges into tens of kilowatts and, for rack-scale AI systems such as GB200 NVL72-class racks, around 100 kW or more. Exact values are product- and facility-specific.

This affects:

- electrical feed sizing,
- floor layout,
- airflow strategy,
- liquid cooling loops,
- service access,
- breaker and PDU design,
- redundancy,
- fire/safety procedures,
- and commissioning.

---

## Figure Placeholder — Fig 5.4

```markdown
![Fig 5.4 — Rack Power Density Evolution](../assets/diagrams/svg/ch05_fig_5_4_rack_power_density_evolution.svg)

**Fig 5.4 — Rack Power Density Evolution.** AI infrastructure has pushed rack power density from traditional enterprise ranges into tens and, for rack-scale AI systems, around 100 kW or more.
```

**Figure intro:**  
Traditional data center rack power budgets are not enough for the densest AI systems. Rack-scale GPU platforms require a different level of power delivery, cooling capacity, commissioning, and monitoring.

**Figure explanation:**  
The figure should use representative ranges, not universal thresholds. It should visually show why 40–60 kW racks and 100 kW+ racks require different cooling and power planning than traditional enterprise racks.

> **Key Takeaway:** Rack density changes the data center design problem.

---

## 5.6 Air Cooling vs Liquid Cooling

Cooling is not a binary religion.

It is an infrastructure design choice.

Air cooling can work well for lower-density systems and mature service environments. But as rack density rises, airflow becomes harder to scale.

[REPRESENTATIVE] As rack power density rises into tens or hundreds of kilowatts, direct liquid cooling or hybrid liquid/air cooling often becomes necessary. The exact threshold depends on rack design, facility airflow, coolant infrastructure, water strategy, and service model.

Liquid cooling is not “free.” It adds:

- cold plates,
- manifolds,
- coolant distribution units,
- leak detection,
- pressure/flow monitoring,
- service procedures,
- facility water loops,
- and operational training.

---

## Figure Placeholder — Fig 5.3

```markdown
![Fig 5.3 — Air Cooling vs Direct Liquid Cooling](../assets/diagrams/svg/ch05_fig_5_3_air_vs_liquid_cooling.svg)

**Fig 5.3 — Air Cooling vs Direct Liquid Cooling.** Air cooling removes heat through airflow and heat sinks; direct liquid cooling moves heat through cold plates, coolant loops, CDUs, and facility heat rejection systems.
```

**Figure intro:**  
As rack density rises, airflow alone becomes increasingly difficult. Direct liquid cooling moves heat more efficiently from high-power devices but adds plumbing, service, monitoring, and facility integration requirements.

**Figure explanation:**  
The figure should show tradeoffs rather than declare one method universally superior. Air cooling can remain suitable for lower-density systems. Liquid cooling becomes increasingly important for dense AI racks and rack-scale systems.

> **Key Takeaway:** Cooling strategy follows rack density, service model, facility design, and reliability requirements.

---

## Table 5.3 — Cooling Method Comparison for AI Infrastructure

| Cooling Method | Best Fit | Strengths | Watchouts |
|---|---|---|---|
| Air cooling | Lower/mid density racks | Simpler, mature, service-friendly | Airflow limits at high density |
| Rear-door heat exchanger | Transitional high-density racks | Can extend air-cooled facilities | Still facility-dependent |
| Direct-to-chip liquid | Dense GPU servers/racks | Efficient heat removal near devices | Plumbing, leak detection, service model |
| Immersion cooling | Specialized high-density environments | High thermal capacity | Ecosystem, maintenance, compatibility |
| Hybrid liquid/air | Most dense AI platforms | Handles device heat and residual air load | Requires integrated design |

[REPRESENTATIVE] Direct-to-chip liquid cooling typically removes heat through cold plates, coolant loops, CDUs, and facility heat rejection systems, while air cooling may still remove residual heat from memory, networking, power supplies, and other components.

> **Key Takeaway:** Cooling is an infrastructure architecture decision, not an afterthought.

---

## 5.7 Thermal Throttling and Sustained Performance

Peak performance is not enough.

Sustained performance matters.

A GPU can start a job at high clocks and then slow down when thermal limits are reached.

[REPRESENTATIVE] When device temperatures approach operating limits, GPUs may reduce clocks or power to protect the hardware, lowering sustained application throughput. Confirm with temperature, clock, power, and throttle-reason telemetry.

Thermal throttling can appear as:

- lower tokens/sec,
- lower samples/sec,
- lower training step rate,
- increased latency,
- higher variance,
- or job instability.

The workload may look like a software problem, but the root cause may be thermal headroom.

---

## Figure Placeholder — Fig 5.6

```markdown
![Fig 5.6 — Thermal Throttling Feedback Loop](../assets/diagrams/svg/ch05_fig_5_6_thermal_throttling_loop.svg)

**Fig 5.6 — Thermal Throttling Feedback Loop.** When temperature rises beyond safe operating limits, the system may reduce frequency or power, lowering throughput and increasing job time.
```

**Figure intro:**  
A GPU can start a job at high performance and then slow down as thermal limits are reached. Sustained performance is what matters for training jobs and production serving.

**Figure explanation:**  
The loop connects workload intensity, power draw, heat, cooling capacity, device temperature, clock behavior, and throughput. Monitoring only average utilization can miss this feedback loop.

> **Key Takeaway:** Thermal throttling converts a cooling problem into an application-performance problem.

---

## Table 5.5 — Power and Thermal Bottleneck Signals

| Symptom | Possible Cause | What to Check | Possible Fix |
|---|---|---|---|
| Throughput drops after warm-up | Thermal throttling | GPU temp, clocks, throttle flags | Improve cooling, adjust power cap |
| Frequent node resets | Power instability | PSU logs, BMC, rack power events | Rebalance load, inspect power chain |
| High tokens/W variance | Scheduling or thermal variation | Power, temperature, batch mix | Power-aware scheduling |
| Lower-than-expected clocks | Power cap or thermals | Clock telemetry, power limit | Tune cap/cooling |
| Rack breaker trips | Rack budget exceeded | Rack PDU telemetry | Reduce density, phase balance |
| Fan/CDU alarms | Cooling issue | Fan speed, CDU flow/temp | Service cooling loop |

[REPRESENTATIVE] Power and thermal instability can appear as application slowdowns, node resets, retries, job failures, or correlated hardware events. Confirm with GPU telemetry, BMC logs, PDU data, cooling-system metrics, and scheduler history.

> **Key Takeaway:** Power and thermal events are performance events.

---

## 5.8 Power Cap vs Performance per Watt

Maximum power is not always maximum efficiency.

At high operating points, additional frequency often requires disproportionately more power.

[REPRESENTATIVE] At high operating points, additional frequency often requires disproportionately more power, so maximum clock is not always the best performance-per-watt point.

A power cap can sometimes:

- reduce energy cost,
- reduce thermal stress,
- improve reliability,
- reduce fan/pump overhead,
- improve cluster stability,
- and improve performance per watt.

But it can also reduce throughput.

[REPRESENTATIVE] Power caps can improve performance per watt when the workload is in a region where extra power produces diminishing throughput gains. The optimal cap is workload-, hardware-, cooling-, and SLA-specific.

---

## Figure Placeholder — Fig 5.5

```markdown
![Fig 5.5 — Power Cap vs Performance Curve](../assets/diagrams/svg/ch05_fig_5_5_power_cap_performance_curve.svg)

**Fig 5.5 — Power Cap vs Performance Curve.** Performance often increases sublinearly with power. A lower power cap may improve performance per watt even if maximum throughput drops slightly.
```

**Figure intro:**  
Maximum power is not always maximum efficiency. GPUs often have voltage/frequency regions where additional watts produce diminishing performance gains.

**Figure explanation:**  
Power caps can improve fleet efficiency, reduce thermal stress, and increase reliability, but the optimal cap is workload-specific. Memory-bound workloads may lose little throughput under power caps, while compute-bound workloads may be more sensitive.

> **Key Takeaway:** The best operating point may be the best sustained performance per watt, not the highest possible clock.

---

## 5.9 PUE and Efficiency Metrics

PUE means Power Usage Effectiveness.

[SHIPPED] Power Usage Effectiveness, or PUE, is total facility energy divided by IT equipment energy. PUE measures facility overhead; it does not measure tokens per watt or model efficiency.

Formula:

```text
PUE = Total Facility Energy / IT Equipment Energy
```

A PUE of 1.2 means:

```text
For every 1.0 unit of IT energy,
the facility uses 0.2 additional units for cooling, power overhead, and other infrastructure.
```

But PUE does not tell you:

- tokens/sec/W,
- cost/token,
- GPU utilization,
- model quality,
- latency,
- or training throughput per watt.

A data center can have good PUE and still run inefficient model serving. A model can have good tokens/W and still be deployed in a facility with poor overhead.

---

## Table 5.4 — PUE and AI Efficiency Metrics

| Metric | Formula | Measures | Does Not Measure |
|---|---|---|---|
| PUE | Total facility energy / IT equipment energy | Facility infrastructure efficiency | Model efficiency |
| IT power | Server/network/storage power | Compute load | Cooling overhead |
| Tokens/sec/W | Token throughput / power | Inference efficiency | Quality or latency alone |
| Samples/sec/W | Samples throughput / power | Training/inference efficiency | Job completion alone |
| TFLOPS/W | FLOPs / power | Math efficiency | Memory/communication efficiency |
| Cost/token | Cost / output token | Business efficiency | Hardware utilization alone |

[REPRESENTATIVE] Performance per watt must be measured for a specific workload and metric: tokens/sec/W for inference, samples/sec/W for training/inference throughput, TFLOPS/W for math kernels, or jobs/day/MW for fleet planning.

> **Key Takeaway:** PUE is not tokens per watt. Facility efficiency and workload efficiency are different layers.

---

## 5.10 Power and Cost of AI Infrastructure

At scale, cost is not only the purchase price of GPUs.

Cost includes:

```text
GPU/server cost
networking cost
storage cost
power cost
cooling cost
rack/floor space
facility upgrades
spares
maintenance
downtime
staffing
software operations
```

[REPRESENTATIVE] At scale, cost per token depends not only on GPU price but also on energy cost, cooling overhead, power availability, utilization, failure rate, operational staffing, and amortization model.

Useful questions:

- What is the cost per generated token?
- What is the cost per training step?
- What is the cost per job completed?
- What is tokens/sec/W?
- What is jobs/day/MW?
- What is the reliability impact of running near thermal limits?
- What happens to throughput when power caps are applied?

> **Key Takeaway:** Power and cooling turn performance engineering into business economics.

---

## 5.11 Power/Thermal Observability

You cannot manage what you cannot observe.

Power and thermal telemetry should be correlated across:

- GPU power,
- GPU temperature,
- memory temperature if available,
- clocks,
- throttle reasons,
- fan speed,
- pump/CDU status,
- PDU and rack power,
- inlet temperature,
- outlet temperature,
- BMC events,
- job throughput,
- latency,
- failures,
- retries,
- scheduler placement.

---

## Figure Placeholder — Fig 5.7

```markdown
![Fig 5.7 — Power/Thermal Observability Stack](../assets/diagrams/svg/ch05_fig_5_7_power_thermal_observability_stack.svg)

**Fig 5.7 — Power/Thermal Observability Stack.** A production AI cluster should collect GPU, server, rack, cooling, and facility telemetry and correlate it with application throughput and failures.
```

**Figure intro:**  
Power and thermal events are not isolated facilities events. They affect application throughput, job completion time, failure rate, and cost.

**Figure explanation:**  
The observability stack should collect metrics from GPUs, BMCs, PDUs, CDUs, facility systems, schedulers, and applications. Correlation is the key: power and thermal signals must be tied to throughput and reliability.

> **Key Takeaway:** You cannot optimize what you cannot observe across hardware, facility, and workload layers.

---

## 5.12 Power-Aware Scheduling

At small scale, scheduling can focus on:

```text
How many GPUs are free?
```

At AI data center scale, that is not enough.

A scheduler may need to consider:

- rack power budget,
- thermal headroom,
- cooling zone,
- workload power intensity,
- topology,
- failure history,
- current inlet temperature,
- power cap policy,
- and performance-per-watt target.

[REPRESENTATIVE] At AI-cluster scale, scheduling can improve stability and efficiency by considering rack power budget, cooling headroom, workload intensity, and failure history, not only free GPU count.

---

## Figure Placeholder — Fig 5.8

```markdown
![Fig 5.8 — Power-Aware Scheduling Decision Flow](../assets/diagrams/svg/ch05_fig_5_8_power_aware_scheduling_flow.svg)

**Fig 5.8 — Power-Aware Scheduling Decision Flow.** A scheduler can improve stability and efficiency by considering rack power budget, thermal headroom, workload intensity, and performance-per-watt targets.
```

**Figure intro:**  
At scale, power and thermal constraints become scheduling constraints. Not every job should be placed only by free GPU count.

**Figure explanation:**  
A power-aware scheduler can avoid placing too many high-power jobs in the same rack, can respect facility constraints, and can use power caps where performance loss is small but efficiency gain is high.

> **Key Takeaway:** In AI data centers, scheduling is also power and thermal management.

---

## 5.13 Wrong Fix vs Right First Question

Power and thermal issues are often misdiagnosed as software performance issues.

A principal engineer avoids jumping to software fixes when the limiting resource may be power delivery or thermal headroom.

---

## Table 5.6 — Wrong Fix vs Right First Question for Power/Thermal Problems

| Symptom | Wrong Fix | Right First Question |
|---|---|---|
| Throughput drops after minutes | Tune kernels | Are clocks dropping due to thermals? |
| Rack trips power | Replace GPUs | Is rack power budget exceeded? |
| Poor performance/W | Max out clocks | Are we past the efficient voltage/frequency region? |
| Random job failures | Retry jobs | Are failures correlated with power or thermal events? |
| Slow cluster during heat wave | Blame scheduler | Did cooling capacity or inlet temperature change? |
| Good benchmark, poor production | Tune model only | Is sustained power/cooling different from benchmark environment? |

> **Key Takeaway:** If performance changes over time under constant workload, check power and thermals.

---

## 5.14 Production AI Infrastructure Readiness Checklist

Dense AI systems require readiness across facilities, hardware, software, and operations.

---

## Table 5.7 — Production AI Infrastructure Readiness Checklist

| Category | Questions |
|---|---|
| Power | Is rack/system power budget validated under sustained workload? |
| Cooling | Is thermal capacity validated at peak and steady state? |
| Redundancy | What fails when one PSU, pump, fan, or CDU path fails? |
| Telemetry | Are GPU, PDU, BMC, CDU, and app metrics correlated? |
| Scheduling | Can the scheduler avoid rack-level power/thermal hotspots? |
| Reliability | Are thermal/power events linked to job failures? |
| Safety | Are liquid cooling and electrical procedures documented? |
| Cost | Are tokens/W, jobs/day/MW, and cost/token tracked? |

The checklist helps principal engineers connect architecture decisions to deployment readiness.

A powerful rack is not production-ready until it is:

- power-stable,
- thermally stable,
- observable,
- serviceable,
- resilient,
- and operationally safe.

> **Key Takeaway:** Production AI infrastructure is a system of systems.

---

## 5.15 How to Discuss Power/Thermal Infrastructure in a Principal Interview

A weak answer sounds like this:

```text
We need enough power and cooling.
```

That is true, but too shallow.

A principal-level answer sounds like this:

> I treat power and thermal as performance constraints, not facilities details. I separate GPU, board, system, rack, and facility power, then check whether the power delivery and cooling design can sustain the workload. For AI clusters, I monitor power draw, temperatures, throttling, clocks, fan or CDU behavior, failures, and throughput per watt. I also evaluate power caps because maximum power is not always maximum useful throughput per watt.

### Scenario 1 — Why Can Throughput Drop After a Few Minutes?

Answer:

```text
If throughput drops after warm-up under a stable workload, I check thermal throttling, power caps, clocks, inlet temperature, cooling telemetry, and rack power events before assuming the model code changed.
```

### Scenario 2 — Why Is GB200 NVL72 a Facility-Level Discussion?

Answer:

```text
GB200 NVL72 is a rack-scale platform, not just a GPU. A rack-level power number around 120 kW affects facility power delivery, liquid cooling design, commissioning, redundancy, and operational safety. It must be planned at rack and data-center level.
```

### Scenario 3 — Why Is DGX B200 14.3 kW Not a Per-GPU Number?

Answer:

```text
DGX B200 is a system-level product. The 14.3 kW value includes the whole system power envelope, not just one GPU. I would not compare it directly to MI300X 750 W TBP without normalizing the boundary.
```

### Scenario 4 — Why Might Power Capping Improve Efficiency?

Answer:

```text
Performance often increases sublinearly with power near the high end of the voltage/frequency curve. A power cap can reduce energy and heat with limited throughput loss for some workloads, improving tokens/sec/W or jobs/day/MW. But the optimal cap must be measured per workload.
```

### Scenario 5 — What Metrics Would You Monitor?

Answer:

```text
I would monitor GPU power, clocks, temperatures, memory temperatures if available, throttle reasons, PDU/rack power, inlet/outlet temperature, CDU flow and alarms, BMC events, job failures, throughput, latency, and performance per watt.
```

---

## 5.16 Chapter Cheat Sheet

### Power Boundary

```text
GPU/device power
  != board/module power
  != system power
  != rack power
  != facility power
```

### Validated Reference Values

```text
[SHIPPED] MI300X: 750 W peak TBP
[SHIPPED] DGX B200: ~14.3 kW max system power
[SHIPPED] GB200 NVL72 / DGX GB200 rack: ~120 kW rack power
```

### PUE

```text
PUE = Total Facility Energy / IT Equipment Energy
```

### Performance per Watt

```text
tokens/sec/W
samples/sec/W
TFLOPS/W
jobs/day/MW
cost/token
```

### Cooling

```text
Air cooling: simpler, mature, lower-density fit
Liquid cooling: higher density support, more facility/service complexity
Hybrid cooling: common for dense AI systems
```

### Principal Rule

```text
If performance changes over time under constant workload, check thermals and power.
```

---

## 5.17 Key Takeaways

1. Power and thermal are performance constraints, not side details.
2. Almost all consumed IT power becomes heat.
3. Power numbers require measurement boundaries.
4. GPU power, board power, system power, rack power, and facility power are different.
5. MI300X 750 W TBP is an accelerator/module-level value.
6. DGX B200 ~14.3 kW is a system-level value.
7. GB200 NVL72 ~120 kW is a rack-level value.
8. Rack density changes the data center design problem.
9. Air cooling and liquid cooling are design choices driven by density, facility readiness, service model, and reliability.
10. Thermal throttling turns a cooling problem into an application-performance problem.
11. PUE measures facility overhead, not model efficiency.
12. Tokens/sec/W and cost/token are workload-level efficiency metrics.
13. Power caps can improve performance per watt when extra power gives diminishing throughput returns.
14. Power-aware scheduling can improve stability and efficiency at cluster scale.
15. Power and thermal telemetry must be correlated with application throughput and failures.
16. A production AI rack is not ready until it is power-stable, thermally stable, observable, and serviceable.

---

## 5.18 Review Questions

### Conceptual

1. Why are power and thermal performance constraints?
2. Why does almost all IT power become heat?
3. What is the difference between GPU power, board power, system power, and rack power?
4. Why is MI300X 750 W TBP not comparable to DGX B200 14.3 kW system power?
5. Why is GB200 NVL72 a rack-level infrastructure discussion?
6. Why does high rack density often push infrastructure toward liquid cooling?
7. Why is thermal throttling an application-performance problem?
8. What does PUE measure?
9. What does PUE not measure?
10. Why can power capping improve performance per watt?
11. Why should scheduling consider rack power and thermal headroom?
12. Why should power/thermal telemetry be correlated with job metrics?

### Calculation

1. Estimate accelerator-only power for eight MI300X accelerators at 750 W each.
2. If a DGX B200 system is approximately 14.3 kW max power, what is the rough power for four such systems before facility overhead?
3. If a rack consumes 120 kW and runs for one hour, how much energy is consumed at the rack level?
4. If PUE is 1.2 and IT load is 1 MW, what is total facility power?
5. If a power cap reduces power by 15% and throughput by 5%, what happens to performance per watt?

### Principal-Level Interview Practice

1. Explain why power and thermal constraints matter for AI performance.
2. Explain the product-level power guardrail.
3. Explain the difference between MI300X TBP and DGX B200 system power.
4. Explain why GB200 NVL72 requires rack-level planning.
5. Explain air cooling vs liquid cooling tradeoffs.
6. Explain how you would diagnose thermal throttling.
7. Explain how power caps can improve efficiency.
8. Explain why PUE is not enough to judge model-serving efficiency.
9. Explain how you would design power/thermal observability for a GPU cluster.
10. Explain how power-aware scheduling could improve reliability.

---

## 5.19 Production Notes for This Chapter

### Figure Assets Needed

| Figure | Status |
|---|---|
| Fig 5.1 — AI Data Center Power Delivery Chain | Must be created |
| Fig 5.2 — GPU Power-to-Heat Flow | Must be created |
| Fig 5.3 — Air Cooling vs Direct Liquid Cooling | Must be created |
| Fig 5.4 — Rack Power Density Evolution | Must be created |
| Fig 5.5 — Power Cap vs Performance Curve | Must be created |
| Fig 5.6 — Thermal Throttling Feedback Loop | Must be created |
| Fig 5.7 — Power/Thermal Observability Stack | Adapt from observability stack |
| Fig 5.8 — Power-Aware Scheduling Decision Flow | Must be created |

### Table Assets Included

| Table | Status |
|---|---|
| Table 5.1 — Power Terminology | Included |
| Table 5.2 — Accelerator and System Power Reference | Included |
| Table 5.3 — Cooling Method Comparison | Included |
| Table 5.4 — PUE and AI Efficiency Metrics | Included |
| Table 5.5 — Power/Thermal Bottleneck Signals | Included |
| Table 5.6 — Wrong Fix vs Right First Question | Included |
| Table 5.7 — Production Infrastructure Checklist | Included |

### Source Notes to Add in Final Book

Use official or primary sources for:

- AMD MI300X product page / datasheet
- NVIDIA DGX B200 product page / user guide
- NVIDIA DGX GB200 rack-scale systems user guide
- NVIDIA H100/H200 datasheets if GPU-level power is stated
- ASHRAE / data center cooling guidance
- Green Grid / DOE PUE references
- NVIDIA DCGM, BMC, PDU, CDU telemetry references
- ROCm SMI and AMD telemetry references
- Internal/fleet measurements labeled `[ENV-SPECIFIC]`

---

## 5.20 Bridge to Chapter 6

Chapter 5 showed why AI infrastructure must sustain power, cooling, and reliability before theoretical performance becomes real performance.

Chapter 6 moves from physical infrastructure back into workload execution:

```text
How do training and inference pipelines stress the hardware differently?
```

The next chapter connects model execution phases, batching, scheduling, serving, and training loops to the silicon, memory, power, and network constraints introduced so far.
