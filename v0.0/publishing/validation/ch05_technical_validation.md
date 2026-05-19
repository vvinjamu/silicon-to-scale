# Chapter 5 Technical Validation Plan

**Book:** *AI/ML Infrastructure from Silicon to Scale*  
**Chapter:** Ch05 — *Power, Thermal, and AI Data Center Infrastructure*  
**Target file:** `publishing/validation/ch05_technical_validation.md`  
**Production status:** Production Planning Pack  
**Last reviewed:** 2026-04-30

---

## 0. Executive Summary

Ch05 has a high risk of incorrect comparisons because power and thermal values are measured at different boundaries:

```text
GPU/device power
board/module power
server/system power
rack power
facility power
```

The chapter must never mix these boundaries without stating the product level.

The most important validated examples:

- `[SHIPPED]` AMD MI300X lists **750 W peak TBP**.
- `[SHIPPED]` NVIDIA DGX B200 official system materials list **~14.3 kW max system power**.
- `[SHIPPED]` NVIDIA DGX GB200 rack-scale guide lists **approximately 120 kW rack power consumption** for an NVL72 rack.
- `[SHIPPED]` PUE is defined as **total facility energy divided by IT equipment energy**.
- `[REPRESENTATIVE]` Air cooling vs liquid cooling thresholds are facility-dependent and should be discussed as ranges, not universal rules.

---

# 1. Validation Table

---

## 1.1 GPU Power and TDP/TBP Terminology

| Field | Validation |
|---|---|
| Claim | GPU power values must distinguish TDP, TBP, GPU power, system power, rack power, and facility power. |
| Current value/formula | Measurement-boundary discipline. |
| Validation status | Valid and required. |
| Corrected value/safe wording | Define terms before using power numbers. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | Vendor datasheets and architecture docs |
| Recommended final wording | `[REPRESENTATIVE] TDP, TBP, GPU power, system power, rack power, and facility power describe different measurement boundaries. Do not compare them unless the boundary is normalized.` |
| Priority | P0 |

---

## 1.2 H100 SXM5 Power Claim

| Field | Validation |
|---|---|
| Claim | H100 SXM5 is often treated as a ~700 W class GPU/module. |
| Current value/formula | H100 SXM5 thermal design / max power class around 700 W depending source/SKU. |
| Validation status | Valid if sourced to official H100 SXM spec. |
| Corrected value/safe wording | Use “H100 SXM5 700 W class” or exact official value with SKU. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H100 datasheet / product page / OEM spec |
| Recommended final wording | `[SHIPPED] H100 SXM5 should be treated as a 700 W class accelerator when using official SXM5 power specifications. Always keep SKU and form factor visible.` |
| Priority | P0 |

---

## 1.3 H200 Power Claim

| Field | Validation |
|---|---|
| Claim | H200 SXM is a similar high-power Hopper-generation accelerator, commonly discussed around 700 W class depending SKU. |
| Current value/formula | H200 SXM product specs/OEM sheets often list 700 W class thermal design power. |
| Validation status | Valid if official product or OEM datasheet is used. |
| Corrected value/safe wording | Tie to SKU and source. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA H200 datasheet / OEM platform spec |
| Recommended final wording | `[SHIPPED] H200 SXM should be treated as a high-power Hopper-generation accelerator; use the exact official SKU/platform power value when publishing tables.` |
| Priority | P1 |

---

## 1.4 MI300X TBP / Power Claim

| Field | Validation |
|---|---|
| Claim | AMD Instinct MI300X has 750 W peak TBP. |
| Current value/formula | 750 W peak typical board power / TBP. |
| Validation status | Valid from AMD product specification. |
| Corrected value/safe wording | Say board/module-level TBP, not full server power. |
| Confidence label | `[SHIPPED]` |
| Source type needed | AMD MI300X product page / datasheet |
| Recommended final wording | `[SHIPPED] AMD Instinct MI300X lists 750 W peak TBP. This is an accelerator/module-level value, not a full server or rack power value.` |
| Priority | P0 |

---

## 1.5 B200 Per-GPU Power Claim

| Field | Validation |
|---|---|
| Claim | B200 per-GPU power should be discussed carefully. |
| Current value/formula | Product-level values vary by B200, GB200 Superchip, DGX B200, GB200 NVL72 context. |
| Validation status | Requires official per-GPU/product datasheet before publication. |
| Corrected value/safe wording | Avoid per-GPU B200 power unless sourced. Use DGX B200 system-level and GB200 rack-level where official. |
| Confidence label | `[SHIPPED]` if official source; otherwise avoid or `[ANNOUNCED]` |
| Source type needed | NVIDIA B200/HGX B200/DGX B200 official datasheet |
| Recommended final wording | `Use B200 power values only with product-level context. If only DGX B200 or GB200 NVL72 values are sourced, present them as system-level or rack-level values, not per-GPU values.` |
| Priority | P0 |

---

## 1.6 DGX B200 System Power

| Field | Validation |
|---|---|
| Claim | NVIDIA DGX B200 system power usage is approximately 14.3 kW max. |
| Current value/formula | ~14.3 kW max system power. |
| Validation status | Valid as system-level value from NVIDIA DGX B200 materials. |
| Corrected value/safe wording | State “system-level” every time. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA DGX B200 product page / user guide |
| Recommended final wording | `[SHIPPED] NVIDIA DGX B200 lists approximately 14.3 kW maximum system power usage. This is a system-level value, not per-GPU power.` |
| Priority | P0 |

---

## 1.7 GB200 NVL72 / DGX GB200 Rack Power

| Field | Validation |
|---|---|
| Claim | GB200 NVL72 / DGX GB200 rack power consumption is approximately 120 kW. |
| Current value/formula | NVIDIA DGX GB rack guide lists approximately 120 kW rack power consumption for NVL72 rack. |
| Validation status | Valid as rack-level value if sourced to NVIDIA user guide. |
| Corrected value/safe wording | State rack-level and approximate. |
| Confidence label | `[SHIPPED]` |
| Source type needed | NVIDIA DGX GB200 rack-scale system user guide |
| Recommended final wording | `[SHIPPED] NVIDIA DGX GB200 rack-scale documentation lists approximately 120 kW rack power consumption for an NVL72 rack. This is a rack-level value and must not be compared directly to a single GPU or server.` |
| Priority | P0 |

---

## 1.8 Rack Power Density

| Field | Validation |
|---|---|
| Claim | AI racks increasingly reach tens of kW and, for rack-scale AI systems, around 100 kW or more. |
| Current value/formula | Representative industry trend; GB200 NVL72 class around 120 kW. |
| Validation status | Valid as trend with examples. |
| Corrected value/safe wording | Use representative ranges and product examples. |
| Confidence label | `[REPRESENTATIVE]`; specific products `[SHIPPED]` |
| Source type needed | NVIDIA/OEM rack specs, ASHRAE/data center guidance |
| Recommended final wording | `[REPRESENTATIVE] AI infrastructure has pushed rack power density from traditional enterprise ranges into tens of kilowatts and, for rack-scale AI systems such as GB200 NVL72-class racks, around 100 kW or more. Exact values are product- and facility-specific.` |
| Priority | P0 |

---

## 1.9 Air Cooling vs Liquid Cooling

| Field | Validation |
|---|---|
| Claim | Higher rack density often requires liquid or hybrid cooling. |
| Current value/formula | Representative industry guidance. |
| Validation status | Valid when not absolute. |
| Corrected value/safe wording | Avoid “air cooling cannot cool AI.” |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | ASHRAE/OEM/data center cooling guidance |
| Recommended final wording | `[REPRESENTATIVE] As rack power density rises into tens or hundreds of kilowatts, direct liquid cooling or hybrid liquid/air cooling often becomes necessary. The exact threshold depends on rack design, facility airflow, coolant infrastructure, water strategy, and service model.` |
| Priority | P0 |

---

## 1.10 Thermal Throttling

| Field | Validation |
|---|---|
| Claim | If thermal limits are reached, the system may reduce clocks or power, lowering sustained throughput. |
| Current value/formula | Thermal protection / DVFS behavior. |
| Validation status | Valid generally; implementation-specific. |
| Corrected value/safe wording | Use “may” and monitor throttle reasons. |
| Confidence label | `[REPRESENTATIVE]`; measured incidents `[ENV-SPECIFIC]` |
| Source type needed | Vendor GPU docs / telemetry docs / profiler docs |
| Recommended final wording | `[REPRESENTATIVE] When device temperatures approach operating limits, GPUs may reduce clocks or power to protect the hardware, lowering sustained application throughput. Confirm with temperature, clock, power, and throttle-reason telemetry.` |
| Priority | P0 |

---

## 1.11 PUE Formula

| Field | Validation |
|---|---|
| Claim | PUE = total facility energy / IT equipment energy. |
| Current value/formula | `PUE = Total Facility Energy / IT Equipment Energy` |
| Validation status | Valid standard metric. |
| Corrected value/safe wording | State that PUE is facility efficiency, not model efficiency. |
| Confidence label | `[SHIPPED]` for standard definition |
| Source type needed | Green Grid / DOE / industry standard source |
| Recommended final wording | `[SHIPPED] Power Usage Effectiveness, or PUE, is total facility energy divided by IT equipment energy. PUE measures facility overhead; it does not measure tokens per watt or model efficiency.` |
| Priority | P0 |

---

## 1.12 Performance per Watt Formula

| Field | Validation |
|---|---|
| Claim | Performance per watt must be workload-specific. |
| Current value/formula | tokens/sec/W, samples/sec/W, TFLOPS/W, jobs/day/MW. |
| Validation status | Valid. |
| Corrected value/safe wording | Define metric and workload. |
| Confidence label | `[REPRESENTATIVE]`; measured values `[ENV-SPECIFIC]` |
| Source type needed | Benchmark methodology |
| Recommended final wording | `[REPRESENTATIVE] Performance per watt must be measured for a specific workload and metric: tokens/sec/W for inference, samples/sec/W for training/inference throughput, TFLOPS/W for math kernels, or jobs/day/MW for fleet planning.` |
| Priority | P0 |

---

## 1.13 Power Cap vs Performance Curve

| Field | Validation |
|---|---|
| Claim | Power caps can improve performance per watt because performance often increases sublinearly with power. |
| Current value/formula | Representative DVFS behavior. |
| Validation status | Valid directionally; workload-specific. |
| Corrected value/safe wording | Do not claim universal savings. |
| Confidence label | `[REPRESENTATIVE]`; measured values `[ENV-SPECIFIC]` |
| Source type needed | Vendor power management docs / benchmarking |
| Recommended final wording | `[REPRESENTATIVE] Power caps can improve performance per watt when the workload is in a region where extra power produces diminishing throughput gains. The optimal cap is workload-, hardware-, cooling-, and SLA-specific.` |
| Priority | P0 |

---

## 1.14 Frequency/Voltage Tradeoff

| Field | Validation |
|---|---|
| Claim | Higher frequency usually requires higher voltage and can reduce energy efficiency at the high end. |
| Current value/formula | DVFS principle. |
| Validation status | Valid generally; implementation-specific. |
| Corrected value/safe wording | Keep conceptual. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | GPU power management docs / DVFS references |
| Recommended final wording | `[REPRESENTATIVE] At high operating points, additional frequency often requires disproportionately more power, so maximum clock is not always the best performance-per-watt point.` |
| Priority | P1 |

---

## 1.15 Power-Aware Scheduling

| Field | Validation |
|---|---|
| Claim | Scheduling should consider rack power budget and thermal headroom, not only free GPU count. |
| Current value/formula | Operational guidance. |
| Validation status | Valid representative guidance. |
| Corrected value/safe wording | Use “can” and “should be considered.” |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | Cluster scheduler practices / operational telemetry |
| Recommended final wording | `[REPRESENTATIVE] At AI-cluster scale, scheduling can improve stability and efficiency by considering rack power budget, cooling headroom, workload intensity, and failure history, not only free GPU count.` |
| Priority | P1 |

---

## 1.16 Failure and Reliability Risks

| Field | Validation |
|---|---|
| Claim | Power and thermal instability can contribute to failures, retries, throttling, and job interruptions. |
| Current value/formula | Operational reality; environment-specific. |
| Validation status | Valid but deployment-specific. |
| Corrected value/safe wording | Use telemetry correlation wording. |
| Confidence label | `[ENV-SPECIFIC]` for incidents; `[REPRESENTATIVE]` for general risk |
| Source type needed | Fleet telemetry / BMC / PDU / scheduler logs |
| Recommended final wording | `[REPRESENTATIVE] Power and thermal instability can appear as application slowdowns, node resets, retries, job failures, or correlated hardware events. Confirm with GPU telemetry, BMC logs, PDU data, cooling-system metrics, and scheduler history.` |
| Priority | P1 |

---

## 1.17 Power Delivery Chain

| Field | Validation |
|---|---|
| Claim | Power delivery includes utility/grid, transformers, switchgear, UPS, PDUs, rack power shelves, PSUs, voltage regulators, and devices. |
| Current value/formula | Infrastructure chain. |
| Validation status | Valid conceptual description. |
| Corrected value/safe wording | Keep architecture-level, not electrical-code-level. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | Data center electrical design references / vendor deployment guides |
| Recommended final wording | `[REPRESENTATIVE] AI infrastructure power delivery spans facility power sources, switchgear, UPS/PDU layers, rack power distribution, server power supplies, voltage regulation, and device-level power management. Each layer can affect reliability and sustained performance.` |
| Priority | P1 |

---

## 1.18 Cooling Loop / CDU Claims

| Field | Validation |
|---|---|
| Claim | Direct liquid cooling may involve cold plates, coolant distribution units, and facility heat rejection. |
| Current value/formula | Direct-to-chip liquid cooling architecture. |
| Validation status | Valid. |
| Corrected value/safe wording | Avoid over-specific claims unless based on vendor design. |
| Confidence label | `[REPRESENTATIVE]` |
| Source type needed | OEM/CDU/vendor cooling docs |
| Recommended final wording | `[REPRESENTATIVE] Direct-to-chip liquid cooling typically removes heat through cold plates, coolant loops, CDUs, and facility heat rejection systems, while air cooling may still remove residual heat from memory, networking, power supplies, and other components.` |
| Priority | P1 |

---

## 1.19 Cost and TCO Claims

| Field | Validation |
|---|---|
| Claim | Power and cooling materially affect TCO and cost per token. |
| Current value/formula | Cost = hardware + energy + cooling + space + operations + reliability impacts. |
| Validation status | Valid. Exact numbers environment-specific. |
| Corrected value/safe wording | Use framework, not universal cost values. |
| Confidence label | `[REPRESENTATIVE]`; measured costs `[ENV-SPECIFIC]` |
| Source type needed | TCO model, electricity rates, cloud/provider pricing |
| Recommended final wording | `[REPRESENTATIVE] At scale, cost per token depends not only on GPU price but also on energy cost, cooling overhead, power availability, utilization, failure rate, operational staffing, and amortization model.` |
| Priority | P1 |

---

## 1.20 Any Claim That Needs Confidence Labels

| Claim Type | Required Label |
|---|---|
| Official GPU/system/rack power spec | `[SHIPPED]` |
| Roadmap power/cooling requirement | `[ANNOUNCED]` |
| Power math from official values | `[DERIVED FROM SHIPPED]` |
| Facility/rack planning estimate | `[ESTIMATED]` |
| Cooling or scheduling guidance | `[REPRESENTATIVE]` |
| Real measured throttling/failure behavior | `[ENV-SPECIFIC]` |

---

# 2. P0 / P1 / P2 Validation Action List

## P0 — Must Validate Before Production Source

| Task |
|---|
| Define TDP/TBP/GPU/system/rack/facility power |
| Validate MI300X 750 W TBP |
| Validate DGX B200 ~14.3 kW system power |
| Validate GB200 NVL72 / DGX GB200 ~120 kW rack power |
| Validate PUE formula |
| Add product-level power guardrail |
| Use representative wording for air vs liquid cooling |
| Use representative wording for thermal throttling |
| Use workload-specific performance/W metrics |
| Avoid unsourced B200 per-GPU power claims |

## P1 — Strongly Recommended

| Task |
|---|
| Validate H100/H200 power values by SKU |
| Validate MI350/MI355 power values if referenced |
| Add rack power density trend with caveats |
| Validate power cap vs performance wording |
| Validate frequency/voltage tradeoff wording |
| Add power-aware scheduling guidance |
| Add power/thermal observability metrics |

## P2 — Nice to Have

| Task |
|---|
| Add tokens/W worksheet |
| Add cost-per-token power model |
| Add rack-level checklist |
| Add cooling system appendix |
| Add reliability-event examples |

---

# 3. Corrected/Safe Wording Blocks

## Power Measurement Boundary

```markdown
[REPRESENTATIVE] TDP, TBP, GPU power, system power, rack power, and facility power describe different measurement boundaries. Do not compare them unless the boundary is normalized.
```

## MI300X TBP

```markdown
[SHIPPED] AMD Instinct MI300X lists 750 W peak TBP. This is an accelerator/module-level value, not a full server or rack power value.
```

## DGX B200 System Power

```markdown
[SHIPPED] NVIDIA DGX B200 lists approximately 14.3 kW maximum system power usage. This is a system-level value, not per-GPU power.
```

## GB200 NVL72 Rack Power

```markdown
[SHIPPED] NVIDIA DGX GB200 rack-scale documentation lists approximately 120 kW rack power consumption for an NVL72 rack. This is a rack-level value and must not be compared directly to a single GPU or server.
```

## PUE

```markdown
[SHIPPED] Power Usage Effectiveness, or PUE, is total facility energy divided by IT equipment energy. PUE measures facility overhead; it does not measure tokens per watt or model efficiency.
```

## Liquid Cooling

```markdown
[REPRESENTATIVE] As rack power density rises into tens or hundreds of kilowatts, direct liquid cooling or hybrid liquid/air cooling often becomes necessary. The exact threshold depends on rack design, facility airflow, coolant infrastructure, water strategy, and service model.
```

## Power Cap

```markdown
[REPRESENTATIVE] Power caps can improve performance per watt when the workload is in a region where extra power produces diminishing throughput gains. The optimal cap is workload-, hardware-, cooling-, and SLA-specific.
```

---

# 4. Recommended Source Categories

| Source Category | Use |
|---|---|
| NVIDIA H100/H200 official datasheets | H100/H200 power values |
| AMD MI300X product page/datasheet | MI300X TBP |
| NVIDIA DGX B200 product page/user guide | DGX B200 system power |
| NVIDIA DGX GB200 rack-scale user guide | GB200 NVL72 rack power |
| ASHRAE / data center cooling guidance | Air/liquid cooling wording |
| Green Grid / DOE PUE sources | PUE definition |
| NVIDIA DCGM / BMC / telemetry docs | Power/thermal observability |
| ROCm SMI / AMD telemetry docs | AMD power/thermal observability |
| Cluster scheduler docs | Power-aware scheduling examples |
| Internal/fleet measurements | Label `[ENV-SPECIFIC]` |

---

# 5. Recommended Commit

Save this file as:

```text
publishing/validation/ch05_technical_validation.md
```

Then run:

```powershell
git add publishing\validation\ch05_technical_validation.md
git commit -m "Add Chapter 5 technical validation plan"
git push origin production-v1.0
```
