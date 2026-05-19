# AI/ML Infrastructure from Silicon to Scale
### The Principal Performance Architect's Complete Guide

**Author:** Venkat Vinjam · Principal Performance Architect · AMD | Ex-Intel, Cisco, IBM  
**Edition:** First Edition, 2026  
**License:** Free for personal and educational use

---

## 🌐 Read Online

**[silicon-to-scale.github.io](https://vvinjamu.github.io/silicon-to-scale)**

No signup. No account. Open the link and start reading.

---

## 📖 About This Book

A generational reference for engineers who need both deep technical grounding and architectural decision-making frameworks — covering the complete stack from GPU silicon physics to 10,000-GPU fleet operations.

**Who it's for:**
- Senior/Principal Performance Engineers moving into AI infrastructure
- ML Engineers who want to understand the systems beneath their code
- Infrastructure Engineers scaling LLM serving or training to production
- Anyone preparing for Principal-level AI infrastructure interviews

**What makes it different:**
- Every number is a real production number (with [SHIPPED]/[ANNOUNCED] labels)
- Principal-level review questions with full answers in every chapter
- Complete worked examples: ChatGPT at scale, LLaMA-3 70B sizing, full parallelism configs
- Not a survey — a field guide from 20 years of performance architecture experience

---

## 📁 Repository Structure

```
silicon-to-scale/
│
├── index.html                    ← START HERE — main landing page
│
├── pdfs/                         ← All 18 chapters + 3 appendices as PDFs
│   ├── ch01_performance_mindset.pdf
│   ├── ch02_transformer_deep_dive.pdf
│   ├── ch03a_gpu_fundamentals.pdf
│   ├── ch03b_gpu_roadmap.pdf
│   ├── ch04_memory_hierarchy.pdf
│   ├── ch05_thermal_power_tco.pdf
│   ├── ch06_inference_systems.pdf
│   ├── ch07_gpu_kernels.pdf
│   ├── ch08_quantization.pdf
│   ├── ch09_fusion_compiler.pdf
│   ├── ch10_training_complete.pdf
│   ├── ch11_kv_cache_complete.pdf
│   ├── ch12_benchmarking_complete.pdf
│   ├── ch13_speculative_moe.pdf
│   ├── ch14_networking.pdf
│   ├── ch15_storage_io.pdf
│   ├── ch16_cluster_management.pdf
│   ├── ch17_telemetry_tools.pdf
│   ├── ch18_principal_career.pdf
│   ├── appendix_a_hardware_reference.pdf
│   ├── appendix_b_tools_reference.pdf
│   └── appendix_c_glossary.pdf
│
├── diagrams/                     ← 8 standalone interactive HTML diagrams
│   ├── diagram_01_memory_hierarchy.html
│   ├── diagram_02_kv_cache.html
│   ├── diagram_03_transformer_pipeline.html
│   ├── diagram_04_compiler_stack.html
│   ├── diagram_05_parallelism_topology.html
│   ├── diagram_06_speculative_decoding.html
│   ├── diagram_07_moe_routing.html
│   └── diagram_08_observability_stack.html
│
└── README.md                     ← This file
```

---

## 📚 Chapter Overview

| # | Chapter | Key Topics |
|---|---------|-----------|
| **Part I — Foundations** | | |
| 1 | The Performance Architecture Mindset | Roofline, 7-layer stack, MFU, measurement discipline |
| 2 | Transformer Architecture Deep Dive | Attention math, GQA/MLA, RoPE, SwiGLU, scaling laws |
| 3A | GPU Architecture — Fundamentals | SM, Tensor Cores, NVLink, memory hierarchy |
| 3B | GPU Roadmap | H100→B200→MI300X, hardware selection matrix |
| **Part II — Hardware Performance** | | |
| 4 | Memory Hierarchy & HBM | HBM physics, FlashAttention, KV sizing |
| 5 | Thermal, Power & TCO | Cooling physics, DVFS, cost-per-token, build vs buy |
| **Part III — LLM Inference** | | |
| 6 | Inference Systems | Continuous batching, vLLM, PagedAttention, P-D disagg |
| 7 | GPU Kernels | GEMM tiling, FlashAttention proof, Triton, CUDA Graphs |
| 8 | Quantization | FP8, GPTQ, AWQ, KV quant, QLoRA — decision framework |
| 9 | Fusion & Compiler Stack | torch.compile, TensorRT, NVFuser, min-cut recomputation |
| **Part IV — Distributed Training** | | |
| 10 | Distributed Training Systems | 4D parallelism, ZeRO/FSDP, fault tolerance, RLHF |
| 11 | KV Cache | PagedAttention, prefix caching, MLA (57× compression) |
| 12 | Benchmarking Methodology | 5 measurement rules, MLPerf, 5-step decision tree |
| **Part V — Advanced & Operations** | | |
| 13 | Speculative Decoding & MoE | Speedup math, EAGLE, MoE routing, load balance |
| 14 | Networking | NVLink, InfiniBand, RoCEv2, NCCL, rail topology |
| 15 | Storage & Data Pipelines | Lustre, async checkpoints, sequence packing |
| 16 | Cluster Management | SLURM, Kubernetes, gang scheduling, multi-tenancy |
| 17 | Telemetry & Observability | DCGM, Prometheus, Grafana, straggler detection |
| **Part VI — Career** | | |
| 18 | The Principal Architect | 4 superpowers, STAR+Impact, system design walkthrough |

---

## 🖼️ Architecture Diagrams

Eight production-quality interactive diagrams (HTML/SVG):

| Diagram | Chapter | What it shows |
|---------|---------|---------------|
| Memory Hierarchy | Ch04 | Registers → SMEM → L2 → HBM → DRAM with exact bandwidth |
| KV Cache | Ch11 | PagedAttention blocks, size growth, cascade cliff at 82% |
| Transformer Block | Ch02 | FLOPs labeled at every op, residual paths, KV cache arrow |
| Compiler Stack | Ch09 | Dynamo→Inductor→Triton JIT with speedup source breakdown |
| 4D Parallelism | Ch10 | TP/PP/DP on physical H100 nodes with communication budget |
| Speculative Decoding | Ch13 | Draft→verify→accept/reject with exact min(1,p/q) math |
| MoE Routing | Ch13 | Router, All-to-All dispatch, load imbalance warning |
| Observability Stack | Ch17 | 7 layers from silicon to CI with alert thresholds |

Open any diagram directly in a browser — no server required.

---

## 🚀 How to Deploy on GitHub Pages

### Step 1 — Create the repository

```bash
# On GitHub: create a new public repo named "silicon-to-scale"
# Then locally:
git clone https://github.com/vvinjamu/silicon-to-scale.git
cd silicon-to-scale
```

### Step 2 — Add all files

```bash
# Copy files to the repository (use the structure above)
cp index.html .
mkdir pdfs diagrams
cp /path/to/pdfs/*.pdf  pdfs/
cp /path/to/diagrams/*.html  diagrams/
```

### Step 3 — Commit and push

```bash
git add .
git commit -m "Initial release: AI/ML Infrastructure from Silicon to Scale"
git push origin main
```

### Step 4 — Enable GitHub Pages

1. Go to your repo on GitHub
2. Click **Settings** → **Pages** (left sidebar)
3. Under **Source**, select: `main` branch, `/ (root)` folder
4. Click **Save**
5. Your site will be live at: `https://vvinjamu.github.io/silicon-to-scale`

> ⏱ GitHub Pages usually goes live within 1–5 minutes after enabling.

### Step 5 — Update links in index.html

Open `index.html` and replace these two placeholders:

```html
<!-- Line ~240: replace with your actual GitHub URL -->
href="https://github.com/vvinjamu/silicon-to-scale"

<!-- Line ~490: same -->
href="https://github.com/vvinjamu/silicon-to-scale"

<!-- Line ~492: your LinkedIn -->
href="https://www.linkedin.com/in/vinjam-venkateswarlu-a751332/"

<!-- Line ~494: your email -->
href="mailto:vinjam5678@gmail.com"
```

---

## 🖼️ How to Convert Markdown Chapters to PDF

The source chapters are in Markdown (`.md`). Convert them to PDF for the `pdfs/` folder using any of these methods:

### Method A — Chrome (recommended, preserves formatting)
```bash
# Install markdown-to-html converter
npm install -g markdown-it

# Convert a chapter to HTML then PDF
markdown-it ch01_performance_mindset.md > ch01.html
# Open ch01.html in Chrome → File → Print → Save as PDF
```

### Method B — Pandoc + LaTeX (highest quality)
```bash
brew install pandoc basictex   # macOS
pandoc ch01_performance_mindset.md \
  -o pdfs/ch01_performance_mindset.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt
```

### Method C — VS Code (easiest)
1. Open `.md` file in VS Code
2. Install extension: **Markdown PDF** (yzane)
3. Right-click in editor → **Markdown PDF: Export (pdf)**
4. Move exported file to `pdfs/` folder

### Method D — Online converter
Upload to [pdf24.org](https://www.pdf24.org) or [pandoc.org/try](https://pandoc.org/try/)

---

## 📍 Where to Embed the Diagrams in Your PDFs

Each diagram belongs at the beginning of its chapter, right after the chapter title and overview section, before the first numbered section:

| Diagram file | Insert in chapter | Placement |
|---|---|---|
| `diagram_01_memory_hierarchy.html` | Chapter 04 | After §4.0 overview, before §4.1 |
| `diagram_02_kv_cache.html` | Chapter 11 | After §11.0 overview, before §11.1 |
| `diagram_03_transformer_pipeline.html` | Chapter 02 | After §2.0 overview, before §2.1 |
| `diagram_04_compiler_stack.html` | Chapter 09 | After §9.0 overview, before §9.1 |
| `diagram_05_parallelism_topology.html` | Chapter 10 | Before §10.2 (4D Parallelism section) |
| `diagram_06_speculative_decoding.html` | Chapter 13 | Before §13.2 (Spec Decode algorithm) |
| `diagram_07_moe_routing.html` | Chapter 13 | Before §13.3 (MoE architecture) |
| `diagram_08_observability_stack.html` | Chapter 17 | After §17.0 overview, before §17.1 |

### To embed a diagram in a Markdown chapter:

**Option 1: Screenshot the HTML diagram → embed as image**
```markdown
![GPU Memory Hierarchy — H100 SXM5](../diagrams/diagram_01_memory_hierarchy.png)
*Figure 4.1 — GPU Memory Hierarchy from Registers to Host DRAM*
```

**Option 2: Include as iframe in HTML output**
```html
<iframe src="../diagrams/diagram_01_memory_hierarchy.html" 
        width="860" height="540" frameborder="0"></iframe>
<p><em>Figure 4.1 — GPU Memory Hierarchy from Registers to Host DRAM</em></p>
```

**Option 3: Screenshot workflow (for PDF)**
1. Open the `.html` diagram in Chrome
2. Press `Cmd+Shift+4` (Mac) or `Snipping Tool` (Windows)
3. Capture at 2× zoom for retina quality
4. Save as `diagram_01_memory_hierarchy.png` in `diagrams/`
5. Reference in Markdown as shown in Option 1

---

## 🔑 Key Numbers (Quick Reference)

```
H100 SXM5:   989 TFLOPS BF16  ·  3.35 TB/s HBM  ·  80 GB  ·  700W
H200 SXM5:   989 TFLOPS BF16  ·  4.8 TB/s HBM   · 141 GB  ·  700W
B200 SXM6:   ~2,250 TFLOPS    ·  8.0 TB/s HBM   · 192 GB  · 1000W
MI300X:      1,307 TFLOPS     ·  5.3 TB/s HBM   · 192 GB  ·  750W

TP rule:     ALWAYS within NVLink domain (≤8 GPUs)
Training:    16 bytes/param for full AdamW state
Decode:      model_bytes / HBM_bandwidth = TPOT lower bound
KV alert:    82% utilization (cascade cliff starts at 88%)
MFU target:  45–60% for H100 cluster (< 40% = investigate)
```

---

## 💬 Feedback and Contributions

Found an error? Have a suggestion? 

- **GitHub Issues:** [Open an issue](https://github.com/vvinjamu/silicon-to-scale/issues)
- **LinkedIn:** [Venkat Vinjam](https://www.linkedin.com/in/vinjam-venkateswarlu-a751332/)

If this book helped you prepare for an interview, understand a system, or ship a production ML workload — a ⭐ star on GitHub is appreciated and helps others find the resource.

---

## 📜 License

This book is released free for personal and educational use.

- ✅ Read, share, and reference for personal learning
- ✅ Use in educational courses with attribution
- ✅ Quote with attribution ("Vinjam, *AI/ML Infrastructure from Silicon to Scale*, 2026")
- ❌ Commercial reproduction without permission
- ❌ Re-publishing as your own work

---

*"The engineer who can name the bottleneck before the profiling tool is opened — who can size the fleet before the whiteboard marker is uncapped — is worth more than the sum of their tools."*

**— Venkat Vinjam, Chapter 18**

