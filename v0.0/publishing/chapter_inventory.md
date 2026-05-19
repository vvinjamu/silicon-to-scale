# Chapter Inventory — Production v1.0

| ID | Title | PDF | Source File | Diagrams | Status | P0 Issues |
|---|---|---|---|---|---|---|
| Ch00 | Front Matter | pdfs/ch00_front_matter.pdf | TBD | TBD | Present | Final copyright, ISBN, disclaimer check |
| Ch01 | AI/ML Performance Architecture Mindset | pdfs/ch01_performance_mindset.pdf | TBD | Roofline, performance stack | Present | Needs figure integration check |
| Ch02 | Transformer Architecture Deep Dive | pdfs/ch02_transformer_architecture.pdf | TBD | Transformer block, attention flow | Present | Needs formula/diagram consistency check |
| Ch03A | GPU Architecture Fundamentals | pdfs/ch03a_gpu_architecture.pdf | TBD | GPU memory, CUDA hierarchy | Present | Hardware spec validation |
| Ch03B | GPU Architecture Roadmap | pdfs/ch03b_gpu_roadmap.pdf | TBD | GPU generation timeline | Present | High-priority roadmap validation |
| Ch04 | Memory Hierarchy and HBM Deep Dive | pdfs/ch04_memory_hierarchy_hbm.pdf | TBD | Memory pyramid, HBM stack | Present | HBM spec validation |
| Ch05 | Thermal Design, Power, and TCO | pdfs/ch05_thermal_power_tco.pdf | TBD | TCO breakdown | Present | Print tables and TCO assumptions |
| Ch06 | Inference Systems Architecture | pdfs/ch06_inference_systems.pdf | TBD | Prefill/decode, batching | Present | Serving-framework claim validation |
| Ch07 | GPU Kernels and CUDA Optimization | pdfs/ch07_gpu_kernels_cuda.pdf | TBD | FlashAttention, CUDA hierarchy | Present | Dense code/diagram print risk |
| Ch08 | Quantization | pdfs/ch08_quantization.pdf | TBD | Precision ladder, error visual | Present | Validation of FP8/INT4 claims |
| Ch09 | Operator Fusion and Compiler Optimization | pdfs/ch09_operator_fusion_compiler.pdf | TBD | Compiler stack | Present | torch.compile version validation |
| Ch10 | Training Systems Architecture at Scale | pdfs/ch10_training_systems.pdf | TBD | 4D parallelism, ZeRO | Present | Parallelism formulas and diagrams |
| Ch11 | KV Cache | pdfs/ch11_kv_cache_complete.pdf | TBD | KV cache, PagedAttention | Present | Flagship chapter; needs polish and validation |
| Ch12 | Benchmarking and Performance Measurement | pdfs/ch12_benchmarking_measurement.pdf | TBD | Benchmark harness, regression CI | Present | Add benchmark smell-test checklist |
| Ch13 | Speculative Decoding and MoE Infrastructure | pdfs/ch13_speculative_decoding_moe.pdf | TBD | Spec decoding, MoE routing | Present | Split dense topics clearly |
| Ch14 | Networking and Collective Communication | pdfs/ch14_networking_collectives.pdf | TBD | Ring AllReduce, Fat-Tree | Present | NCCL/network claim validation |
| Ch15 | Storage, Data Pipelines, and I/O | pdfs/ch15_storage_data_io.pdf | TBD | Storage pipeline | Present | Needs stronger visuals |
| Ch16 | Cluster Management and Multi-Tenant Systems | pdfs/ch16_cluster_management.pdf | TBD | Scheduler, tenancy | Present | Add SLURM vs Kubernetes matrix |
| Ch17 | Telemetry, Observability, and Profiling | pdfs/ch17_observability_profiling.pdf | TBD | Observability stack | Present | Add alert severity table |
| Ch18 | Principal Architect Career and Interviews | pdfs/ch18_principal_architect.pdf | TBD | System design maps | Present | Resolve duplicate Ch18 file |
| AppA | Hardware Reference Tables | pdfs/appendix_a_hardware_reference.pdf | TBD | Tables | Present | Must validate before publishing |
| AppB | Profiling Tool Quick Reference | pdfs/appendix_b_tool_reference.pdf | TBD | Tool tables | Present | Convert to quick lookup |
| AppC | Glossary | pdfs/appendix_c_glossary.pdf | TBD | None | Present | Add cross-references |
