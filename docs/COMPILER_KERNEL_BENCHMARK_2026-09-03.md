# Compiler-kernel benchmark — 2026-09-03

This report tracks the Qwen3-0.6B compiler-only path on the same RTX 5060 Ti,
model files, HTTP streaming protocol, greedy sampling policy, and tokenized
workload. Throughput is end-to-end output tokens per second. TTFT and ITL are
milliseconds. Results from different rows must not be combined unless the
complete workload vector matches.

## Kernel evolution

| Revision / candidate | Phase | Compiler strategy and lowering | Launch / tile | Result |
| --- | --- | --- | --- | --- |
| attention 310 | prefill | `AttentionMatrixQkPv`, shared Q/K/V tiles, matrix QK and PV | Q16 × KV32 × H128; 256 threads; 31,952 B shared | Active compiler prefill kernel; no isolated end-to-end A/B |
| attention 400 | decode | scalar direct-global diagnostic | one subgroup per Q head; grid 32 × 16 | Rejected: structurally serialized and slower |
| attention 410 | decode | subgroup dot with staged K/V | Q1 × KV64 × H128; 128 threads | Superseded by grouped-query reuse |
| attention 420 (`4f40938`) | decode | one subgroup per Q head, K/V shared across each GQA group | Q1 × KV64 × H128; 128 threads | 80.895 / 422.085 / 757.397 tok/s at c1/c8/c32 |
| attention 430 (`fc50a39`) | decode | two subgroups per Q head plus final online-softmax merge | Q1 × KV64 × H128; 256 threads; 34,076 B shared | 119.199 / 530.447 / 1,196.400 tok/s; +47.4% / +25.7% / +58.0% over 420 |
| graph-bucket launch (`72e53a6`) | decode | compiler kernel retained; phase-specific graph launch geometry | per-bucket decode grid | 122.531 / 551.152 / 1,205.002 tok/s; +2.8% / +3.9% / +0.7% over 430 |
| vectorized BF16x2 projection (`03645e1`, fixed by `fda4c25`) | decode projection | two BF16 values per load in subgroup GEMV | same block resources, smaller cubin | Rejected: 113.80 versus 122.61 tok/s on the same 32-token c1 quick vector, about -7.2% |
| explicit decode FMA (`c967d5b`) | decode projection | source-level `fmaf` while global reassociation remains disabled | unchanged subgroup GEMV | Rejected as neutral: median 122.623 versus 122.612 tok/s, +0.009% |
| padded matrix GEMV (`29155bd`) | decode projection | backend-neutral `PaddedMatrixGemv`; CUDA pads one live row to 16 × 16 WMMA | dense: 128 outputs/block; MLP: 256 outputs/block | Rejected after trial 1: 54.168 versus 122.61 tok/s, about -55.8%; output-token hash also changed |

The 32-token quick-vector medians above use four trials and eight successful
requests per trial. They are candidate gates, not substitutes for the complete
256-token decode vector below.

The padded-matrix experiment was stopped after its first complete eight-request
trial because the regression was far outside run-to-run noise. Padding M=1 to
M=16 spent matrix throughput on fifteen dead rows and added a shared-memory
barrier for every reduction tile; fewer launches did not compensate for that
work amplification.

## Latest accepted token vector

The accepted compiler-only LunaFlux result is `72e53a6`/`44652a0`. Each row is
one `(input tokens, maximum output tokens, concurrency)` point.

| Input | Output | Concurrency | LunaFlux tok/s | vLLM tok/s | SGLang tok/s | llama.cpp tok/s | LunaFlux TTFT | LunaFlux ITL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | 256 | 1 | 122.531 | 279.118 | 269.837 | 83.309 | 19.062 | 7.895 |
| 59 | 256 | 8 | 551.152 | 1,865.623 | 1,793.475 | 30.889 | 87.552 | 14.078 |
| 59 | 256 | 32 | 1,205.002 | 5,263.451 | 1,965.078 | 28.578 | 182.063 | 25.350 |
| 128 | 128 | 1 | 42.673 | 274.895 | 262.502 | 87.382 | 231.507 | 21.989 |
| 128 | 128 | 8 | 144.961 | 1,776.647 | 1,713.128 | 30.428 | 1,103.530 | 42.060 |
| 128 | 128 | 32 | 213.947 | 4,746.347 | 1,886.328 | 27.931 | 4,249.298 | 92.109 |
| 512 | 64 | 1 | 40.291 | 255.304 | 249.478 | 31.041 | 232.108 | 21.262 |
| 512 | 64 | 8 | 117.986 | 1,212.893 | 1,160.051 | 22.931 | 1,059.831 | 41.267 |
| 512 | 64 | 32 | 155.514 | 2,086.492 | 1,409.170 | 20.085 | 4,202.141 | 90.854 |
| 1,528 | 32 | 1 | 35.732 | 189.162 | 183.968 | 4.719 | 231.516 | 21.153 |
| 1,528 | 32 | 8 | 84.786 | 463.076 | 425.355 | 4.648 | 1,124.408 | 41.329 |
| 1,528 | 32 | 32 | 100.078 | 521.568 | 489.844 | 4.635 | 4,228.509 | 89.532 |

All 896 LunaFlux requests completed successfully. The joined comparison reports
zero exact four-engine output-token hashes, so these measurements establish
throughput and latency only; they do not establish identical generated token
sequences across all four engines.

## Interpretation

- Compiler attention is already materially faster than llama.cpp on the long
  prompt and concurrent vectors, and faster at the short decode c1 point.
- LunaFlux remains behind vLLM and SGLang, especially for concurrent prefill.
  That gap is consistent with projection/MLP tiling and batch-wide GEMM
  utilization, not with HTTP or security checks in the token step.
- A candidate is accepted only when the full token vector improves without a
  material regression in another region. Source-size reductions or isolated
  instruction changes are not performance claims by themselves.
