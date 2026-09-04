# Compiler-kernel benchmark — updated 2026-09-04

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
| query-row result demand (`cf8883d`) | LM head | functional demand analysis prunes logits for prompt rows that cannot be sampled | one logical LM-head row per request | Accepted after the mixed-ABI fix; drives the large 128/512-token prefill gains below |
| functional projection middle end (`dc91805`, `947474f`) | projection | immutable strategy → tile program → rewrite → schedule pipeline | backend-neutral schedule consumed by CUDA lowering | Accepted structurally; no direct speed claim independent of its lowered kernels |
| cooperative projection tile (`7d9a2c9`) | prefill projection | workgroup-reused input/weight tiles | 64 × 32 × 16, eight warps | Rejected from the mixed prefill/decode entry point: changing its shared-memory and grid contract made the worker exit during longer decode |
| phase-safe mixed lowering (`608db7a`) | projection | compiler keeps the cooperative plan, but mixed ABI lowers through the established WMMA/GEMV entry point | 8,192 B shared for mixed QKV/dense | Accepted: c32 smoke and 64-token probe pass; complete 12-point vector finishes without worker loss |

The 32-token quick-vector medians above use four trials and eight successful
requests per trial. They are candidate gates, not substitutes for the complete
256-token decode vector below.

The padded-matrix experiment was stopped after its first complete eight-request
trial because the regression was far outside run-to-run noise. Padding M=1 to
M=16 spent matrix throughput on fifteen dead rows and added a shared-memory
barrier for every reduction tile; fewer launches did not compensate for that
work amplification.

## Latest accepted token vector

The accepted compiler-only LunaFlux result is source commit `608db7a`, source
archive SHA-256 `205a7c1c…7e1aa`. Each row is one `(input tokens, maximum
output tokens, concurrency)` point. `Change` compares LunaFlux with the prior
`72e53a6` vector; positive values are faster.

| Input | Output | Concurrency | LunaFlux tok/s | Change | vLLM tok/s | SGLang tok/s | llama.cpp tok/s | LunaFlux TTFT | LunaFlux ITL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | 256 | 1 | 122.665 | +0.1% | 279.118 | 269.837 | 83.309 | 19.028 | 7.896 |
| 59 | 256 | 8 | 544.235 | -1.3% | 1,865.623 | 1,793.475 | 30.889 | 76.502 | 14.320 |
| 59 | 256 | 32 | 1,091.658 | -9.4% | 5,263.451 | 1,965.078 | 28.578 | 169.861 | 28.121 |
| 128 | 128 | 1 | 121.573 | +184.9% | 274.895 | 262.502 | 87.382 | 24.122 | 7.879 |
| 128 | 128 | 8 | 517.813 | +257.2% | 1,776.647 | 1,713.128 | 30.428 | 104.785 | 14.407 |
| 128 | 128 | 32 | 974.359 | +355.4% | 4,746.347 | 1,886.328 | 27.931 | 462.606 | 28.298 |
| 512 | 64 | 1 | 83.264 | +106.7% | 255.304 | 249.478 | 31.041 | 68.316 | 11.100 |
| 512 | 64 | 8 | 319.303 | +170.6% | 1,212.893 | 1,160.051 | 22.931 | 383.243 | 17.136 |
| 512 | 64 | 32 | 444.627 | +185.9% | 2,086.492 | 1,409.170 | 20.085 | 1,419.975 | 40.074 |
| 1,528 | 32 | 1 | 35.763 | +0.1% | 189.162 | 183.968 | 4.719 | 231.622 | 21.138 |
| 1,528 | 32 | 8 | 86.557 | +2.1% | 463.076 | 425.355 | 4.648 | 1,064.516 | 41.360 |
| 1,528 | 32 | 32 | 101.741 | +1.7% | 521.568 | 489.844 | 4.635 | 4,060.388 | 92.365 |

All 896 LunaFlux requests completed successfully. Across the 896 four-engine
joins, 202 have the same complete output-token hash and 694 diverge. The latter
include rows where all four engines have different greedy hashes, so the
published harness correctly marks `speed_comparison_valid=false`. The numbers
above are descriptive throughput and latency measurements until correctness is
compared against a numerical/reference tolerance rather than requiring every
long greedy continuation to remain bit-identical.

## Interpretation

- Compiler attention is already materially faster than llama.cpp on the long
  prompt and concurrent vectors, and faster at the short decode c1 point.
- Query-row demand pruning removes most of the earlier 128/512-token LM-head
  waste. It does not materially move the 1,528-token points, where attention
  and MLP now dominate.
- LunaFlux remains behind vLLM and SGLang, especially at c32 and long prefill.
  The next structural work is a distinct prefill ABI with batch-wide tiled
  projection/MLP and long-prefill tiled attention; the cooperative projection
  kernel must not be forced through the decode entry point.
- A candidate is accepted only when the full token vector improves without a
  material regression in another region. Source-size reductions or isolated
  instruction changes are not performance claims by themselves.
