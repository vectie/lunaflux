# Matched serving attribution: what actually explains the remaining gap

## Scope and correction

This is a new diagnosis, not another compiler optimization. Production code,
deployment, and kernel selection were not changed. The measured LunaFlux source
is `d3d81ca`, the matrix-pipeline implementation; unrelated dirty-tree work is
excluded. vLLM is 0.24.0 and SGLang is 0.5.2.

**Correction to the previous explanation:** this serving bundle actually
executes standalone QKV projection followed by a separate fused
QKNorm/RoPE/KV-write kernel. The new standalone QKV pipeline **is exercised**.
Missing full fused-QKV pipeline coverage does not explain why this particular
increment improved serving by only a few percent.

All engines ran serially on RTX 5060 Ti, UUID
`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, using the same pinned Qwen3-0.6B
BF16 model files and input token IDs, greedy decoding, fixed output lengths,
prefix reuse disabled, and maximum running requests 32. Loading and startup
are excluded. LunaFlux uses its native-framed token-ID bridge; baseline
protocols and execution policies retain their existing configurations.

## Fresh unprofiled workload vector

Arithmetic mean of two measured trials after one warmup per cell. These are
output tokens/second over the complete request batch, including prefill—not
prefill tokens/second. This is a small reproducibility sample, not a confidence
interval or a universal ranking.

| Input | Output | C | LunaFlux | vLLM | SGLang |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | 256 | 1 | 245.68 | 278.42 | 265.01 |
| 59 | 256 | 8 | 1420.25 | 1849.22 | 1755.68 |
| 128 | 128 | 1 | 236.60 | 269.47 | 257.58 |
| 128 | 128 | 8 | 1314.51 | 1745.95 | 1659.71 |
| 512 | 64 | 1 | 198.76 | 244.76 | 243.35 |
| 512 | 64 | 8 | 730.39 | 1183.82 | 1132.77 |
| 1528 | 32 | 1 | 111.70 | 181.82 | 173.05 |
| 1528 | 32 | 8 | 199.22 | 444.45 | 420.71 |

Long C8 therefore remains 2.23× slower than vLLM and 2.11× slower than
SGLang on batch completion. Mean client TTFT is respectively 857.75,
223.00, and 245.69 ms. Mean client inter-token interval is 13.35, 10.84,
and 11.58 ms; these intervals include interference from requests still
prefilling and must not be interpreted as isolated decode-kernel latency.

All timed requests have the required output counts. All 16 timed long-C8
sequences match LunaFlux exactly in both baselines. Other cells do not all
have token-for-token agreement; this performance comparison is not a blanket
numerical-equivalence or model-quality certification. Raw token IDs are retained.

## GPU execution versus idle time

All three engines were freshly captured with the same CUDA-only Nsight Systems
options, including CUDA graph node tracing. Each traced cell has one warmup and
one measured trial. The client batch timestamps select the window. GPU busy
time is the union of kernel, memcpy, and memset intervals, not their sum;
overlap is not double counted. Internal idle is the first-to-last GPU span
minus that union. Edge time is the remainder outside that span, including
client/request/drain overhead. It is not exclusively scheduler time.

| 1528/32/C8, ms | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Client window | 1301.218 | 580.409 | 615.554 |
| GPU busy union | 1245.429 | 555.444 | 586.189 |
| Internal idle | 40.645 | 8.371 | 20.840 |
| Window edges | 15.143 | 16.594 | 8.526 |

The 720.81 ms traced gap to vLLM is overwhelmingly additional GPU work:
689.99 ms more GPU busy time. Eliminating every internal LunaFlux idle gap
would recover only 40.65 ms, not the missing approximately 721 ms. Host
scheduling is not the primary explanation for this workload.

This does **not** generalize to every cell. At 59/256/C1, internal idle is
122.23 ms for LunaFlux versus 22.93 ms for vLLM; linear-kernel totals are
779.99 versus 764.04 ms. That case requires a different attribution.

## Where the long-C8 GPU time goes

Conservative name-based families from actual serving kernels, in milliseconds.
Linear includes fused epilogues and vocabulary projection. Baseline generated
elementwise kernels that cannot be assigned confidently remain in `other`;
their work is not assumed absent. These are summed kernel durations, not an
additive critical-path decomposition, and fusion boundaries differ.

| Family | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Linear, including fused epilogues | 750.08 | 327.18 | 348.19 |
| Attention | 422.36 | 188.94 | 174.35 |
| Explicit QKNorm/RoPE/KV-write | 52.94 | in other/fused kernels | in other/fused kernels |
| Normalization | 19.26 | 8.62 | 22.65 |
| Sampling excluding fused vocabulary work | 0.49 | 0.10 | 0.34 |
| Unclassified other | 0.38 | 30.53 | 47.48 |

Linear and attention excess together are 656.33 ms versus vLLM, about 91%
of the traced client-window difference. This localizes the next investigation;
it does **not** prove whether the excess is redundant work, inefficient work
partitioning, memory traffic, instruction scheduling, or arithmetic throughput.

Selected actual LunaFlux launches make the attribution concrete:

| Kernel / launch variant | Calls | Total ms | Mean µs |
| --- | ---: | ---: | ---: |
| Prefill attention, grid 94×16, block 256 | 336 | 295.89 | 880.63 |
| Gate/up, grid 768, block 512 | 328 | 218.73 | 666.85 |
| QKV, grid 2048, block 256 | 336 | 216.52 | 644.42 |
| Decode attention, grid 8×8, block 256 | 812 | 111.05 | 136.77 |
| Output projection, grid 512, block 256 | 328 | 107.85 | 328.81 |
| MLP down, grid 1024, block 128 | 328 | 80.74 | 246.17 |
| QKNorm/RoPE/KV-write, grid 1024×32 | 336 | 48.46 | 144.22 |
| Vocabulary segmented greedy, rows8 | 31 | 23.75 | 766.14 |

These rows are selected variants, not complete family totals. The large QKV
variant executes 336 times: 12 layer passes for a 28-layer model. Small-row
QKV adds 868 calls, or 31 passes. Chunking and mixed prefill/decode work must
be accounted for before comparing per-launch costs. A grid dimension is not
enough to infer live rows, matrix dimensions, or useless arithmetic: the
attention kernel maps tile ordinals through actual request offsets and returns
when no tile is assigned.

The selected attention candidate is 313, query tile 32, KV tile 64, head width
128, 256 threads, 49,152 shared bytes, maximum 128 registers. Actual large
baseline GEMMs use CUTLASS schedules such as 64×64/K32 with six stages in
vLLM; this is a real schedule difference, not yet a causal proof that adding
more pipeline stages to LunaFlux will close the gap.

## Controlled prefill-budget test

The clean vLLM counterfactual uses the same model, flags and Conda environment,
with only `--max-num-batched-tokens 1024` added. Fresh startup, one warmup and
two timed trials per cell; no concurrent benchmark runner. Compilation caused
by the new budget is outside measurement.

| Case | Default vLLM tok/s | Budget 1024 tok/s | Change |
| --- | ---: | ---: | ---: |
| 59/256/C8 | 1849.22 | 1851.72 | +0.14% |
| 512/64/C8 | 1183.82 | 1168.95 | −1.26% |
| 1528/32/C1 | 181.82 | 172.51 | −5.12% |
| 1528/32/C8 | 444.45 | 429.53 | −3.36% |

At the same 1024 budget, vLLM is still **2.16×** LunaFlux's long-C8
throughput. All 16 timed long-C8 output sequences still match LunaFlux.
This refutes the hypothesis that the larger baseline budget accounts for
most of the 2.23× gap. It does not measure the benefit of increasing
LunaFlux's own budget, whose kernels may scale differently, nor does equal
budget guarantee identical request packing or execution graphs.

## Limits and next causal tests

1. Prefill-budget isolation is complete above. LunaFlux uses 1024; SGLang's
   startup log explicitly reports 2048. vLLM compiles through 2048 and its
   trace contains 2048-row KV-write launches. The capped vLLM run shows that
   matching the budget leaves most of the performance difference intact.
2. Add layer/shape attribution to the baseline trace before claiming an exact
   QKV-versus-QKV or output-versus-output ratio. Generic CUTLASS names and
   grid geometry alone do not establish the logical operation. Existing NVTX
   capture contained library annotations, not sufficient model-layer labels.
3. Measure instruction/memory/occupancy counters on the **selected** long
   projection and attention launches. Registers and shared-memory size alone
   cannot identify the limiting resource or justify a compiler transformation.

The evidence supports prioritizing linear and attention execution. It does
not support another assertion that a particular unmeasured compiler feature
will provide the missing 2×.

## Collection caveats and artifacts

LunaFlux's first richer CUDA/NVTX/OS-runtime tracing attempt aborted before
readiness. It is excluded. CUDA-only tracing succeeded; all final compared
traces use those identical tracing options. To trace the spawned worker, a
separate diagnostic supervisor forwards profiler injection variables. Worker,
kernel artifacts and model are unchanged. Unprofiled measurements use the
original production supervisor.

The first budget experiment omitted Conda's bin directory from PATH and
failed to find ninja during compilation. Its waiting health poll subsequently
attached to the second launch; the second runner encountered the first
runner's result-directory collision and shut down. Both attempts are excluded.
A clean retry uses a unique output directory after confirming both earlier
runners and GPU processes have exited. No partial output counts are admitted.

Remote root: `/run/user/1000/lunaflux-matched-attribution-20260909-r1`.
Source archive SHA-256:
`bcfec0602ee66bbe61f27031763ca678fe69b352ec6e8a716e77b4b9cb0f5480`.
The report's companion archive retains raw CUDA traces, SQLite exports,
per-kernel geometry and timing tables, raw SSE/token IDs, client timestamps,
launch arguments, analysis scripts, and the diagnostic supervisor diff.

Local archive:
`/private/tmp/lunaflux-attribution-results-vIcHdI/attribution-results-r1.tar.gz`.
Archive SHA-256:
`3eb27050ee1c1b229c54793a1de4dca9530614edb42fbf896e7882736bf46cb0`.
