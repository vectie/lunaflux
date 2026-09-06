# Decode K/V transfer vectorization — 2026-09-06

The functional attention schedule already specified an 8-byte contiguous
column map, but grouped split-subgroup CUDA decode emitted scalar BF16 K/V
loads and stores. Commit `d33a707` realizes that existing schedule in both
direct and split-partial kernels. No model-name branch, new scheduler policy,
tile-size change, shared-memory increase, or softmax reassociation is involved.

## Matched compiler experiment

The scalar control was generated from parent `f698b70` with the same new
test-only serving-ABI exporter as the vector version. The generated sources
differ only in the K/V staging loop. This is a stronger isolation than comparing
against the previously deployed attention module, whose direct entry point
predates empty-partition merge guards. Both comparisons were run; the table
below uses the same-compiler scalar control.

RTX 5060 Ti (`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`), CUDA 13.1.115,
`sm_120`, `-O3 --fmad=false --maxrregcount=128`. Shape: 16 query heads,
8 KV heads, head dimension 128, page size 8, KV tile 64, block 256.
Compact grid rows equal B. Fixed-context, graph-replayed warm operator timing:
200 invocations per graph, three warm replays and one timed replay, repeated
in three processes per shape/version. Version order is old/new, new/old,
old/new. Values are arithmetic means in microseconds, not serving latency.

| B | Context | Route | Scalar µs | Vector µs | Speedup |
|---:|---:|---|---:|---:|---:|
| 1 | 59 | direct | 12.737 | 8.832 | 1.44× |
| 1 | 128 | split8 + merge | 15.106 | 11.112 | 1.36× |
| 1 | 256 | split8 + merge | 15.454 | 11.300 | 1.37× |
| 1 | 512 | split8 + merge | 15.638 | 11.539 | 1.36× |
| 1 | 1528 | split8 + merge | 39.844 | 27.274 | 1.46× |
| 1 | 4096 | split8 + merge | 100.946 | 67.226 | 1.50× |
| 8 | 59 | direct | 13.708 | 9.750 | 1.41× |
| 8 | 128 | direct | 26.878 | 18.202 | 1.48× |
| 8 | 256 | direct | 51.966 | 34.386 | 1.51× |
| 8 | 512 | direct | 101.557 | 66.659 | 1.52× |
| 8 | 1528 | direct | 428.300 | 240.338 | 1.78× |
| 8 | 4096 | direct | 1144.765 | 620.977 | 1.84× |

Routes here are measured independently, not a new dispatch selection policy.
In particular, the 128-token B1 split row does not imply that production selects
split at that exact context. Full raw CSV also includes direct B1, split B8,
and separate partial/merge measurements. B8 split is still slower than direct
at 1528 (243.041 versus 240.338 µs); this change does not widen its serving
dispatch threshold.

The unchanged merge stage remains about 1.37–1.48 µs at B1 and 2.30–2.36 µs
at B8. The gain is in K/V staging and the partial/direct attention kernels,
not the merge. `cuobjdump` confirms `LDG.E.64` and `STS.64` at staging sites.
Registers are 48 for direct and 52 for partial, with zero local memory and
zero stack. Dynamic shared memory remains 34,076 bytes (the compiler also
reports 1,024 static shared bytes).

## Correctness and scope

- Warning-denied native check and `moon info` pass; affected package tests:
  32/32; scoped formatting and diff checks pass.
- Scalar/vector full BF16 output buffers compare bit-for-bit at contexts
  1, 7, 8, 9, 59, 63, 64, 65, 128, 256, 512, 1528, 4096 (B1), plus
  mixed prefill/decode at B2/context256 and B8/context1528. Both direct and
  split8 are compared, including empty partitions and untouched prefill rows.
- An independent double-precision paged-attention referee passes at the
  existing 0.004 absolute tolerance. The broad boundary run's maximum observed
  error is below 0.000975; bit equality is asserted between compiler versions,
  not between floating-point orders and the double referee.
- The timing matrix additionally compares every result to that referee and
  verifies unchanged complete K/V buffers and explicit resource release.
- Memcheck, racecheck, synccheck and initcheck pass at B8/context59 with the
  exact new module, including its masked tile tail and empty split partitions.
  These sanitizer timings are excluded from performance results.
- Head64 and non-vector-aligned stride fallback have software source tests;
  this campaign does not claim physical coverage of every supported shape.

This is **general schedule-driven vectorization with CUDA-specific lowering**.
It is not an additional functional rewrite of softmax, an async-copy pipeline,
or a model-specific optimization. The implementation still synchronously stages
one tile at a time. Wider transfers, staged overlap and resource-aware pipeline
selection remain subsequent independent experiments.

## Reproduction and raw results

The test exporter is `tests/attention_tile_cuda_source_probe` with argument
`decode-serving`. Its `decode_probe.cu` accepts
`SCALAR_MODULE VECTOR_MODULE unused 1 compare` for the bitwise boundary run.
The graph timing harness and MoonBit orchestration are preserved with raw output.

- Matched compiler results: `/dev/shm/lunaflux-vector-control-20260906-r1`
  on the test host and `/private/tmp/lunaflux-vector-control-20260906-r1` locally.
- Earlier deployed-module comparison and four sanitizers:
  `/dev/shm/lunaflux-vector-decode-20260906-r1`, with the same basename locally.
- New CUBIN SHA-256:
  `f65f760f8edf0b205db3747d087f47e021d891560725104efb4dd580bba281d2`.
- Matched scalar CUBIN SHA-256:
  `3590c6ec80f3d0c8e30cb69159842c65b71b8759b045c897d2d87ea46f3f3d34`.
- New generated CUDA SHA-256:
  `292d7f993a41e19b4467008546c6c4bed5c61f59f632050494f1326fffd89344`.

## Qwen end-to-end check

The new module was bound into a separate Qwen3-0.6B BF16 benchmark deployment.
Comparing the old and new runtime bundles shows exactly two changed fields:
`module_8_module_sha256` and `module_8_module_hex`. Worker/launcher binaries,
other eight modules, graph geometry, model, token vectors, greedy/EOS policy,
and benchmark client remain unchanged. The runtime execution baseline is the
one described in [the four-engine report](BASELINE_BOTTLENECK_COMPARISON_2026-09-06.md).
This is a new compiler artifact exercised through the existing whole-model
serving path, not a claim that unrelated working-tree code was deployed.

Both versions were run afresh, serially, without profiling: one warmup plus two
measured trials per shape/concurrency. New then old server order was used;
unlike the operator experiment, these server runs are not counterbalanced.
Throughput is completed output tokens divided by batch wall time, including
prefill. C8 means concurrent requests, not constant batch size every step.

| Input → output | C | Old tok/s | New tok/s | Gain | Old decode ms/token | New decode ms/token |
|---|---:|---:|---:|---:|---:|---:|
| 59 → 256 | 1 | 208.9 | 221.6 | 6.1% | 4.704 | 4.443 |
| 59 → 256 | 8 | 701.5 | 738.6 | 5.3% | 11.213 | 10.638 |
| 128 → 128 | 1 | 204.0 | 213.9 | 4.8% | 4.697 | 4.480 |
| 128 → 128 | 8 | 660.2 | 693.1 | 5.0% | 11.356 | 10.775 |
| 512 → 64 | 1 | 162.2 | 172.7 | 6.5% | 5.198 | 4.786 |
| 512 → 64 | 8 | 383.2 | 418.5 | 9.2% | 15.681 | 13.871 |
| 1528 → 32 | 1 | 79.3 | 84.4 | 6.5% | 6.016 | 5.290 |
| 1528 → 32 | 8 | 108.9 | 116.4 | 6.9% | 40.567 | 35.927 |

C8 mean TTFT stays effectively unchanged: 49.0→49.7, 91.5→92.0,
342.5→343.1, and 1042.6→1042.1 ms respectively. Prefill has not been optimized
in this change. The 1.4–1.8× isolated attention speedup must not be described
as a whole-model speedup; projections, prefill, other kernels and serving
overhead still account for most execution time.

All requests produced the exact requested token count and terminal event.
Every C1 old/new sequence matches. At C8, 128/128, 512/64 and 1528/32 each
match 16/16 request-index sequences across the measured trials; 59/256 matches
6/16. Request index does not fix internal scheduling order. The previous report
already observed within-version C8 sequence variability; this comparison alone
does not attribute the differences to a particular cause or prove quality
equivalence. The controlled operator inputs independently remain bit-identical.

vLLM, SGLang and llama.cpp were **not rerun** in this optimization experiment.
Using their earlier same-day measurements only as context, 59/256 C8 remains
about 2.50× behind vLLM and 2.36× behind SGLang (previously 2.63× and 2.49×).
This optimization is useful, but does not close the remaining compiler/kernel
schedule gaps.

Raw serving directories are `lunaflux-baselines-20260906-r1-lunaflux-vector-old`
and `lunaflux-baselines-20260906-r1-lunaflux-vector-new-run2` under `/dev/shm`.
New-server logs and its startup-only unsuccessful measurement attempt remain
in the separate `...-vector-new` directory and are excluded. The first offline
materialization attempt used the wrong working directory and failed; r2
corrected it without overwriting r1. Neither startup issue is a kernel failure.

Both benchmark servers drained and closed their children with exit status zero.
All GPU experiments are serialized. Production deployments are untouched.

Combined raw archive (including both CUDA sources, the exact graph timing
harness, comparison probe, MoonBit runners, sanitizer output, SASS, serving
results and logs): `/private/tmp/lunaflux-vector-transfer-20260906-r2.tar.gz`.
Its SHA-256 matches the remote archive:
`e47308b27ad916a99edbe16477f8b2b9d73f3d9a90625f4f929c1ea5f74d6fa1`.
The earlier r1 archive is incomplete because the timing-harness path was
local-only; it is not the result archive.
