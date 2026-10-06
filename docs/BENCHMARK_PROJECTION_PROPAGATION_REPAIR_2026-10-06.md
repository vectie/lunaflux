# Projection-policy propagation regression and repair

## Scope

Qwen3-0.6B BF16 on Spark .179, GB10 sm121,
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. This is an offline AOT composition
and selection repair, not a new model-specific kernel, compiler rollback or
production deployment. Functional IR and runtime ownership are unchanged.

The October 5 long-context experiment inadvertently replaced the previously
tuned projection policy with defaults while rebuilding an 8192-query AOT
envelope. Long-context attention improved, but short-context serving regressed.
Context admission, query-artifact capacity and per-step scheduling budget are
different quantities; increasing the first does not require increasing all three.

## Regression reproduced

Serial old/current/current/old starts on the same GPU, one excluded warmup and
three measured trials per start/cell: six samples per side. Exact input token
generator, output count and client are reused. Timings are unprofiled serving
wall time; tok/s counts only generated output tokens.

| Input/output/concurrency | October 4 ms | Accepted October 5 ms | October 4 tok/s | October 5 tok/s |
| --- | ---: | ---: | ---: | ---: |
| 4096/64/C1 | 687.5 | 734.5 | 93.09 | 87.13 |
| 4096/64/C16 | 4517.5 | 5370.5 | 226.67 | 190.67 |
| 32512/64/C1 | 7378.5 | 4336.5 | 8.67 | 14.76 |

C16 completion time increased 18.9%; throughput fell 15.9%. At 32K the newer
attention route reduced completion time 41.2%. A blanket rollback would lose
that improvement.

Raw evidence: `/home/wlc004s/lunaflux-short-regression-20261006.5UrNGvun`.

## Actual executed-kernel diagnosis

Paired Nsight Systems captures used the same diagnostic parent and unchanged
arm-specific device workers. One warmup plus one measured C16 wave per arm.
These profile totals include both waves and must not be reported as serving
throughput or an exact additive decomposition of the unprofiled wall-time gap.

| Kernel family | October 4 selected geometry | October 5 selected geometry | Old total ms | New total ms |
| --- | --- | --- | ---: | ---: |
| Full QKV | grid 32×32, 116 registers | grid 128×32, 75 registers | 774.983 | 2053.158 |
| Output | grid 512, 80 registers | grid 1024, 76 registers | 319.954 | 392.191 |
| Gate/up | grid 1536, block 512, 58 registers | same grid/block, 64 registers | 909.013 | 1026.763 |
| Down | grid 256, block 128, 218 registers | grid 2048, block 128, 92 registers | 421.328 | 579.394 |
| Prefill attention | grid 63×16, block 128, 235 registers | same | 1126.306 | 1097.529 |
| Ordinary C16 decode | grid 16×8, block 64 | same | 3476.313 | 3466.702 |

Full QKV executes 1792 calls on both sides. Four times as many CTAs and an
extra 1278 ms across two waves identify a major regression. Prefill attention
and ordinary decode did not become slower in this capture. The initial suspicion
that missing attention-route coverage was the primary short-context regression
is therefore corrected, not retained as the diagnosis.

The generator `benchmarks/gpu_pipeline/ako_query_chunk_serving.mbtx` omitted
`--compiler-register-limit 255` and the measured `--projection-folds` inputs at
both candidate export and release bind. The default register envelope is 128;
the resulting ingress schedule used 16-row rather than 64-row tiles. This was a
failure to propagate earlier optimizations, not evidence that another IR layer
or a new attention algorithm was required.

Raw trace: `/home/wlc004s/lunaflux-short-trace-20261006.dgDnW3Wf`.

## Repair

1. The normal query-capacity helper now requires either matching frozen export
   and bind commands via `--projection-policy EXPORT_COMMAND_JSON BIND_COMMAND_JSON`,
   or an explicit `--unmeasured-projection`. Missing/duplicate policy fields and
   inconsistent export/bind options are rejected. Subsequent phases verify the
   policy receipt and command hashes. No global register-limit default is changed.
2. The serving repair reuses the exact qualified projection/reduction release
   and cubins. Only the new page-batched alternate attention is recompiled for
   the 2048-query envelope. The 32512-token input admission remains available;
   the per-step budget remains 2048.
3. The original complete fused-bundle arguments are transformed immutably.
   Only the alternate attention module, its geometry/symbol/query tile, output
   path and explicit approximate-exp2 law change. Regression tests check that
   unrelated selected arguments and the input array are preserved.
4. Attention routes are freshly calibrated for the repaired bundle scope.
   Prefill/mixed coverage includes rows 1/2/4 through 32K, rows 8 through 16K,
   and rows 16 through 8K; aggregate diagnostic KV stays within 131072 tokens.
   Old records are not relabeled as measurements of new artifacts.
5. Deployment admission remains strict. Canonical release and executable paths
   are passed instead of symlink aliases, and driver/subprocess logs have
   distinct names.

Repair root: `/home/wlc004s/lunaflux-propagation-repair-20261006.kKDWuXV4`.
Calibration passed; 247 probe/sanitizer output records are preserved. The first
rejected calibration used an 8K metadata spec with 2K artifacts and is retained
as a harness failure, not a valid performance measurement. Alias-rejection and
log-collision attempts are likewise preserved without weakening admission.

## Repaired serving verification

The paired validation campaign covers 128/256/C1, 4096/64/C1, 4096/64/C16,
32512/64/C1 and 32512/64/C2. Accepted October 5 is the control; repaired is the
candidate. ABBA, two starts and six measured samples per side/cell. All token
vectors and TTFT samples are retained, including differing output vectors; this
is not a model-quality or bitwise-generation-equivalence claim.

| Input/output/concurrency | Accepted ms | Repaired ms | Accepted tok/s | Repaired tok/s | Completion change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/256/C1 | 1707.5 | 1726.5 | 149.93 | 148.28 | +1.1% |
| 4096/64/C1 | 743.0 | 689.0 | 86.14 | 92.89 | −7.3% |
| 4096/64/C16 | 5399.0 | 4511.5 | 189.66 | 226.98 | −16.4% |
| 32512/64/C1 | 4351.0 | 3939.5 | 14.71 | 16.25 | −9.5% |
| 32512/64/C2 | 8389.5 | 7981.0 | 15.26 | 16.04 | −4.9% |

C16 throughput increases 19.7%. Its 4511.5 ms median essentially recovers the
October 4 4517.5 ms level; this is a regression repair, not a new 20% improvement
over the already optimized October 4 implementation. Long-context improvements
survive and improve further. The short 128-token control is 19 ms / 1.1% slower;
not every cell improved, and this is not dismissed as proven measurement noise.

Mean-per-request TTFT medians fall 172.5 → 120 ms at 4K C1, 1641.28 → 1161.72 ms
at 4K C16, 2875 → 2456 ms at 32K C1, and 4342.25 → 3716.5 ms at 32K C2.
The short control rises 15 → 16.5 ms.

| Input/output/concurrency | Repaired ms | Pinned vLLM ms | Pinned SGLang ms | Time gap vLLM/SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128/256/C1 | 1726.5 | 2129.0 | 2115.5 | −18.9% / −18.4% |
| 4096/64/C1 | 689.0 | 770.5 | 779.5 | −10.6% / −11.6% |
| 4096/64/C16 | 4511.5 | 4197.5 | 4231.5 | +7.5% / +6.6% |
| 32512/64/C1 | 3939.5 | 3948.5 | 3718.5 | −0.2% / +5.9% |
| 32512/64/C2 | 7981.0 | 7255.5 | 6756.5 | +10.0% / +18.1% |

Negative means lower completion time. Reference results are frozen same-day
measurements, not rerun simultaneously with this repair. The remaining C2 and
C16 gaps are not claimed to be resolved by this propagation fix.

[Local raw comparison JSON](/tmp/lunaflux-propagation-repair-20261006.RYOD0K3A/comparison.json)
retains all six samples, TTFT and full output vectors. Repaired repeats have one
output vector per cell; the accepted control has three C16 and two C2 vectors.
That observation is not independent numerical or model-quality qualification.

The fresh replay independently confirms the intended schedules are executed:

| Selected family | Accepted grid / registers | Repaired grid / registers | Accepted total ms | Repaired total ms |
| --- | --- | --- | ---: | ---: |
| Full QKV, 1792 calls each | 128×32 / 75 | 32×32 / 116 | 2051.978 | 776.338 |
| Output, 1766 calls each | 1024 / 76 | 512 / 80 | 394.237 | 319.006 |
| Gate/up, 1766 calls each | 1536 / 64 | 1536 / 58 | 1024.025 | 909.636 |
| Down | 2048 / 92 | 256 / 218 | 586.274 | 424.946 |
| Prefill attention, 1792 calls each | strict c322, 63×16 / 235 | exp2 c30322, 63×16 / 234 | 1102.576 | 1025.450 |
| Ordinary C16 decode, 3080 calls each | 16×8 / 126 | 16×8 / 148 | 3473.143 | 3481.882 |

Down totals include 1766 accepted and 1822 repaired calls because tiny bounds
select different companions. The table is a two-wave Nsight Systems replay,
not throughput and not a new hardware-stall-counter capture. Restored geometry,
resources and GPU time verify propagation beyond merely writing flags. QKV
recovers the October 4 selected schedule; ordinary decode is effectively unchanged.

All nine scoped sanitizer runs passed, separately from unprofiled timing:

| Gate | Query/rows/history workloads | Result |
| --- | --- | --- |
| memcheck with full leak checks | 128/1/0; 1792/1/30720; 2048/2/31744 | zero errors, zero leaked bytes |
| synccheck | same full-query workloads | zero errors |
| racecheck | 65/1/0; 65/1/32447; 129/2/32703 | zero hazards, errors or warnings |

Odd query tails exercise the same compiled ownership/barriers over 32K history
without instrumenting the full matrix. This is scoped coverage, not exhaustive
full-matrix racecheck. The initial large racecheck was interrupted to bound
instrumentation cost and is preserved as `validation-v2/INTERRUPTED.txt`, not a
pass. An earlier missing-cubin harness attempt is retained in `validation/`.

All probes also pass the independent numerical/KV checks. Maximum observed
pairwise difference is 0.000488281 and sampled FP64-oracle error 0.000454269,
within the existing 0.003 contract. The alternate's approximate-exp2 law remains
explicit; none of this is a claim of bitwise equivalence with strict arithmetic.
Successful gates: `validation-v3/RESULT.txt`. Peak validation memory was 1.0 GiB;
zero swap. No production release or source kernel default is promoted.

Campaign: `/home/wlc004s/lunaflux-propagation-abba-20261006.jqR6QJBw`.
Selected replay: `/home/wlc004s/lunaflux-propagation-trace-20261006.55URDmhl`.

Memory limits: serving 64 GiB, bridge 2 GiB, controller/probes 8 GiB, zero swap,
and at least 32 GiB system available-memory reserve checked every 500 ms.
GPU work is serialized. No new baseline-engine timing is mixed into this A/B;
comparisons use the explicitly pinned [same-day framework campaign](BENCHMARK_SELECTED_FRAMEWORKS_2026-10-06.md).

Minimum available memory across the four serving starts: 101618760 KiB
(96.91 GiB). All serving runs completed with empty runtime stderr and drained
before subsequent GPU work. Fused bundle SHA-256:
`54753db487c343443226c97937efb0571ef7176212843fac03d8495960b2b726`;
route table `7b3dda884f123c2eff8b8ab26304fe1bd3a1bf147e079abf003fb2a1c054dbb0`;
launch `a601ea713669ee2ed9fd6cd6ca69a08c3fce176be04cc4e2252de55403d03555`;
worker `dd4e8e2b5ddfb66cc6c901cf96b5ca14f12f9138841644c4fc2c4b4480928622`.

Source checkpoint: `5776d2e4`. Six focused automation tests passed, and
warning-denied native checks passed for the changed automation. This is not a
claim of full validation of the unrelated dirty source tree.

## Sealed evidence

The [downloaded archive](/tmp/lunaflux-propagation-repair-20261006.RYOD0K3A/verified-measurements.tar.gz)
contains the regression reproduction, original diagnosis trace, repaired ABBA
and selected replay, compiled attention, scope-bound calibration and all scoped
sanitizer results. Rejected/interrupted harness attempts are retained alongside
successful results, not relabeled or overwritten. The archive lists 5009 members;
its downloaded SHA-256 matches the remote sealed archive:

`af50bc042a46007ce2ccf7ce6bc3f7cc783b0d3c89f30b333e68f8b227828dca`.

Remote terminal receipt: `lunaflux-propagation-repair-20261006.kKDWuXV4/VERIFIED.txt`,
`outcome=completed`, benchmark-only and explicitly not production promotion.
