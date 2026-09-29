# Four remaining compiler paths

This implementation continues the [selected-schedule repair](BENCHMARK_SELECTED_SCHEDULES_2026-09-30.md).
The last measured serving result is still 190.48 output tokens/s for
4096/64 C16 on Spark. It is **not** a measurement of the changes below.

## Ownership and design

Four parallel workstreams use the existing compiler, not a second optimizer:

| Workstream | Pure planning responsibility | Terminal realization |
| --- | --- | --- |
| Multi-head/rotary reuse | Complete-head producer packing and invariant value lifetime | Packed projection staging and cross-head rotary reuse |
| Blockwise decode | Explicit blockwise F32 softmax law, schedule and effect ordering | Shared-KV block fold with bounded tails |
| Fragment-copy reduction | Single-use fragment forwarding and operand ownership | Adjacent matrix load/consume instructions without intermediate aggregate copies |
| Exact-workload tuning | Exact workload and compatible hardware-class scope, optional measurements | Offline per-workload AOT selection with runtime-equivalent launch geometry |

Model builders continue to describe semantics. Schedules and lifetimes are
immutable compiler values. CUDA instructions stay in terminal lowering.
There is no request-path compilation, measurement, allocation or artifact scan.

Head packing must reach serving: `heads_per_cta` is carried from the compiler
recipe through the bundle exporter into startup admission. Admission derives
the head grid from total heads and this explicit ownership, never from an
ambiguous quotient of a capped grid. Token-row ownership remains independent.
Legacy bundles retain one head per CTA. The new v7 sidecar binds ownership,
optional bucket routes, and query metadata in one canonical record.

Blockwise decode is not the old ordered per-key FP32 recurrence. It carries a
distinct numeric identity, symbol and runtime ABI. Startup rejects a symbol/law
mismatch and does not infer split-K entry points from the blockwise module.
Its different reduction order requires numerical validation; merely retaining
F32 storage does not establish bitwise equivalence.

Measurements refine selection; they must not require calibration of every
physical GPU. Device identity is measurement provenance, while reuse eligibility
is a compatible hardware/configuration class plus compiler, numeric law, source
and exact workload. Absence of compatible observations retains an explicit
unmeasured legal selection. Per-cell kernel wins are not whole-serving wins.

## Validation boundary

Each workstream needs pure-plan and actual-source regression tests. Integration
tests cover sidecar round trips, incompatible ownership, numerical-law mismatch
and graph-bucket launch propagation. Hardware checks are serialized and preserve
at least 32 GiB available memory; bounded kernel probes do not load a model.

## Implemented paths

- `AttentionIngressPacking` describes complete-head ownership, masked Q/K/V
  boundaries, and prepare/consume/retire lifetime for row-invariant rotary
  values. The CUDA backend exposes one-, two-, and four-head realizations with
  resource-filtered accumulator windows. Packing does not imply profitability:
  a window that cannot retain the packed columns can reload input tiles.
- `BlockwiseFold` describes tile-local F32 probabilities and ordered tile-state
  merges. Decode candidates 450/451 use 32/64-key tiles, with their own
  `blockwise-f32-probability-v1` law, suffixed symbol, and production ABI.
- `FragmentForwarding` validates single-use operand renaming. Its attention
  binding keeps an RHS matrix load and both consumers in one PTX region,
  eliminating the intermediate C++ aggregate without changing the numeric law
  or synchronization. SASS register allocation still needs measurement.
- V2 calibration records preserve ten independent workload cells, actual and
  captured geometry, source/toolchain identity, performance compatibility class,
  and UUID provenance. The real exporter consumes workload-scoped observations.
  A named cell supplies the selected AOT package; automatic cross-cell runtime
  dispatch is not introduced here. See
  [calibration usage](../benchmarks/gpu_pipeline/SELECTED_POLICY_CALIBRATION.md).

The default remains a legal unmeasured fallback when no compatible timing
record exists. New candidates are reachable, but this change does not assert
that the widest packing or blockwise decode always wins.

## Integration checks

The combined affected native suite passed **615/615** tests, using the existing
dependency migration warning exemptions `-79-20-29-25`. Pure physical/attention
IR tests also passed without warning exemptions. Public interfaces and scoped
formatting were regenerated/checked. The runtime builder script type-checks.

A repository-wide warning-denied check additionally encountered two unrelated
working-tree warnings: `fragile_catch_all` in
`engine/joint_diffusion_execution/pipeline.mbt:91`, and `alert_internal` in
`runtime/remote_tls/channel.mbt:174`. Those other workstreams were left intact;
this is not a claim of a clean full-repository check.

No new end-to-end serving performance or production-readiness claim is made.

## Spark physical validation

Implementation commit: `7b6f0468`. Tests ran sequentially on GB10 `sm_121`,
UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, CUDA 13.0.88. The harness
required an idle GPU and at least 32 GiB `MemAvailable` before each bounded
subprocess. No model was loaded; the GPU was idle after completion.

| Path | Numerical coverage | Result |
| --- | --- | --- |
| Packed ingress | D64 h1/h2/h4 with Q6/K3 segment crossings; D128 h1/h2 with masked head tails; 16 token sizes from 1 through 129, reversed pages and distinct weights | All five pass numerical, memcheck, racecheck, synccheck |
| Decode | Existing c430/c440/c441 plus new c450/c451; key-tile boundaries, mixed rows, invalid page, history through 4096 | All five pass independent FP64-oracle checks, unchanged KV, and all three sanitizers |
| Prefill forwarding | c322/c2000/c2001/c2003/c2004; ragged rows, query tails, repeated stage reuse | All five pass numerical and all three sanitizers |
| Extended prefill | Q16/64/128, histories 512/1024/2048/4096, plus short/ragged cases | All five pass; maximum absolute reference error 0.00893354 for c322, 0.0114849 for c2000–2004 |
| Exact-workload probe | CUDA build on Spark and independent CPU launch-geometry regression | Pass; full ten-cell calibration not run |

A separate packed-ingress checksum pass produced **80 records, 32 matching
dimension/token groups, zero mismatches** between h1/h2/h4 where available.
This checks output and complete KV bytes in addition to the independent oracle;
it does not establish bitwise equality outside the tested inputs.

Remote result directories under `/home/wlc004s/`:

- `lunaflux-four-paths.rto0GQsL` — packed ingress sanitizers.
- `lunaflux-four-decode.FxpIJgjm` — decode oracle and sanitizers.
- `lunaflux-four-prefill.VNlLKcBC` — bounded prefill oracle and sanitizers.
- `lunaflux-head-checksums.mcqpsgAF` — packed checksums and calibration probe build.
- `lunaflux-four-prefill-full.w8zga3c6` — extended-history prefill oracle.

Sources, binaries and results were archived without replacing prior runs and
downloaded to
`/tmp/lunaflux-four-paths-export.EqEj6NFI/lunaflux-four-paths-7b6f0468-validation.tar.gz`.
Remote/local SHA-256:
`0567313f7ed1c2f505e63566f6a5a4483d644ff6a48652cc0777a0ae005f53cc`.

These checks do not measure whole-serving gains, prove lower SASS copy counts,
or pick the fastest packed-head/blockwise candidate. Fresh exact-workload
calibration and matched serving comparisons remain the performance boundary.
