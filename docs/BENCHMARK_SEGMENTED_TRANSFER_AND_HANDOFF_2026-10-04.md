# Segmented projection transfers and host handoff

Root-agent implementation and measurements on the DGX Spark GB10, CUDA
13.0.88, sm121, GPU `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
No subagent work was used to finish this round. No production deployment was
changed. This extends the full-step attribution report, not a claim that the
remaining framework gap has disappeared.

## Compiler change and propagation

The generic physical projection pipeline silently replaced any segmented
sibling transfer execution with `SerialTransfer`. The CUDA segmented renderer
also staged vectors through temporary register values. An explicit async
schedule therefore could not reach that selected implementation.

The correction is an immutable effect refinement, not a new model special case:

1. Retain the requested transfer lifetime independently of segmented storage.
2. Refine only sibling effects; preserve numerical order, operand ownership,
   fragments, epilogue, intermediate/down schedule and launch dimensions.
3. Lower pending segmented transfers to direct global-to-shared `cp.async`,
   retaining tail masks, zero fill, commit/wait and CTA publication.
4. Carry `execution:serial|prefetch|overlapped` through source-bound fold-record
   v4 selection, primary AOT entries and bounded row variants.

V1–V3 records and unspecified CUDA defaults retain their previous behavior.
Explicit measured choice is a startup/AOT concern, not runtime JIT or token-step
validation. The same mechanism applies to eligible gated projections, not only
Qwen. CUDA instructions remain in CUDA lowering.

The isolated serving exporter and normal release binder consumed the same real
v4 observations, target and register ceiling. Source **and recipe** identities
matched for all 283 operations. Only the 28 MLP modules changed; other candidate
modules and the selected fused attention/ingress modules were retained. The
full MLP's down schedule and bounded launch descriptors were unchanged.

Two preparation failures were preserved: the first cloned an outdated small-row
test assertion; the second compiled selected sources but asked the release binder
to regenerate defaults. The final preparation uses the actual compiler selector
and normal binder, rather than weakening identity checks or inventing timings.

## Exact installed MLP chain

Five alternating event-time trials, 20 launches per trial. Identical installed
serial source versus async effect refinement, 2048 active tokens, 32-row profile:

| Component | Serial median, µs | Async median, µs | Reduction |
| --- | ---: | ---: | ---: |
| Gate/up | 519.333 | 442.989 | 14.7% |
| Down | 226.706 | 226.478 | approximately unchanged |
| Complete MLP chain | 773.456 | 702.477 | 9.2% |

Primary geometry remains 1536 CTAs × 512 threads, static shared memory 24576
bytes plus 16384 dynamic bytes. Registers decrease from 64 to 58. Down remains
256 CTAs × 128 threads, 218 registers, 49152 shared bytes. Full output/workspace
comparisons are bitwise equal at 32, 129, 512 and 2048 tokens. Memcheck,
racecheck and synccheck pass on the 129-token tail case.

The selector's separate default-versus-selected campaign measured 869.155 →
719.619 µs at 2048 tokens. That comparison also includes the previously selected
down/bounded schedules; it must not be reported as the isolated async gain above.
Its real medians, five samples and exact source identities supplied the v4 MLP
record. Previously measured dense projection records were retained unchanged.

## Hardware counters explain the primary gain

A separate serialized NCU capture of the same primary geometry/transport
alternative recorded the following. This diagnostic fixture had a different
down companion; its primary counters are not a serving-wide measurement.

| Metric | Serial | Async |
| --- | ---: | ---: |
| Warp instructions | 97,554,432 | 93,020,160 |
| Registers/thread | 64 | 58 |
| Achieved occupancy | 54.9% | 66.1% |
| Average warp latency, cycles | 27.644 | 21.405 |
| Long-scoreboard contribution per issue-active | 7.358 | 4.339 |
| Barrier contribution per issue-active | 9.154 | 5.413 |
| Tensor activity, elapsed % | 33.54% | 35.80% |
| Cold counter-replay duration, µs | 643.168 | 603.232 |

The direct transfer eliminates register staging, reduces instructions by 4.65%
and lowers load/barrier latency contributions. These contributions are not
additive percentages of wall time. Cold NCU replay is not warm event time or
end-to-end throughput. Barriers and memory dependencies remain; they did not
become zero.

## Host handoff attribution and bounded-copy fix

The selected comparison endpoint is native-framed, not the OpenAI pool. The
new diagnostic capture distinguishes blocking IPC read completion from frame
validation. Across 287 adjacent measured transitions:

| Signed monotonic span | Aggregate, ms |
| --- | ---: |
| Child retirement | 0.345 |
| Completion publication → parent read | 4.436 |
| Parent completion validation | 1.383 |
| Parent publication/scheduling | 96.457 |
| Plan encoding → write start | 35.420 |
| Parent write progress | 4.767 |
| Parent write completion → child read completion | 100.283 |
| Child plan validation/copy | 39.682 |
| Completion-writer setup | 1.011 |
| Child executor preflight | 0.351 |
| Total | 284.138 |

These are instrumented transition spans, not separate CPU profiles or an
uninstrumented throughput result. Concurrent boundaries stay signed and
telescope exactly. The terminal measured graph-step count selects the final
marker window; this assumes no later workload after the measured trial/drain.
The IPC span cannot be attributed entirely to CPU validation or scheduling.

Validated plan/completion ownership copies now use bounded native bulk moves
instead of scalar byte loops. Source/destination bounds, structural validation,
epochs, deterministic ownership and failure behavior are retained; there is no
new allocation or token-path authentication. Unaligned envelope, sentinel,
roundtrip, invalid-offset and failed-load epoch regressions pass. This change
does not claim to remove the entire 39.682 ms validation/copy span, the 100.283 ms
IPC span or the 96.457 ms publication/scheduling span.

Profiler-only marker logging and descriptor relocation are confined to disposable
diagnostic builds; they are not in the serving binary.

## Software and physical scope

Local warning-denied native check and full suite pass: **4361/4361**. Affected
packages pass **287/287** locally and **285/285** in the narrower cloned Linux
source. Existing toolchain migration exemptions are
`-79-20-29-25-92-14`; these are not unsuppressed warning-denied claims.

Standalone GPU campaigns used 8 GiB limits; servers use 64 GiB and no swap.
Every physical campaign requires an idle GPU and at least 32 GiB available
memory. Frameworks run sequentially. The serving comparison combines the MLP
and host-copy changes; it cannot causally assign the entire speedup to either.

## Fresh end-to-end comparison

Three unprofiled fresh starts per framework, counterbalanced framework order,
one warm-up per cell. BF16 Qwen3-0.6B, C16; generated-token throughput is the
total requested output tokens divided by batch completion time.

| Input/output tokens | LunaFlux, ms / tok/s | vLLM, ms / tok/s | SGLang, ms / tok/s | Luna completion gap, V / S |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 367 / 1395.1 | 327 / 1565.7 | 329 / 1556.2 | 12.2% / 11.6% |
| 4096/64 | 4564 / 224.4 | 4197 / 244.0 | 4203 / 243.6 | 8.7% / 8.6% |
| 4096/256 | 12734 / 321.7 | 11833 / 346.2 | 11972 / 342.1 | 7.6% / 6.4% |

Luna's 4096/256 trials were 12727, 12734 and 12782 ms; 4096/64 trials were
4564, 4559 and 4570 ms. Relative to the earlier same-day Luna medians,
completion time decreased by 0.66% and 1.53%, respectively. These small changes
are not the isolated 9.2% MLP-chain gain. A later unchanged-runtime control is
reported separately below; the old and new campaigns were not interleaved.

The fresh unchanged-runtime medians were 368, 4633 and 12821 ms, respectively.
Against this control, new completion time decreases by 0.27%, 1.49% and 0.68%.
All three long-cell new trials were below their corresponding fresh-control
trial ranges: 4559–4570 versus 4608–4658 ms; 12727–12782 versus
12796–12863 ms. This supports a small observed benefit but is only three starts,
not a randomized/interleaved statistical proof of a sub-percent gain. The
short-cell improvement is within observed timing variation.

Input token IDs and output lengths were checked for every request by its
request-row filename, avoiding completion-order ambiguity. Both 4096-input
cells match generated token IDs exactly across prior Luna, fresh-control Luna,
new Luna, vLLM, SGLang and all three rounds. The 128-input cell does **not**: rows 0/12 can
diverge at generated token 2 (382 versus 624), then continue on different
sequences. Prior Luna and vLLM also vary across their own fresh starts. This
shows the difference is not unique to the new async path, but does not prove
its numerical cause. Short-cell timing is fixed-input/fixed-length, not an
exact-output correctness comparison. No mismatch was suppressed or relabelled
as a successful exact-output check.

The MLP transport change reaches the actual serving package and has a positive
isolated chain measurement. It does not eliminate attention's dominant cost,
the remaining projection work or IPC/scheduling spans. The earlier full-step
attribution remains the explanation for why a 9.2% gain in one chain produces
only a small end-to-end change; this round did not capture a new full-serving
hardware trace, so it cannot assign that small delta to individual components.

## Artifacts

- Exact installed-chain qualification:
  `/home/wlc004s/lunaflux-root-segmented-aot-20261004.Ru35wdzu`.
- Default-versus-selected selector observations:
  `/home/wlc004s/lunaflux-root-default-async-20261004.yBnLDMjs`.
- Primary counter capture:
  `/home/wlc004s/lunaflux-root-segmented-20261004.MYh6LZXU`.
- Refined diagnostic host trace:
  `/home/wlc004s/lunaflux-root-handoff-v5-20261004.xKyUM9sN`.
- Isolated selector/binder/materialization:
  `/home/wlc004s/lunaflux-root-transfer-serving-v3-20261004.l0311nDO`.
- Unprofiled three-round framework comparison:
  `/home/wlc004s/lunaflux-root-transfer-bench-20261004.bymPisbf`.
- Fresh three-start unchanged-runtime control:
  `/home/wlc004s/lunaflux-root-transfer-control-20261004.BiwLbk6Y`.

The compact archive is downloaded to
`/private/tmp/lunaflux-root-transfer-download-20261004.Npgv3GQl/gap-repairs.tar.gz`.
Its local SHA-256 matches the remote archive:
`4047f2b2bc36e003a3d0e1be67ce3c9563703de4b4c8a38bdeccb103f1aaa3cd`.
It contains 10,764 files, including qualification logs, sources/counters,
serving request records, the failed preparations and fresh controls. This is a
compact diagnostic snapshot, not a deployment bundle: build/cache/toolchain,
source-clone and deployment/release exclusions are enumerated in `EXCLUDED.txt`.
The original remote roots remain intact. Final per-request comparison and
timing details are also downloaded beside the archive as `summary-final.json`.

Implementation commits: `3eb059a5` (explicit segmented transfer effects and AOT
selection), `38026615` (validated bulk frame copies). No production deployment
or baseline container configuration was changed. A kernel gain alone is not a
serving gain.
