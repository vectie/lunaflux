# Independent QK dots improve long ordinary decode but not C16

Interleaving two independent key dots yields **3.02% median paired gain** in the
ordinary long-C2 decode probe. C16 and complete split/merge do not improve
reliably. Neither two-dot nor four-dot scheduling is promoted to production.
The remaining serving gap is not fixed by this experiment.

## Schedule change without arithmetic reassociation

The preceding [packed-load experiment](BENCHMARK_DECODE_PACKED_KEY_2026-10-07.md)
added unpacking instructions without reducing the number of QK loads. This
experiment retains the original BF16 loads and exposes independent arithmetic
instead of reading unused components.

Within one tile, each score owner originally completes one key dot before
advancing to another key. The two alternatives interleave two or four independent
key accumulators over the component loop. Key `group + owner + i * 8` retains its
original four-lane owner and ordered 32-component fold. Only operations belonging
to different dots move across one another. Each dot still uses separate F32
multiply and add, the same shuffle tree, scale and publication. Softmax, PV,
operand staging, barriers, launch geometry, shared reservation and merge are
unchanged. Ragged paths guard both loads and publication for inactive keys.

Both ordinary candidate 468 and split partial 3903 are transformed. Merge 3904
is unchanged. Controls remain KV32, D128, GQA2, owned8 blockwise F32 arithmetic,
block 64 and 33040 dynamic shared bytes. The split control uses eight partitions
with partition grain 32; chain timings include partial and merge.

The pure diagnostic transform checks both terminal entries, retains preceding
operand-staging branches and rejects incomplete modules. Tests enumerate all
tails 1 through 32, owners and lanes, and reconstruct the interleaved execution
order per key. No numerical tolerance, FMA, fast math or reassociation changes.
This remains offline diagnostic automation. Production integration would require
a pure independent-work schedule in physical IR and terminal CUDA lowering, not
source-string surgery or a model-specific branch.

## Complete workload results

All **60** alternating paired trials passed bitwise equality, the independent
scalar oracle and KV integrity checks. Each cell has five pairs; acceptance
requires at least 1% gain in every pair. Positive percentages mean shorter
completion time and are median paired gains.

| Useful rows / envelope / history | Ordinary two dots | Ordinary four dots | Split and merge two dots | Split and merge four dots |
| --- | ---: | ---: | ---: | ---: |
| 16 / 16 / 4095 | -0.79%, regression | +0.80%, inconclusive | -0.41%, regression | +0.18%, inconclusive |
| 2 / 8 / 32767 | **+3.02%, improved** | +2.67%, inconclusive | +1.52%, inconclusive | -1.08%, regression |
| 1 / 1 / 127 | +0.22%, inconclusive | +0.05%, inconclusive | 0.00%, inconclusive | +0.40%, inconclusive |

The two-dot ordinary long-C2 cell has minimum paired gain +1.12%. Independent
median durations are 1417.126 → 1371.071 microseconds; their ratio is not the
paired acceptance metric. Four-dot ordinary long C2 has minimum gain only +0.35%,
so its favorable median does not qualify. Two-dot split long C2 includes a -2.02%
pair. Short split two-dot includes a -24.72% outlier; it is retained, not trimmed
or turned into a win.

The split long control is already faster than the ordinary long winner in these
samples. The latter is therefore not grounds to replace the existing split route
or claim an end-to-end improvement.

## Paired counters explain the limited scope

Nsight profiles the actual ordinary control and two-dot CUBINs with matched
inputs. C16 uses grid 16 x 8; long C2 has two useful rows in envelope eight and
grid 8 x 8. These replay durations are secondary diagnostics, not unprofiled
serving measurements. Replay order and cache/clock behavior can affect magnitude;
the five alternating unprofiled pairs own the acceptance decision.

| Counter | C16 control | C16 two dots | Long C2 control | Long C2 two dots |
| --- | ---: | ---: | ---: | ---: |
| Replay duration, microseconds | 1198.144 | 1197.280 | 1613.376 | 1453.568 |
| Warp instructions | 41,500,160 | 41,465,856 | 41,295,520 | 41,262,560 |
| Registers per thread | 148 | 156 | 148 | 156 |
| Shared-memory resident-block limit | 2 | 2 | 2 | 2 |
| Eligible warps per scheduler cycle | 0.09 | 0.09 | 0.36 | 0.36 |
| Issue active | 8.99% | 8.83% | 36.25% | 36.38% |
| Short-scoreboard cycles per active issue | 3.78 | 4.38 | 0.68 | 0.66 |
| Long-scoreboard cycles per active issue | 2.96 | 2.50 | 0.18 | 0.48 |
| MIO-throttle cycles per active issue | 2.21 | 2.31 | 0.07 | 0.10 |
| Fixed-wait cycles per active issue | 0.64 | 0.47 | 0.65 | 0.47 |
| Barrier cycles per active issue | 0.43 | 0.43 | 0.05 | 0.09 |
| Source-correlated excessive shared wavefronts | 0 | 0 | 0 | 0 |

Stall ratios are cycles per active issue, not wall-time shares. They cannot be
summed or subtracted to allocate the completion-time gap.

The transformation reaches executed code: fixed-wait behavior changes and
register usage rises, but all 4,194,304 `LDS.U16` key reads and 1,048,576 `LDS.64`
value reads remain. Within each pair, FADD and FMUL counts are identical. Total
instructions fall only about 0.08%, mostly supporting work, not a reduction of
the mathematical algorithm.

For C16, fewer long-scoreboard cycles per active issue accompany more short
dependencies and unchanged eligible work. Replay is effectively unchanged and
unprofiled timing fails acceptance. Long C2 retains its eligible-work level and
shows less fixed-wait contribution, with a favorable unprofiled result. This
supports the narrow schedule choice, but does not prove that fixed wait alone
accounts for its 3.02% gain. The much larger replay delta is not a serving-speed
claim.

Compilation reports no spills or stack frames. The two-dot ordinary/partial
entries use 156/153 registers; the four-dot entries use 143/143. Fewer registers
in four-dot scheduling do not make it the winning schedule.

## Correctness and resource boundaries

All **44** boundary cases passed, both variants and ordinary/split chains, two
useful rows within envelope eight, histories 0, 6, 7, 8, 30, 31, 32, 63, 64, 66
and 4096. All **36** sanitizer runs passed, including memcheck with leak checks,
racecheck and synccheck at histories 0, 32 and 4096. Each also checks output,
oracle and KV integrity.

The diagnostic passed warning-denied check and **2/2** ownership/order and
transformation tests locally and on the target. Local test construction caught
non-interpolating raw template strings, single-occurrence replacement and
repeated staging delimiters before upload. Only the corrected artifact was
uploaded; no remote GPU failure was hidden. The archived strengthened test
snapshot has identical transform/build/run code to the executed script and adds
more precise per-key order assertions. The reusable finisher has an explicit
interleaved profile; it does not relabel the preceding packed-load non-wins.

GPU work was serialized on Spark 179, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, sm121, with pinned CUDA 13.0.88.
User units enforce 8 GiB memory, no swap and a 900-second limit. Counter
containers enforce 8 GiB combined memory/swap. The 32 GiB host reserve remains
required throughout. Serving processes and unrelated workloads were untouched.

## Decision and remaining work

Retain two-dot scheduling as a qualified **ordinary long-C2 diagnostic**, not a
global production default. Four-dot scheduling has no accepted cell. Together
with the packed-load and register-exchange results, this establishes that local
operand transport or reordering is insufficient to fix C16.

The next experiment must improve C16's actual ready work or reuse, while checking
the complete chain and keeping the numerical law explicit. Repeating wider
loads, replacing packed PV reads with scalar shuffles, or optimizing an ordinary
long cell that is not the serving route cannot close this gap. A fresh matched
whole-chain attribution should decide between changed CTA/work decomposition
and another consumer schedule before more compiler policy is added.

Last verified serving remains **225.600 output tokens/s** for 4096-input/64-output
C16. Historical matched vLLM/SGLang controls are 243.03/241.85; no baseline or
serving retest is implied here. The end-to-end parity gate is still open, and the
broader “fix all” request remains incomplete.

## Reproduction and evidence

Owned automation:

- `benchmarks/gpu_pipeline/decode_interleaved_qk_20261007.mbtx`
- `benchmarks/gpu_pipeline/decode_trial_summary_20261007.mbtx`
- `benchmarks/gpu_pipeline/finish_decode_packed_key_20261007.mbtx`, explicit `interleaved` profile
- `benchmarks/gpu_pipeline/decode_register_exchange_gates_20261007.mbtx`, variants `dots2 dots4`

Terminal root:
`/home/wlc004s/lunaflux-decode-interleaved-qk-20261007.IZcf946q`.
Paired counter roots:
`/home/wlc004s/lunaflux-interleaved-c16-counter-20261007.rqe25fTz` and
`/home/wlc004s/lunaflux-interleaved-long-counter-20261007.jYmJYkGl`.

Archive SHA-256:
`3778cd2587f0ebb7fbd8d386ae05ba625e4c45947c165056dff985e73f704a9b`.
Downloaded without overwrite to
`/tmp/lunaflux-interleaved-qk-evidence-20261007.GAflHTHX`;
local archive hash and all **428** manifest entries verify. Frozen control,
unchanged probe, variants, paired samples, boundaries, sanitizer logs, counter
reports/SASS, automation checks and terminal unit records are preserved.
