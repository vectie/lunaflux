# Decode calibration, row-domain propagation and mixed-chain repair — 2026-10-06

## Scope and decisions

This is a bounded AKO follow-up to the propagation-repair and remaining-gap
diagnoses. It changes startup selection and allocation-free scheduler planning,
not model-specific arithmetic, request-path compilation or production deployment.
The model is Qwen3-0.6B BF16; the selected strict c468 decode CUBIN is unchanged.

GB10 on Spark .179: GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6,
sm121, 48 SMs, CUDA 13.0.88. Servers are bounded at 64 GiB with no swap;
controllers/probes at 8 GiB. Serving monitoring preserves a 32 GiB
MemAvailable reserve. GPU workloads are serialized.

The finite experiment budget was twelve ordinary-route probes, six full split
chain probes, three full mixed-chain probes, and selected end-to-end/trace
verification. Slower or inconclusive arithmetic candidates were not admitted.

## Bugs fixed

1. Decode calibration used compact power-of-two geometry instead of serving's
   1/8/16/32 captured row domains. The probe now mirrors serving buckets.
2. Calibration stopped at 8K although the admitted arena supported bounded 32K.
   All context buckets through the bounded limit are measured: rows 1/2/4 to
   32K, rows 8 to 16K and rows 16 to 8K, retaining the aggregate KV limit.
3. Matrix ordinary and partition helper composition collided through C++ ADL.
   The terminal renderer alpha-renames partition helper identifiers without
   changing arithmetic, exported ABI symbols or pure IR semantics.
4. Aged prefill consumed the entire token budget before runnable decodes could
   join. Pure budget planning now reserves eligible decode tokens, excluding
   in-flight/cancelled rows. This is a latency/throughput tradeoff, not a free
   throughput win.
5. Fresh serving traces exposed a second propagation bug: newly inserted split
   partial/merge steps always received FixedBucketLaunch. A selected eight-row
   graph still launched 32-row attention grids. Prepared companions now retain
   their own admitted AOT row-domain geometry; the pure expansion remapper
   preserves that proof into capture. Unknown ABIs remain fixed. Regression
   tests cover both stages, every bound, unrelated operations, duplicate,
   absent and overriding mappings.
6. Mixed calibration timed only the prefill kernel, excluding the decode
   companion. The offline probe now supports complete ordinary mixed chains;
   calibration times ordinary and split decode companions with independent
   history. Old prefill-only mixed records are not reused as full-chain
   measurements.
7. Mandatory measured mixed captures were constructed after optional variants
   could exhaust their structural budget. Startup now reserves the exact
   mixed-chain node count per unique row/query domain first, without increasing
   the 64-graph/32768-node allowance or changing deployment byte ceilings.
8. Candidate 7 was measured as wide-prefill + split decode + merge but serving
   prepared it from the baseline prefill list. Preparation now carries the
   wide steps, wide launch rules, wide owner and admitted decode row bounds
   together. Context buckets share a graph only when row/query domains match.

The complete mixed-route export initially failed startup before readiness.
The parent diagnosis reported inherited-drain binding after its child failed;
that message alone did not identify the underlying compiler/route defect.
The row-domain-only runtime started, isolating the failure to mixed-route
construction. Reserving mandatory captures and propagating the measured wide
chain restored readiness. The failed startup and diagnostic captures remain
separate from successful serving evidence.

No extra IR layer is needed for these defects: the existing immutable launch
plan was being lost at preparation, and the offline objective measured an
incomplete executable chain. The fix is propagation and objective completeness.
All identity validation remains at startup. There is no token-step profiling,
cryptography, filesystem scan or heap allocation added.

## Isolated measurements and numerical boundaries

Five alternating-order pairs per probe, synthetic output/KV correctness and
FP64 oracle checks; not model-quality equivalence.

| Workload | Ordinary strict | Whole strict split | Change |
| --- | ---: | ---: | ---: |
| C2 / history 32767, serving bucket 8 | 1460.03 us | 1189.72 us | -18.5% |
| C16 / history 4095 | 1148.61 us | 1188.05 us | +3.4%; retain ordinary |

Nine boundary checks (memcheck/racecheck/synccheck, rows 1/2/16) passed in the
extended calibration. KV remained unchanged, maxabs against the strict
baseline was zero in the quoted strict cases.

FMA ordinary gains were small/inconsistent, below the 3% all-pairs criterion.
Matrix ordinary was substantially slower at short/C1/C2 workloads. KV64
candidates exceeding the admitted shared-memory budget were rejected. Apparent
long-history alternative wins versus ordinary were mostly split parallelism:
they did not prove improvement over already-selected strict split. These
alternatives remain diagnostic, not the serving selection.

The complete mixed-chain comparison uses the same wide prefill module on both
sides, 2048 total queries, one prefill and one decode request, decode history
32511. It compares prefill + ordinary decode against prefill + split + merge.

| Prefill history | Baseline median | Candidate median | Worst paired gain |
| --- | ---: | ---: | ---: |
| 0 | 1646.41 us | 923.65 us | 43.9% |
| 14335 | 4898.02 us | 4099.28 us | 15.2% |
| 30720 | 8696.65 us | 7924.33 us | 5.35% |

All three passed correctness. The bounded route record uses the worst-history
median for each complete chain, not a prefill-only score. Unmeasured mixed
records are removed rather than silently trusted. Full mixed-chain memcheck at
2048 queries passed with zero errors. The full 2048-query racecheck exceeded
its 900-second unit limit (exit 143, 932.8 MiB peak, zero swap); this is a
preserved timeout, not a sanitizer pass. The bounded retry retains 30720/32511
prefill/decode histories but reduces the query domain to 128 for race and
synchronization checking. This is narrower coverage, not full-shape racecheck
success. Both bounded checks passed (zero race hazards/errors/warnings and zero
synchronization errors); their maxabs was 0.00012207 against the comparison
chain and 0.000355219 against the oracle, within the declared approximate law.
The full-shape check had zero maxabs against the comparison chain.

The first mixed-route export also exposed a calibration-record construction
error: a 2-versus-7 experiment omitted candidate 1, which the importer requires
as a real baseline. Export correctly rejected the incomplete table. The
publisher now separately measures candidate 1's complete chain and includes
all three identities without relabeling candidate 2. A regression covers this
baseline requirement. Failed exports and the bounded preparation-wait timeout
are preserved; no failed bundle was served. Final results follow below.

## End-to-end ablation before the last two repairs

Six unprofiled waves per engine/cell. Same explicit input-token vectors, greedy
64/256 output tokens, ignore EOS, prefix cache disabled. Different framework
prefill chunk policies are retained: vLLM 2048, SGLang 8192. These are full-engine
comparisons, not equal single-kernel geometry or an approved quality corpus.

| Input/output/concurrency | Repaired baseline LunaFlux | Route + fairness | Route only | Fresh vLLM | Fresh SGLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/256/C1 | 1726.5 ms | 1716.5 ms | 1716.5 ms | 2122.5 ms | 2131 ms |
| 4096/64/C1 | 689 ms | 685 ms | 687.5 ms | 771.5 ms | 781.5 ms |
| 4096/64/C16 | 4511.5 ms | 4550 ms | 4519 ms | 4213.5 ms | 4234 ms |
| 32512/64/C1 | 3939.5 ms | 3945.5 ms | 3953 ms | 3968 ms | 3724.5 ms |
| 32512/64/C2 | 7981 ms | 7901.5 ms | 7561.5 ms | 7280.5 ms | 6789.5 ms |

Route selection alone reduced 32K/C2 completion 5.26%. Adding fairness returned
about 340 ms of that saving while improving progress during prefill. This led
to the complete mixed-chain investigation, not another guessed kernel change.
The pre-projection-repair B5GyxQoX campaign is NOT the ablation baseline.

## Exact-wave trace attribution before the last two repairs

Profiler timings are diagnostic, not the unprofiled throughput table.
Kernel totals are not additive with overlapping CPU/API time.

| 32512/64/C2 traced time | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Prefill attention | 3544.04 ms | 3862.67 ms | 2863.06 ms |
| Decode attention + merge | 2474.01 ms | 1707.63 ms | 2077.70 ms |
| Other kernels | 1724.98 ms | 1561.92 ms | 1791.11 ms |
| No GPU activity in client window | 119.26 ms | 104.73 ms | 41.34 ms |

LunaFlux had 95 graph steps: 32 prefill/mixed, 63 decode. Split partial/merge
were 32-row grids despite the eight-row bucket. Sixteen mixed steps used
ordinary decode, totaling roughly 645 ms. All recorded selected calls used
graphs. This establishes a specific executable-chain gap, not a general claim
that host waits or bank conflicts explain the remaining difference.

SGLang's new trace uses its explicit CUDA_PROFILER start/stop range; the previous
startup-only trace is not used for this attribution. SGLang also displayed a
2158 ms first-request token gap while processing the second prompt: throughput
and inter-token fairness must be reported separately.

## Final verification

Final repaired serving benchmark and executed-symbol/geometry trace completed.
Two independent LunaFlux starts, three measured waves per cell after warm-up;
references are the already-completed six same-session waves above, not new
interleaved reference runs. Prefix caching is disabled; greedy output length is
fixed with EOS ignored. Numbers count output tokens only and include prefill.

| Input/output/concurrency | Final LunaFlux | vLLM | SGLang | Completion gap vs V / S |
| --- | ---: | ---: | ---: | ---: |
| 128/256/C1 | 151.43 tok/s, 1690.5 ms | 120.61 tok/s | 120.13 tok/s | -20.4% / -20.7% |
| 4096/64/C1 | 94.26 tok/s, 679 ms | 82.96 tok/s | 81.89 tok/s | -12.0% / -13.1% |
| 4096/64/C16 | 225.13 tok/s, 4548.5 ms | 243.03 tok/s | 241.85 tok/s | +8.0% / +7.4% |
| 32512/64/C1 | 16.28 tok/s, 3931.5 ms | 16.13 tok/s | 17.18 tok/s | -0.9% / +5.6% |
| 32512/64/C2 | 16.95 tok/s, 7553.5 ms | 17.58 tok/s | 18.85 tok/s | +3.7% / +11.3% |

Final 32K/C2 completion is 4.4% below the route+fairness intermediate and 5.4%
below the repaired pre-change baseline. The vLLM completion gap fell from 8.5%
to 3.7%; SGLang's from 16.4% to 11.3%. C16 did not materially improve. These
specific bugs are fixed; all performance gaps are **not** eliminated.

The final 32K/C2 Nsight Systems wave verifies split partial/merge grids of
eight rows in the ordinary captured decode bucket and one/two rows in measured
mixed buckets; no 32-row split grid remains. Sixteen mixed steps now execute
split partial+merge rather than ordinary decode. The strict c468 CUBIN remains
unchanged. The final wide-prefill plan is consumed with wide launch rules.

| 32K/C2 trace family | Before final repairs | Final |
| --- | ---: | ---: |
| Prefill attention | 3544.04 ms | 3619.35 ms |
| Decode attention + merge | 2474.01 ms | 2071.62 ms |
| Other kernels | 1724.98 ms | 1728.74 ms |
| No GPU activity | 119.26 ms | 131.38 ms |

Decode-chain time fell 16.3% (402 ms); prefill increased about 75 ms in this
single traced wave. Against the earlier exact reference traces, the remaining
gap to vLLM is approximately +364 ms decode and +167 ms other kernels, partly
offset by -243 ms prefill. Against SGLang it is mainly +756 ms prefill;
decode is approximately equal (-6 ms) and other kernels are lower (-62 ms).
These are cross-capture diagnostic differences, not confidence intervals or
an instruction-level explanation of the remaining kernel stalls. The final
trace is Nsight Systems, not a new Nsight Compute counter campaign.

Both unprofiled starts drain successfully with empty runtime stderr, child
exit zero and closed resources. Minimum observed MemAvailable for LunaFlux
was 104236240 KiB (~99.4 GiB), well above the 32 GiB reserve. No production
runtime or OCI was deployed.

Complete output vectors agree with both references for the two short C1
cases, but not all long/concurrent cases. A same-engine check against the
intermediate runtime found differences in 1/96 C16 vectors and 6/12 32K/C2
vectors; the other cases matched. The final runtime's measured vectors are
internally consistent. Approximate kernels and near-tie greedy selection can
change tokens, but this observation alone does not prove that explanation.
The numerical kernel probes pass their declared laws; this benchmark is not
an independent model-quality qualification or proof of bitwise model parity.

Focused native tests for the final affected packages: 369/369 passed (including
218 device-step tests) with existing toolchain migration warnings
disabled only at the command boundary (-79-20-29-25). Whole dirty-tree validation
has unrelated new-package warnings/errors; this is not a full-suite pass.
The existing large unrelated working-tree changes were not staged.
The whole-tree check was attempted and remains blocked by existing warning 92
in engine/joint_diffusion_execution/pipeline.mbt and warning 14 in
runtime/remote_tls/channel.mbt; neither unrelated dirty file was changed.

The exact prepared-owner GPU fixture passed ordinary eager/captured execution,
memcheck, racecheck and synccheck after its admitted dimensions were corrected
to its real one-block/32-thread ABI. A prior fixture incorrectly reused a
94x16/256-thread projection launch contract. The new domain propagation exposed
that mismatch as nondeterministic duplicate writers under memcheck (zero
memory errors, but failed output correctness). That failure is preserved and
is not counted as a sanitizer pass. The corrected fixture additionally tests
that an already-reserved mandatory capture succeeds after optional capacity is
exhausted. These fixture gates do not replace the model numerical probes.

## Remote evidence

- Main experiment: /home/wlc004s/lunaflux-decode-fix-20261006.2hRKeY3C
- Combined benchmark: /home/wlc004s/lunaflux-decode-fix-benchmark-canonical-20261006.ZQIeRVIv
- Route-only ablation: /home/wlc004s/lunaflux-route-only-benchmark-20261006.UxaZQFna
- Three-engine trace: /home/wlc004s/lunaflux-decode-fix-trace-20261006.tcP1UO1p
- Row-domain fix: /home/wlc004s/lunaflux-split-bucket-fix-20261006.yvZtqp0y
- Full mixed-chain repair: /home/wlc004s/lunaflux-mixed-decode-fix-20261006.ckcbaxXd
- Final capture/whole-chain propagation repair: /home/wlc004s/lunaflux-mixed-capture-repair-20261006.sQMel69Z
- Final benchmark: /home/wlc004s/lunaflux-mixed-final-benchmark-20261006.Xzq0NMMd
- Final selected trace: /home/wlc004s/lunaflux-mixed-final-trace-20261006.Iy8LiSeB

Failed canonical-locator attempts and rejected candidates are preserved.
Production was not deployed or promoted.

All listed campaigns, including failed startup and sanitizer attempts, are
sealed remotely with verified FILES.sha256 and read-only archives. The final
benchmark, both old/new GPU traces, same-session references, route-only and
row-only ablations, startup failure diagnostics, initial experiment and compact
final-source receipt were downloaded without overwrite and SHA-256 checked in:

`/tmp/lunaflux-decode-fix-20261006.xDxuHd9Q`

The final benchmark archive SHA-256 is
`3f8c46367f08cc55637fd79c303817f3ac1734201b5f03536196f35c82eed0ea`;
the final trace archive is
`513a70d12c143d72e6bbff8a518dc36b598194914980e2f22b67d6e57fb29397`.
The compact source receipt is
`11cb76db045ce81b19c0e643dca41379303a9d25d0bedc891e258f1af1604db9`.
Large mixed/row preparation archives remain sealed on the server because the
local volume had only about 0.5 GiB free. No user data or evidence was deleted.

Final runtime identities: parent `619140a64d70d77d...`, worker
`7da40e6f9f8c07f4...`, launch `e7e7a251f7474350...`, reusable bundle
`651b0d05ad38074b...`, route records `81983059d5bf6e9b...`.
Local and remote SHA-256 values match for all three final production planning
files. Source fixes are locally committed on `parallel` (including fca4153a,
94d04827, d88c5dea and db24da5d); no unrelated working-tree changes were staged
and the branch's unrelated accumulated commits were not pushed.
