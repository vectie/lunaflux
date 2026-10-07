# Typed prefill performance with common startup routes

Replacing the prefill module while holding startup routes constant reduces
32K completion time by about 0.8–1.0%. It does not improve C16 completion time,
and 4K/C1 regresses. This is not a complete performance fix or a production
selection change. The earlier recalibrated comparison also changed route
coverage and choices, so its gain was not an isolated kernel effect.

## Controlled configuration

Spark .179, GB10 sm121, pinned Qwen3-0.6B BF16. Both configurations retain
170 common measured buckets and identical startup choices. Each scope keeps
its genuine measurements; old times are not relabeled as new measurements.
Seven buckets use the baseline on both sides because the old alternative does
not clear the new scope's minimum improvement margin. The 91 additional mixed
buckets from the recalibrated configuration are excluded from this experiment.
These common routes are a diagnostic control, not a claim of fastest routing.

The startup selector requires ceil(1% of baseline latency) improvement and
retains the baseline on ties. Reanalysis of the prior experiment finds 15
changed startup policies, versus 21 changed calibration minima. Neither is
alone proof of executed dispatch. The updated reporter distinguishes them and
rejects a purported common-route report with differing startup choices.

Only module 5 differs. Baseline SHA-256 is
`616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`;
typed-read SHA-256 is
`301f8fd9db1e0221e4b0b5048d08c03d55ee49f5f50d8ea04e5df33e4582ebf3`.
The eight other module identities and worker bytes match. Compiler default
selection, production services and numerical tolerances remain unchanged.

## Timing and an external read exclusion

Four old/new/new/old arms ran with two fresh starts each. Every start has one
warmup and three measured waves for each of five cells. Request bodies and
output lengths match. All eight starts drain and close with zero exit and
empty runtime stderr; raw results remain preserved.

An accidental CPU/filesystem read of the binary `qwen.numeric` artifact
overlapped `s2/r1-luna-full`. It was stopped, not interpreted as model policy.
The conservative recorded interval is 04:00:00–04:00:52 UTC. The preceding
start completed at 03:59:46 UTC. The affected start and its matched baseline
`s3/r1-luna-full` are excluded together, irrespective of their measured speed.
The primary table therefore uses a balanced **six-start subset**, three starts
and nine measured waves per cell per side. Its first ABBA half contains two
starts per side; its second contains one. The separate all-eight-start report
is retained but is not a clean causal timing estimate.

| Input / output / concurrency | Baseline tok/s | Typed read tok/s | Baseline ms | Typed read ms | Completion reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 256 / C1 | 151.30 | 150.68 | 1692 | 1699 | −0.41% |
| 4096 / 64 / C1 | 94.26 | 93.16 | 679 | 687 | −1.18% |
| 4096 / 64 / C16 | 225.60 | 225.60 | 4539 | 4539 | 0.00% |
| 32512 / 64 / C1 | 16.33 | 16.46 | 3918 | 3888 | +0.77% |
| 32512 / 64 / C2 | 16.97 | 17.14 | 7543 | 7467 | +1.01% |

Times are unprofiled complete-wave wall medians. Output tok/s includes prefill
and completion; it is aggregate for concurrent cells, not input plus output
throughput. The two ABBA-half reductions are −0.651%/+0.059% for short/C1,
−0.737%/−0.291% for 4K/C1, −0.110%/+0.395% for C16,
+0.716%/+1.042% for 32K/C1, and +0.619%/+1.698% for 32K/C2.
These small descriptive gains do not establish a universal or optimal route.

| Cell | Median TTFT baseline → typed ms | Median mean TPOT baseline → typed ms |
| --- | ---: | ---: |
| 128/256/C1 | 14 → 15 | 6.490 → 6.529 |
| 4096/64/C1 | 118 → 118 | 8.556 → 8.667 |
| 4096/64/C16 | 1211.5 → 1207 | 48.833 → 48.833 |
| 32512/64/C1 | 2449 → 2418 | 22.937 → 22.984 |
| 32512/64/C2 | 3848.5 → 3799.5 | 55.167 → 54.881 |

vLLM and SGLang are not rerun here. Against their historical October 6
243.03/241.85 tok/s at C16, this controlled candidate takes about 7.7%/7.2%
longer; against 17.58/18.85 tok/s at 32K/C2, about 2.6%/10.0% longer.
Those are historical comparisons, not a new matched three-framework campaign.

## Output stability and remaining work

With common routes, only 1 of 192 candidate C16 output vectors differs from
the first baseline vector, by one token. Within-side comparisons find 3 of
176 baseline vectors and 1 of 176 candidate vectors different, each by one
token. Every other cell remains identical within and across sides. Counts
include warmup vectors, not just timed waves.

The prior recalibrated comparison had 109 of 256 differing candidate C16
vectors. Most differences disappear under this control, which narrows the
investigation to schedule/route-dependent arithmetic rather than establishing
that typed reads introduced a large numerical fault. Remaining differences
still block whole-model bitwise parity. They do not alone prove a race or a
violation of the declared approximate numerical contract. Exact same-schedule
activation/logit comparison and independent model-quality checks remain open.

The previous separate serving timeline puts prefill attention at roughly
11.5% of C16 kernel time, versus about 48% for 32K/C2. A few-percent prefill
kernel improvement therefore cannot eliminate the whole C16 deficit. Those
shares come from the prior independent capture, not an additive explanation
of this table. The next optimization must target the actual selected decode
and projection chains as well as prefill dependency latency. Repeating only
prefill fragment rewrites would miss most C16 work.

Functional layering is unchanged: pure numeric/schedule/ownership plans,
explicit effects and lifetimes, then CUDA realization. This experiment fixes
measurement and selection attribution; it does not add a new compiler layer
or change production arithmetic.

## Selected execution and verification

A separate Nsight Systems start covers 4K/C16 and 32K/C2, each with a warmup
and one measured wave. It executes
`lunaflux_attention_prefill_tile_compiler_exp2_v1`, grid 63 × 16, block 128,
236 registers per thread: the typed-read artifact's resource identity. Across
both cells and warmups it records 3584 such launches. Module 5 is the bundle's
only provider of that symbol. This establishes propagation into actual serving
execution, not a kernel count inferred from calibration alone.

Both ordinary blockwise decode and split/merge paths also execute. The trace
does not turn per-symbol totals including warmup into unprofiled throughput,
and it is not a new Nsight Compute instruction-counter capture. Its supervisor
drains and closes with exit zero; the terminal GPU is idle.

The first trace attempt fails because the generic trace directory reused an
already-loaded user-systemd bridge unit name. No usable GPU capture results.
The retry gives the trace owner a unique evidence-directory suffix; a regression
checks independent names. The failed attempt remains in `selected-trace`.

All updated offline helpers pass native warning-denied checks. Their ten tests
pass: four common-route/controller tests, five report tests and one archiver
test. No production kernel changes are made in this follow-up; the prior
compiler phase's 4427-test result is not represented as a new dirty-tree test.

Workers retain 64 GiB process limits, a separate 2 GiB bridge envelope, no
process swap, and a 32 GiB MemAvailable reserve. Evidence is sealed remotely
without overwriting. The archiver now excludes `deployment.base` as well as
active materializations, using one shared exclusion definition for discovery
and validation; original deployments and failed attempts remain untouched.

Remote root:
`/home/wlc004s/lunaflux-typed-common-routes-20261007.OpHcoecq`.
Archive SHA-256:
`7b63c55d7b00e3b416a1f88b6608fc9a1a73537d496c57acf8aab686aae98b28`.
The balanced and all-start comparisons, authentic route tables, runtime
identities, interference record, failed trace and successful trace are retained.
The archive is downloaded without overwrite to
`/tmp/lunaflux-typed-common-verified-20261007.GzlQ8J/evidence.tar.gz`;
its local hash matches, and all **2699** extracted `FILES.sha256` members
verify locally.
