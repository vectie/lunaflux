# Selected prefill PV lifetime experiment

## Result

Neither joined PV emission nor two-slot PV operand lookahead passes the
declared performance gate. Both contain regressions. Production source,
selection, numerical law and serving artifacts remain unchanged. This round
does not fix the remaining vLLM/SGLang completion-time gap.

The fresh matched counter capture shows that lookahead adds supporting
instructions without reducing mathematical work or publication counts.
This rejects operand lookahead alone as the next production fix. It does not
prove that every possible lifetime schedule is slower.

One offline tooling bug is fixed: the operand experiment finalizer previously
required every cell to be inconclusive. It now preserves improved, regressed
and mixed outcomes instead of failing to seal them. A kernel win would still
be marked serving-unverified, not promoted automatically.

## Frozen workload and alternatives

Spark .179, GB10 sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Selected baseline:
`lunaflux_attention_prefill_tile_compiler_exp2_v1`, candidate 30322,
Q64/K64, single-stage split-copy lifetime, `approx-base2-f32-v1` numerical law.
The immutable synthetic operands use the real runtime bucket geometry, not
model activations. Q2048/R2/H28672 executes bucket rows 32 with grid 63×16
and 128 threads per CTA.

The two alternatives change only PV operand lifetime and terminal emission:

- Joined: eight output-column products in one PTX region, one operand slot.
- Lookahead: the same products with two live operand slots and rolling reads.

Both retain ascending product order, exact published XOR addresses, transposed
BF16 fragment mapping and accumulator semantics. QK, softmax, output rescaling,
shared layout and publications are not changed. The pure schedule expresses
`Read(column, slot)` and `Product(column, slot)`; ownership tests cover two
through eight columns. CUDA is only the terminal renderer. This is an offline
prototype, not another model-specific production path or runtime JIT.

Compilation is pinned to CUDA 13.0.88, sm121, O3, FMA contraction off,
FTZ off, precise division/square root and a 255-register ceiling. Both
alternatives use 242 registers/thread versus 234 for the baseline, zero local
bytes and zero stack. The matched capture still permits two resident CTAs/SM.

## Paired measurements

Five alternating pairs per cell; each member averages 30 CUDA-event
repetitions. Positive paired gain means faster. The admission rule was at
least 3% gain in every pair in every cell. Median paired gain is not the ratio
of latency medians. No failed pair is dropped.

| Alternative | Q/R/H | Baseline median µs | Alternative median µs | Median paired gain | Worst pair | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Joined | 1792/1/30720 | 6298.182 | 6225.113 | +0.79% | −2.64% | Inconclusive |
| Joined | 2048/2/28672 | 6594.695 | 6654.110 | −1.40% | −1.86% | Regression |
| Joined | 2048/16/2048 | 912.336 | 937.579 | −2.64% | −3.20% | Regression |
| Lookahead | 1792/1/30720 | 6293.772 | 6287.309 | +0.10% | −2.79% | Inconclusive |
| Lookahead | 2048/2/28672 | 6625.870 | 6684.115 | −0.71% | −2.80% | Regression |
| Lookahead | 2048/16/2048 | 925.456 | 942.229 | −0.60% | −3.62% | Regression |

All 30 pairs have complete-output bitwise agreement with the baseline,
maximum absolute difference zero, and sampled FP64 oracle error within 0.003.
Both alternatives pass memcheck, racecheck and synccheck at Q129/R2/H128:
zero errors/hazards, empty stderr. These six runs do not establish full-model
or long-context sanitizer qualification.

GPU workloads are serialized. Each timing/profile process has an 8 GiB memory
limit, no swap and a 900-second timeout, with a 32 GiB MemAvailable reserve.
Minimum recorded timing MemAvailable is 121,473,712 KiB. Terminal GPU is idle.

## Fresh matched counters

Nsight Compute 2025.3.1 captures baseline and lookahead separately on the
same Q2048/R2/H28672 workload. These are new captures of both binaries, not a
new alternative compared with old baseline samples. Profiler elapsed times
are not used as paired performance measurements.

| Counter | Baseline | Lookahead |
| --- | ---: | ---: |
| Total executed warp instructions | 891,530,112 | 900,062,080 |
| Dynamic HMMA | 119,668,736 | 119,668,736 |
| Dynamic transposed LDSM | 29,917,184 | 29,917,184 |
| Dynamic NOP | 25,242,624 | 31,787,008 |
| Dynamic IMAD | 14,876,672 | 19,551,232 |
| Dynamic IMAD.SHL | 15,681,536 | 13,870,080 |
| Dynamic WARPSYNC | 3,739,648 | 3,739,648 |
| Dynamic CTA barrier | 2,808,832 | 2,808,832 |
| Tensor activity, % of peak sustained elapsed | 64.99% | 63.64% |
| Active warps, % of peak sustained active | 16.04% | 16.01% |
| Average warp latency per issued instruction | 6.15 | 6.11 |
| Long-scoreboard wait per active issue | 0.72 | 0.73 |
| Short-scoreboard wait per active issue | 0.45 | 0.52 |
| Barrier wait per active issue | 0.17 | 0.30 |
| Fixed wait per active issue | 2.19 | 2.19 |

Total instructions increase by 8,531,968 (0.96%); dynamic NOPs increase by
6,544,384 (25.9%). Unchanged barrier counts do not imply unchanged barrier
waiting: arrival timing can differ. The capture records zero source-correlated
excessive shared wavefronts. Local-sector metrics are not collected in these
source reports; the compiled resource record shows no local allocation.

Not-issued fixed-wait samples shift from HMMA 106,322 / NOP 42,043 to HMMA
92,307 / NOP 53,086. Retained instruction windows show LDSM, fragment MOVs,
HMMA and NOPs. Samples name waiting consumers, not proven producer edges;
they cannot be added into elapsed-time attribution. This capture also does
not isolate QK from PV at every hot PC. The PV-only source perturbation can
change allocation and scheduling elsewhere in the compiled kernel.

The supported conclusion is narrower than “more buffering solves load
latency”: this exact buffering implementation changes supporting code and
waiting distribution without a repeatable win. A next schedule must expose
independent scalar/fragment work without making the joined PTX region overly
restrictive, then measure the resulting SASS and complete kernel. Simply
raising occupancy, buffer count or IR layer count is not justified.

## Remaining work and current serving result

The [October 6 serving report](BENCHMARK_DECODE_ROUTE_AND_FAIRNESS_FIX_2026-10-06.md)
remains the latest end-to-end result: 4096/64/C16 is 225.13 tok/s versus
243.03/241.85; 32512/64/C2 is 16.95 versus 17.58/18.85. Those reference
measurements are earlier same-session runs, not new interleaved comparisons
for these prototypes. No serving benchmark is repeated for rejected variants.

The [batched activation investigation](BENCHMARK_BATCH_ACTIVATION_MAPPING_2026-10-07.md)
also remains open: exact first-changed V input/weight replay is needed before
calling the solo/paired token difference a contract violation. This PV test
neither diagnoses nor repairs that separate numerical question.

The experiment budget of two PV alternatives is exhausted. Neither is
selected. Follow-up compiler work should retain semantic/numeric contract →
pure schedule and ownership → effects/lifetime → CUDA lowering, with an
explicit live-register and instruction-scheduling budget. No performance
regression is installed to make an isolated counter look better.

## Reproduction and sealed evidence

`benchmarks/gpu_pipeline/prefill_value_pipeline_20261007.mbtx` accepts
`FROZEN_OPERAND_ROOT PROBE_ROOT NEW_ROOT prepare|qualify|profile|finish`.
The existing AKO trial executes the prepared contracts. Profile requires the
administrator's existing profiler permission; the controller itself runs as
the ordinary user. Running it as root creates a different MoonBit dependency
cache and is not required. Failed administration/cache attempts are retained;
no existing evidence or build cache was deleted or broadly re-permissioned.

`summarize_value_pipeline_20261007.mbtx` extracts aligned CSV metrics and keeps
the raw SASS summaries. The finalizer preserves regressions. Five helper tests
pass: two PV ownership/address tests, two operand/finalizer tests and one CSV
alignment test. All three helpers pass warning-denied native checks.

Baseline source SHA-256:
`3738eb47bb25c18c6b9cb621d01725f45979fd5cb72b1b5ccbaa9be2f5aebd78`.
Baseline CUBIN SHA-256:
`616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`.
Probe SHA-256:
`78901fc5e3222e4394b3fb35746d60ad36a0181f22472c0397c5dee1e5079536`.
Candidate identities, 30 paired samples, six sanitizer outputs, two NCU
reports, raw CSV, instruction windows and executed controller revisions are
sealed under `/home/wlc004s/lunaflux-prefill-value-20261007.GiRKnpIO`.

Archive SHA-256:
`97fa71a00b452925149068d4b90cad69779f9c80d1cd50abc1536a84eebb7c02`.
The non-overwriting local download is
`/tmp/lunaflux-pv-final-20261007.DcPviFZj/lunaflux-prefill-value-20261007.GiRKnpIO.verified.tar.gz`;
its hash and all 226 extracted `FILES.sha256` entries verify locally.
Build/dependency caches are excluded; original failed
attempts, kernels, counters and commands remain preserved.
