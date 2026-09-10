# Shared operand-copy fixes — 2026-09-10

## Outcome

The 55 affected cases in the September 9 retest now have zero source-attributed
shared excessive wavefronts (copy and ordinary load/store), combining the r2
results with the final r3 replacement of its two small-group down failures.
Four additional K=512 small-group cases also report zero. This is **not** a
claim that aggregate hardware replay counters are zero or every kernel is
covered.

The original 75-report inventory and counter definitions are in
[SHARED_CONFLICT_RETEST_2026-09-09.md](SHARED_CONFLICT_RETEST_2026-09-09.md).

## Implementation

Implementation commit: `0e1e1a6`.

- Sibling gate/up uses the same row-contiguous swizzled operand map as the
  other matrix pipelines. Producer vectors follow logical row order rather
  than gathering global rows in compact shared-tile order.
- One private `projection_operand_transfer_source` lowers all four pipeline
  copy sites. Invalid rows become explicit zero vector stores, rather than
  zero-byte asynchronous global-copy requests.
- Narrow transfer rows use register-materialized vector reads/stores; wider
  rows retain asynchronous copies. Small one/two-group intermediate folds
  also use register materialization after their predicated bypass-copy SASS
  retained excess in physical measurements.
- The obsolete compact microtile implementation was removed. Its consumer
  coverage was replaced by tests against the shared authoritative transport
  map, including matrix-load transactions and every partial-row tail.

These are CUDA-backend transport decisions over immutable tile geometry and
workgroup distribution. Model semantics, ordered arithmetic, scheduler policy,
public API, buffer capacity, and launch ABI are unchanged. No Qwen-name branch,
runtime profiler, filesystem check, or new token-step allocation was added.
The MoonBit agent/refactoring skills guided the shared lowering and regression
tests; diagnostic orchestration uses `.mbtx`.

## Counter progression

| Case | Before copy excess | Final copy excess | Final other excess |
| --- | ---: | ---: | ---: |
| QKV T=7 | 358400 | 0 | 0 |
| Output T=7 | 179200 | 0 | 0 |
| Gate/up T=1024 | 33030144 | 0 | 0 |
| Head, 2 selected rows | 3722432 | 0 | 0 |
| Small MLP down, group 1 | 4928 | 0 | 0 |
| Small MLP down, group 2 | 1792 | 0 | 0 |

The full affected matrix covers QKV/output/gate/up/down at
`[7,17,63,64,65,255,256,257,504,1024]`, default head at selected rows `[2,8,32]`,
streamed eight-group masked head at `[2,32]`, and small MLP gate/down for
groups `[1,2,4,8,16]`. Single-token shared-free cases are not counted as
positive shared-layout coverage.

An intermediate row-contiguous asynchronous gate/up implementation reduced
T=1024 excess to 2,359,296 and latency to about 346 microseconds, but did not
meet the requested zero-source-excess goal. The final register-materialized
narrow transport trades some of that speed for zero source excess.

## Paired timing, T=1024

Same GPU, captured repeated isolated launches, three interleaved trials;
numbers below are medians in microseconds. These are operation measurements,
not end-to-end serving or baseline-framework comparisons.

| Kernel | Previous | Final | Change in latency |
| --- | ---: | ---: | ---: |
| QKV | 387.13 | 407.19 | +5.2% |
| Output | 187.13 | 191.33 | +2.2% |
| Gate/up | 500.14 | 389.90 | -22.0% |
| Down | 175.52 | 187.04 | +6.6% |

All paired outputs were bitwise equal across the eleven-token vector including
T=1. Regressions are retained explicitly under the user's zero-conflict-first
instruction, not labelled performance wins. r3 changes only four small-group
MLP sources; these four main-kernel timing sources are byte-identical to r2.

## Validation and remaining boundary

- Native full suite at r2: 3,653/3,653; final r3 projection package: 39/39.
- Final warning-denied native check, interface generation and scoped format.
- Independent GPU numerical probes across nine reduction widths for QKV,
  dense and head; small MLP group coverage; paired main-kernel bit equality.
- 60 completed sanitizer invocations across r2/r3: memcheck, racecheck,
  initcheck and synccheck, all exit zero.
- No deployment or production service was changed.

Aggregate hardware counters are still nonzero. For example final T=1024
gate/up reports load 57,286 and store 484,410 despite zero source-attributed
excess. Those totals include replay behavior beyond the isolated address-bank
metric; this run does not prove their entire cause or eliminate them. Their
remaining cost needs separate replay/instruction analysis, not an assertion
that all physical conflicts disappeared.

Fresh norm/sampling/decode sources were exported locally, but uploading their
supplemental diagnostic driver was rejected by automatic approval. Those
additional counter runs have **not** run. Fused ingress also remains outside
this retest. Thus all known source-copy failures in the affected matrix are
fixed, but the broader all-kernel/all-hardware-counter objective is unfinished.

## Artifacts

Remote directories beneath `/run/user/1000/`:

- `lunaflux-copy-fix-20260909-r1`: initial copy separation/layout experiment.
- `lunaflux-copy-fix-20260909-r2`: 55-case counter matrix, numerical probes,
  paired timings, and 44 sanitizer runs.
- `lunaflux-copy-fix-20260910-r3`: final small-group replacement, eight counters,
  independent numerics and 16 sanitizer runs.

The generated source comparison shows only `mlp-{256,512}-g{1,2}.cu` changed
between r2 and r3; all other source files are byte-identical. Raw NCU reports,
sources, CUBINs, stdout/stderr and diagnostic drivers are preserved, not
overwritten. Supplemental driver awaiting approval:
`/private/tmp/lunaflux-copy-coverage-run.mbtx`.

Downloaded archive: `/private/tmp/lunaflux-copy-fix-20260910-final.tar.gz`.
SHA-256: `fbf369a605b38aeba4e3492141fcd5b4cb56bfb48602421d79407dccd20349e8`.

## Later current-runtime follow-up

After renewed explicit approval, the previously blocked supplemental work was
completed in a new, bounded current-runtime matrix: 178 cases / 356 valid
samples, zero source excess, with shared-free cases separated from positive
shared-layout coverage. Fresh Qwen comparison also found a small-batch
performance regression; source-counter zero is not performance parity. See
[the current-runtime report](SOURCE_COUNTERS_AND_QWEN_COMPARISON_2026-09-10.md)
for the full vectors, corrected classifier history, timing table, and archive.
The earlier campaign records and their original limitations remain unchanged.
