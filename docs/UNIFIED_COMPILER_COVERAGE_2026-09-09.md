# Unified projection compiler: coverage expansion

## Architecture and completion criterion

One semantic program owns `Dot`, selected rows, sibling products, activation
rounding, and stores. Pure scheduling transforms partition independent maps
and ordered folds. Device lowering owns matrix instructions, lane movement,
operand transport, and physical storage. Different schedules are legitimate;
parallel legacy generators and silent precision fallbacks are not.

Removal is complete only when the shared framework covers the corresponding
shape, row demand, numeric order, and target capability. Reducing supported
inputs or merely renaming an old renderer does not satisfy this criterion.

## First increment (`c3052dc`)

- Deleted the separate cooperative QKV and dense CUDA generators and their
  source-selection branches. Their tests now exercise the common compiler
  route while retaining single-token and launch-ABI assertions.
- Removed the artificial K >= 512 and K % 64 pipeline restrictions. Every
  complete K16 matrix reduction can select the same ordered pipeline. A pure
  transfer-group function chooses K64, K32, or K16 without padding global
  weights or changing reduction order. Arbitrary non-K16 scalar reductions
  are not reclassified as matrix operations.
- Removed the selected-row pipeline's separate WMMA fragment implementation.
  Selected rows now share the explicit operand permutation, matrix fragments,
  and output permutation for both narrow and wide transfer groups.
- Added regressions for QKV, dense, and selected-row head with K =
  16, 32, 48, 64, 128, 256, 384, 512, 1024 and output width 80. These explicitly
  exercise partial output tiles rather than only large aligned matrices.

## Validation

The native projection AOT tests pass 35/35; the functional projection compiler
tests pass 36/36. The warning-denied native check and generated-interface
check pass. No public interface changes were needed.

Physical target: RTX 5060 Ti, CUDA 13.1, sm120. The isolated campaign is
`/run/user/1000/lunaflux-unified-fold-20260909-r2`; downloaded results are in
`/private/tmp/lunaflux-unified-fold-results-20260909-r2`.

- All 27 generated shape/family combinations compile and match an independent
  ordered-F32 CPU referee at token counts 1, 2, 7, 17, 33: 135 cases.
- The numerical fixture uses exactly representable dyadic BF16 inputs. It
  checks every output, including untouched sentinel regions and noncontiguous
  selected-row scatter. This is not arbitrary-data numerical equivalence.
- K48 QKV, dense, and head each pass memcheck, racecheck, initcheck, and
  synccheck across those five token counts: 12 sanitizer invocations.
- The existing large-shape paired graph probe checks QKV, output, gate/up,
  and down at 1, 7, 17, 63, 64, 65, 255, 256, 257, 504, 1024 tokens, with
  three paired trials and bitwise agreement throughout.

Representative graph replay medians, microseconds; baseline is `2bca035`:

| Kernel | Tokens | Baseline | This increment |
|---|---:|---:|---:|
| QKV | 7 | 15.36 | 15.32 |
| QKV | 1024 | 387.10 | 387.13 |
| Output | 7 | 17.29 | 17.29 |
| Output | 1024 | 187.14 | 187.14 |
| Gate/up | 1024 | 500.23 | 500.16 |
| Down | 1024 | 174.62 | 174.63 |

Large-shape speed is unchanged within the paired trial variation: this
increment expands compiler coverage and removes duplication, not a claimed
end-to-end acceleration. Gate/up and down are unchanged controls. No fresh
bank-conflict-counter or full-model serving result is claimed here. The first
campaign directory retains a probe compilation failure caused by using an
obsolete CUDA context API; r2 uses the primary-context API and releases it.

## Completion increment (`f65963c`): remaining projection generators removed

- Deleted the independent selected-row strip and resident source generators.
  Both now use the same masked matrix map and immutable transport layout;
  strip and whole-row residency remain selectable storage strategies.
- Consolidated scalar QKV/dense/head and single-row subgroup arithmetic into
  one product-of-dot-fold lowerer. Ordered accumulation and explicitly allowed
  strided trees remain distinct numerical schedules, not silent fallbacks.
- Consolidated scalar MLP into one sibling product with optional F32
  materialization. Matrix MLP keeps its BF16 materialization contract.
- Removed the old non-pipelined matrix MLP implementation and the remaining
  WMMA sibling fallback. Small admitted MLP matrix extents (256) use the same
  staged product folds as large extents. Groups 1/2/4/8/16 retain coverage.
- Removed the permanently-false public `reuse_input_tile` flag. Schedule v6/v7
  records actual scalar versus masked-matrix row tails. Reuse and lifetime
  decisions live in the existing immutable plans.

Physical campaign: `/run/user/1000/lunaflux-unified-fold-20260909-r5`,
RTX 5060 Ti, CUDA 13.1. This campaign checks generated kernels, not serving:

Native validation passes: `moon info`, `moon fmt`, warning-denied `moon check`,
and the full suite **3653/3653**. Loopback listener tests require the normal
unsandboxed test environment. The projection package has 39 tests; the pure
projection compiler has 36. The final regeneration matches the GPU-tested
sources byte-for-byte (the additional inline scalar MLP was tested separately).

Downloaded archive: `/private/tmp/lunaflux-unified-fold-20260909-r5.tar.gz`,
SHA-256 `0d51f335fda06c5feb83e0674370670878ed0cf27482c8e8a0b0b8161a30f409`.

- Repeated all 135 short-K QKV/dense/head cases.
- 60 independent MLP cases: input K256/K512, intermediate/output256,
  groups1/2/4/8/16, tokens1/2/7/17/33/65. Both BF16 intermediate and final
  output match the independent CPU calculation exactly on the dyadic fixture.
- 20 head cases: strip/resident, groups1/8, tokens1/2/7/17/33, K1024/output80.
  Sparse selected rows and untouched output sentinels match exactly.
- 20 scalar cases: QKV/dense/MLP/head plus non-materialized MLP at
  tokens1/2/3/4 match CPU results,
  including scalar head's existing packed selected-row output ABI.
- 20 sanitizer invocations passed: the previous 12 plus all four tools for
  the small group1 MLP and group8 resident head. Zero errors/hazards; these
  checks are not bank-conflict-counter measurements.
- 44 large-shape paired cases, three trials each, remain bitwise identical.

Graph replay medians in microseconds versus the same `2bca035` baseline:

| Kernel | Tokens | Baseline | Unified |
|---|---:|---:|---:|
| QKV | 7 | 15.37 | 15.38 |
| QKV | 1024 | 387.04 | 387.10 |
| Output | 7 | 17.30 | 17.30 |
| Output | 1024 | 187.14 | 187.16 |
| Gate/up | 1024 | 499.83 | 499.86 |
| Down | 1024 | 174.64 | 175.97 |

This removes parallel projection implementations without removing declared
capabilities. It is not a speedup claim: down is approximately 0.8% slower in
this paired run. No fresh end-to-end or all-shape bank-conflict-zero claim is
made. Historical immutable fixtures remain historical tests, not runtime
fallbacks. Different physical schedules and scalar/matrix numeric contracts
are intentional, not duplicate legacy implementations.

The scope is the previously listed projection migration debt, not a claim
that the entire repository, arbitrary-DAG compiler, or every backend is free
of technical debt. No model-name condition is introduced.
