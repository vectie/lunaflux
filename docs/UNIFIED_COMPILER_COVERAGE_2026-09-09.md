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

## Implemented in this increment

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

## Remaining migration, not completed work

The complete projection cleanup is still open. Selected-row strip and
whole-row-resident generators, scalar/non-matrix lowering, duplicated
single-token lowering, and non-pipelined MLP implementations still need
consolidation. Their replacement must preserve declared storage and numerical
semantics rather than silently discard an offline strategy. The owner is this
compiler-unification workstream; its completion boundary requires removing
these parallel implementations after replacement coverage and GPU checks.

There is no model-name condition in this increment. It does not claim that
every operation, shape, or backend has finished migration.
