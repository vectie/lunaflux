# Scratch lifetime reuse and decode readiness pipelines

This follow-up implements the source-level diagnosis rather than adding another
IR layer or forcing a new kernel family to win. End-to-end and fresh pinned
reference measurements are recorded below when their terminal runs complete.

## Changes

1. **Projection geometry:** an immutable `SequentialScratch` plan distinguishes
   overlapping from disjoint operand-staging and epilogue lifetimes. Disjoint
   storage uses `max(producer, consumer)`, not their sum. CUDA lowering retires
   all operand readers before reusing the arena. Multiple column windows retain
   separate storage because the lifetimes genuinely overlap. This admits larger
   row tiles without weakening the shared-memory ceiling or changing arithmetic.
2. **Decode readiness:** two-stage blockwise candidates express K readiness,
   probability production, V readiness and reader retirement as effects. The
   lowering consumes that plan instead of synchronously staging the entire tile.
3. **History-axis parallelism:** blockwise decode now has an explicit partial
   map and F32 `(max, denominator, numerator)` merge ABI, including empty
   partitions and invalid metadata. It is independently exported and probed.
   This new numerical family is **not yet a production runtime split route**.
4. **Selection:** bounded offline measurements compare all legal finalists,
   including the existing schedules. Typed canonical exports are used for the
   runtime entry points; diagnostic symbols cannot be substituted into a serving
   bundle. Candidates absent under the resource cap are skipped, not fabricated.

The semantic and physical plans are immutable MoonBit values. GPU writes,
barriers and asynchronous readiness remain explicit lowering effects. No model
builder or scheduler imports CUDA-specific scheduling decisions. These changes
are general compiler mechanisms exercised with a Qwen3-0.6B workload; they do
not establish equal benefit across all models and devices.

Kernel source snapshot: `466ede5c`, archive SHA-256
`1882919d5c6757c39c929dde146df3fbd2fcc786be84be022bdb36759558089c`.
Calibration availability fix: `9d4ea9a3`. Canonical export and isolated campaign
packaging fixes: `8cfb7fc1`, `0fa441b0`, `0d5eee11`. Spark-only target substitutions
are sm121/CUDA 13.0.88; unrelated working-tree changes are excluded.

## Calibration

Spark GB10, 48 SMs, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
Five paired unprofiled samples per candidate/cell; runtime-bucket geometry.
Ten cells cover short/long prefill, 1/8/16 decode rows and 127/4095 history.
Resources and candidate source identities accompany the workload-scoped records.

| Operation / cell | Candidate | Median GPU time |
| --- | --- | ---: |
| Full ingress, query2048/rows8/history2048 | row32, K32, two stages | 508.277 µs |
| Same | row64, K32, two stages | 500.274 µs |
| Same | row16, head2, K16 | 1075.380 µs |
| Decode, query16/rows16/history4095 | existing c441 | 1262.948 µs |
| Same | blockwise sync c450 | 1783.399 µs |
| Same | blockwise async c452 | 1639.102 µs |

In the paired C1/history4095 probe, c452 improves blockwise c450 from
1005.899 to 724.478 µs (28.0%). It does not beat every existing decode schedule.
Its reported local allocation is 192 bytes/thread; register and local-memory
cost must not be hidden behind the asynchronous-copy label. Candidate c453 is
absent from the 48 KiB serving frontier, not a failed numerical experiment.

The selected serving package therefore uses row64/K32 ingress, async c322
prefill and c441 decode. It is not a forced-new-family experiment. Automatic
per-cell runtime dispatch is still absent: measurements across ten cells do not
imply ten live routes.

## Validation and remaining scope

The affected package tests pass 194/194. The exact packaged ingress, prefill
and decode cubins pass memcheck (including leaks), racecheck and synccheck.
These numerical probes are not a broad model-quality evaluation.

Failed isolated setup attempts are preserved separately: diagnostic decode
symbol substitution, missing probe source, then missing geometry header. None
reached model serving, and none is relabeled as a performance result.

Primary run directory:
`/home/wlc004s/lunaflux-scratch-pipeline-v2-20260930.3g24CVOM`.
Selected package: `selection-v2`; serving: `serving-v4`; completed calibration:
`calibration-final`. GPU workloads are serialized. LunaFlux runs have a 64 GiB
unit ceiling, no swap growth and a 32 GiB available-memory reserve. Pinned
reference containers have an 80 GiB ceiling and no swap growth, plus the same
reserve monitor; their limits are not claimed to be identical to LunaFlux's.

Fresh serving, reference, partition and counter results: pending terminal runs.
No performance parity or production promotion is claimed at this stage.
