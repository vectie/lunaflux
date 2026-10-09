# Compiler selection value: controlled ablation

This is an offline diagnostic, not a production selector or a new kernel.
It tests the 2026-10-09 claim that the execution-frontier redesign makes better
decisions than its predecessor given the same alternatives and observations.

## Predeclared experiment

- Old: `24ef8d5f`, immediately before continuation-aware frontiers.
- New: `b8d23b2f`, current committed implementation before this experiment.
- Compile both **unaltered** `compiler/fusion_regions` packages with the same
  MoonBit toolchain and the same diagnostic caller. No selector backport or
  outcome-dependent ranking rule is allowed.
- Hold kernels constant: the already-qualified Q64/KV128 and Q64/KV64 paged
  BF16 reference-prefill cubins. This is a single-operation complete region,
  not evidence of whole-model/producer-consumer search.
- Give both selectors identical five-pair calibration observations. Freeze
  selections before reading independent five-pair validation observations.
  Both receive the same 15 repetitions per pair and candidate. Validation
  reverses candidate timing order and workload order.
- Eight `(rows, queries/row, total KV length/row)` cases: `(8,64,512)`,
  `(8,256,8192)`, `(1,2048,30720)`, `(1,7,19)`, `(4,128,2048)`,
  `(16,128,2048)`, `(2,512,16384)`, `(1,1024,65536)`.
- Keep numerical permission fixed: BF16 probabilities/base-2 softmax, sampled
  independent scalar error and pairwise absolute error at most 0.003, repeat
  determinism, read-only operands and inactive-output checks. Not bitwise
  equivalence or generation-quality parity.
- Use the compacted runtime Q64 grid and the unchanged existing paired probe.
  The adapter only translates its timing fields to `ako_trial.mbtx`'s schema;
  original output is retained. No new hardware-counter hypothesis is needed
  to compare selectors that dispatch identical cubins.
- GPU: Spark .179 GB10, serialize workloads, 8-GiB process cap, no extra swap,
  32-GiB available-memory reserve, 20-minute external deadline.
- Report every case, winner disagreement, validation regret against the
  measured best candidate, and fixed-KV128 control. Small differences are
  noise-sensitive; do not call selecting a calibration minimum a held-out win.
- Also test the *actual* reference-prefill construction path without an
  override: does it consume measurements, or keep its fixed default?

The scope is the latest execution-economics upgrade, not every historical
compiler refactor. One GPU cannot establish cross-hardware generality. A
generic selector experiment cannot establish that serving invokes it for this
kernel family. Report those boundaries separately. Existing measured decode
route propagation is useful functionality but not automatically an advantage
over an old selector supplied the same complete choices.

## Files

- `selector.mbt`: shared diagnostic caller of the real old/new package API.
- `setup.mbtx`: extracts exact package snapshots and builds/tests both.
- `reference_pair.mbtx`: lossless probe-output adapter for the existing AKO
  paired harness. It never creates timing values or chooses a winner.
- `measure.mbtx`: fixed remote calibration/validation campaign.
- `analyze.mbtx`: freezes old/new choices from calibration only, then evaluates
  independent validation and resource-envelope queries.
- `source_control.mbtx`: builds the real old/new source probes, compares their
  default output, and compares the old manual schedule backport with the new
  explicit schedule. `backport.patch` changes four lines in two isolated old
  source files; it is a diagnostic control, not production code or automatic
  optimization. Apply with `git apply --unidiff-zero` only inside the extracted
  old snapshot, after recording its original default output.
- `seal_remote.mbtx`: preserves raw records, initial non-primary attempt and
  hashes in a non-overwriting archive; excludes build/dependency caches.
- `verify.mbtx`: checks frozen decision agreement, validation regret, sampled
  memory reserve, and byte identity of both versioned selector packages.

Temporary extracted packages and experiment output are retained outside the
working tree; no second compiler implementation enters production.
