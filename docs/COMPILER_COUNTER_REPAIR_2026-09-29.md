# Counter-driven compiler repair

This work implements the source findings in
[the three-kernel review](BENCHMARK_SPARK_THREE_KERNEL_SOURCE_REVIEW_2026-09-29.md).
The measured Spark baseline is `53deab59`, not a measurement of these changes.

## Architectural contract

Preserve the existing chain:

```
semantic operations and numerical law
  → pure legal schedule enumeration
  → resource eligibility and measured selection
  → immutable ownership/storage/effect programs
  → terminal device lowering
```

Do not introduce a parallel compiler, model-name heuristics, runtime JIT,
token-path validation, or a second numerical implementation hidden in a
renderer. CUDA shuffle, copy and matrix instructions belong only to lowering.
Changes to exponential approximation, probability precision or reduction order
must have a distinct numerical contract and independent correctness tests.

## Work items and acceptance

1. **Projection/epilogue ownership:** separate instruction tile, CTA tile and
   complete-head epilogue ownership. Keep producer-separated and fully fused
   partitions in whole-chain selection. Enlarge the realizable domain rather
   than assuming greater fusion is faster. Count operand rereads across live
   accumulator windows. Cache rotary values only with explicit lifetime and
   operand contracts.
2. **Prefill domain and selection:** include larger KV tiles in the same pure
   Cartesian domain, retain old candidate identities, check shared/register
   resources, and test terminal emission. An empty timing table is unmeasured,
   not an async win. Use exact device/toolchain/shape records; do not reuse an
   RTX timing identity on GB10. Keep strict arithmetic until a separately
   qualified numerical variant exists.
3. **Decode address realization:** consume the existing page-lookup hoist in
   grouped copy lowering. One address owner distributes the invariant page
   identity to its vector consumers. Preserve zero-fill, invalid-page handling
   and partial tiles; only issue subgroup collectives uniformly.
4. **Decode effects:** derive publication coalescing from the double-buffer
   lifetime. The next acquire can end the previous read epoch, but the final
   arena reuse still needs a publication. Test the effect sequence rather than
   deleting barriers in generated text.
5. **Further algorithm work:** matrix decode/blockwise softmax, fragment
   representation, affine current-KV views and measured split policy require
   their own typed transformations. Existing per-key FP32 and blockwise BF16
   laws are not interchangeable. No implementation checkbox substitutes for a
   selected-kernel measurement.

## Validation

Host checks cover pure enumeration, unique identities, resource bounds,
effect/lifetime ordering, deterministic source and actual emitted address
sharing. At phase completion run native warning-denied check/tests and review
interfaces. Physical follow-up must compare selected artifacts, deterministic
numerics, sanitizer results and full-chain time, then repeat 4096/64 C16 with
matched warm-up. Collect executed instructions, memory sectors, stalls and
kernel time; no single counter is a performance goal. Spark runs remain
sequential and memory-bounded, with the established 32 GiB available-memory
stop reserve. Bounded physical correctness results are listed below, separately
from performance and release claims.

## Implementation status

### Implemented source repairs

| Finding | Implemented change | Boundary preserved |
| --- | --- | --- |
| Full ingress fixes GEMM to 16 rows | Independent row multiplicity in the compiler request; pure column-window ownership derives fragment rows; backend frontier includes factors 1/2/4/8 subject to shared storage | Semantic digest and ordered reduction unchanged; no arbitrary fragment-row override in the renderer |
| Prefill search excludes KV128 | Cartesian query 32/64 × KV 32/64/128 × sync/async stage domain; old IDs retained | Resource limits reject the two KV128 double-buffer points at the tested 100000-byte budget |
| Fresh Spark plans use an RTX namespace | Exporter derives the uncalibrated architecture ID from the actual target major/minor | Historic board timings remain separate; measured resource scope still binds device/toolchain/source |
| Grouped decode does not realize page hoisting | `PagedRowOwnership` supplies the address owner for synchronous and asynchronous transfer lowering | Uniform collectives, invalid-page rejection, zero-fill and exact packed BF16 copies |
| Extra double-ring reader publication | The next key acquire closes the preceding reader epoch; explicit final handoff precedes shared-arena reuse | Per-key FP32 arithmetic unchanged; defect-injection lifetime tests retained |

For the existing async-32 fold at 4096 keys, this changes loop publications
from 384 to 256 plus one final reader handoff (excluding common setup/merge).
This is an effect-plan count, **not a measured latency reduction**.

The row-factor frontier does not silently switch the production default to a
larger tile. Full/producer-separated/unfused choices already exist and remain
whole-span measurement decisions. New row variants participate in the existing
offline resource selector and source identities, not a second autotuner.

### Validation recorded this turn

- **382/382 targeted native tests passed** across the affected compiler,
  strategy, source, AOT, tuning and exporter packages. Targeted native check,
  interface generation and formatting also passed. Commands use
  `--deny-warn --warn-list -79-20-29-25` because the workspace has existing
  toolchain-migration warnings (implicit trait promotion, deprecated APIs,
  unused imports and unqualified test imports). This is not a clean full-repo
  warning-denied release claim.
- **Spark GB10, sm121, CUDA 13.0.88:** four emitted KV128 prefill variants
  (2000/2001/2003/2004), three emitted head128 ingress variants (16/32/64 CTA
  rows), and three grouped decode variants (430/440/441) passed numerical
  probes plus memcheck, racecheck and synccheck. All sanitizer summaries report
  zero errors; racecheck reports zero hazards/warnings.
- Ingress tests cover 1–129 tokens, both sides of tile boundaries, and exact
  KV/output agreement. Bounded prefill checks cover single 16/65/128, ragged
  17+65, and query17/history257; maximum absolute error was 0.000490576.
  Decode covers tails, repeated slot reuse, mixed rows, C1/C2/C8, invalid pages
  and history4096 outside sanitizers; history4096 error was 0.0000151385.
- A separate unsanitized prefill run then covered query 16/64/128 × history
  512/1024/2048/4096, in addition to the small/ragged cases. All four variants
  passed the existing combined absolute/relative oracle. Larger cases use a
  deterministic sampled reference, not an exhaustive comparison; maximum
  absolute error over that run was 0.0114849. These structured fixture tensors
  are not model-activation qualification.
- Two diagnostic fixture mistakes were corrected without changing kernel
  semantics: private block-size macros are undefined after source emission,
  so probes now consume explicit exported launch metadata; invalid decode
  cache pages leave output untouched rather than writing NaN. Failed attempts
  remain separate from successful runs.
- The existing prefill probe also printed its full vector list when built in
  bounded mode. Its summary now reports the actual compiled scope. Earlier
  bounded logs must be interpreted by their per-case records, not that stale
  terminal vector list.
- Runs were sequential, with no model loading and small fixed probe buffers.
  Available host memory was 120534 MiB before and 120526 MiB afterward;
  no compute processes remained after the completed checks. This is before/
  after availability, not a peak-memory trace.

The prefill probe also records timings. For query128/history4096/C1 on the
same structured fixture:

| Candidate | Query tile / KV tile | Transfer | Median kernel µs |
| --- | --- | --- | ---: |
| 2000 | 32 / 128 | synchronous | 690.107 |
| 2001 | 32 / 128 | asynchronous, one slot | 430.599 |
| 2003 | 64 / 128 | synchronous | 231.049 |
| 2004 | 64 / 128 | asynchronous, one slot | 151.666 |

This is a within-frontier microprobe, not an old/new serving comparison or a
vLLM comparison. It supports keeping both ownership and overlap alternatives
available; it must not be installed as a timing record for a different shape,
ABI, device or tensor workload.

Reproduction scripts: `benchmarks/gpu_pipeline/export_counter_repair.mbtx` and
`benchmarks/gpu_pipeline/check_counter_repair.mbtx`.
Downloaded logs, sources and binaries:
`/tmp/lunaflux-counter-repair.vYVRLK9A/{prefill-ingress-results,decode-results}`.
Remote successful directories:
`/home/wlc004s/lunaflux-counter-repair.n7W4MtMY` and
`/home/wlc004s/lunaflux-decode-repair.TMJoZnwH`.
The full prefill follow-up is at
`/home/wlc004s/lunaflux-prefill-repair.w1K8Y0AV`, downloaded to the local
`prefill-full-results` subdirectory. Post-follow-up available memory was
120543 MiB, with an empty compute-process list.

### Still open — do not call the whole optimization plan finished

- Exact-shape complete-chain tuning and fresh selected-kernel counters, followed
  by matched end-to-end comparison. No speedup is claimed for these changes.
- Multi-head producer packing and cross-head rotary-value reuse with explicit
  operand/lifetime contracts; widening rows alone does not eliminate these.
- Matrix/blockwise decode with a separately represented numerical law, rather
  than silently replacing the existing FP32 per-key recurrence.
- Fragment-copy instruction reduction, proven affine current-KV views, and
  measured split-policy selection beyond the existing compiler-readonly batch
  cutoff. Do not remove a cutoff merely because more partitions are available.

These are additional algorithm/lowering and selection tasks, not completed by
the correctness of the repairs above. The functional multi-layer architecture
is retained; production deployment and release promotion were not changed.
