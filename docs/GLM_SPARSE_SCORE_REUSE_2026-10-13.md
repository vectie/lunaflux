# GLM sparse attention: retained scores and parallel independent axes

## Problem and reference implementation

The selected native GLM sparse attention launches one block with one thread.
The recorded first-rank sample attributes 90.425% of kernel time to it. Each
QK dot is evaluated for the maximum, denominator, and again for every value
component (258 evaluations for value dimension 256). This is selected executable
coverage, not a limitation of functional compilation or evidence that numerical
agreement is sufficient for performance.

The local vLLM implementation in
`../vlm/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` converts
selected indices to physical addresses and dispatches FlashInfer sparse MLA.
It retains compressed FP8 MLA KV, unlike our expanded BF16 Q/K/V. We learn
independent-axis parallelism and score reuse without pretending that changing
cache precision or reduction order is an equivalent rewrite. Framework
comparison remains outside the compiler.

## One bounded implementation iteration

1. Keep the existing numerical expressions: ordered RN multiply/add QK and PV,
   stable softmax, existing exponential/division, BF16 probabilities/output.
2. Add a pure physical plan for disjoint query/head workgroups, stable retained
   valid-key rows, score storage and independent score/value workers. Preserve
   sentinel, duplicate, causal, invalid-row and inactive-output semantics.
3. CUDA lowering implements publication barriers and device indexing. The
   actual frame consumes generated launch metadata and owns/account/releases
   scratch at preparation. No allocation, comparison or hashing in execution.
4. Retain the serial leaf only as the existing offline diagnostic/reference;
   the generic indexed-attention execution source selects the new plan.

Scratch capacity is `min(selected_width, maximum_history)` per query/head,
with four bytes for each retained key and score. Valid rows have unique keys,
so this bound cannot truncate legal work. Rows failing validation publish zero.
Scores are overwritten by exactly represented BF16 probabilities only after
all score consumers retire; no additional model weights or KV allocation.

Validation budget: deterministic varied/tail/GQA/invalid fixtures, up to five
paired timing trials per representative short/long kernel shape, changed-source
sanitizer/leak checks, and the existing six-prompt GLM corpus against the frozen
vLLM results. No new full-model GPU run until both hosts are reachable and idle.
Keep 96 GiB process caps, no process swap and 16 GiB host reserve for full-model
runs; kernel probes use 2 GiB/no swap. New results go to non-overwriting paths.
No payload hashes or integrity scans. The unavailable .179 is an infrastructure
blocker for two-Spark timing, not a reason to fabricate or reuse unmatched times.

## What the installed vLLM backend does

Read-only inspection of the actual retained GLM image's FlashInfer source
(`flashinfer/mla/_sparse_mla_sm120.py` and
`data/csrc/sparse_mla_sm120_decode_dsv3_2.cu`) shows:

- At most 64 query tokens dispatch to decode; larger query batches use prefill.
- Decode distributes tokens, head groups and KV splits across the launch grid.
  Top-k 2048 creates 32 splits of 64 selected keys before a separate merge.
- Each block has 288 threads: one IO warp and eight math warps. Two shared
  buffers overlap TMA KV gathers with computation; FP8/BF16 matrix operations
  implement the score work. The block uses approximately 98 KiB of shared memory.
- Online block softmax and a split merge have a different arithmetic topology
  from LunaFlux's ordered scalar folds. Its compressed FP8 KV is also a different
  storage contract. These are future explicit alternatives, not bitwise rewrites.

This is source-level dispatch inspection, not a new hardware-counter capture of
the full request. The present change learns the independent-axis decomposition
and materialization principle. It does not claim to implement all of FlashInfer.

## Spark kernel result, 13 October

Paired root: `/tmp/lunaflux-glm-sparse-retained-20261013-v4`, on .178 GB10.
Generated old and new leaves share the same varied BF16 inputs, dimensions and
numeric expressions; five alternating-order CUDA-event trials follow warmup.
New-kernel events average 20 launches, old-kernel events one launch. These are
isolated GPU times, not request latency or throughput.
This repeat uses production CUDA flags (`-O3`, default FMA setting, sm121).
The retained v2 run additionally disabled implicit FMA and showed the same
numerical results and essentially the same timing. Arithmetic intrinsics
already preserve the specified ordered RN operations in both builds.

| Geometry | Old median ms | Retained-score median ms | Scratch |
| --- | ---: | ---: | ---: |
| 64 heads, 256 QK/V, 16 live keys, 2051 selected slots | 1936.603 | 0.356702 | 32 KiB |
| Same, 32 live keys | 2485.177 | 0.383163 | 32 KiB |
| Same, 64 live keys | 3549.122 | 0.411600 | 32 KiB |
| 2 query heads/1 KV head, 2048 live keys, 2051 slots | 4037.726 | 0.794133 | 32816 B |
| 3 queries, 6/2 heads, QK17/V33, 7 live keys | 1.085824 | 0.006174 | 1584 B |
| 2/1 heads, QK8/V8, 16 live keys, 65536 slots | 87.026 | 25.616364 | 512 B |

All 60 valid, sentinel, duplicate, invalid-index, future-key, invalid-mask and
inactive-query fixtures are bitwise identical to the retained serial leaf.
This establishes preservation of that numerical program, not independent proof
that the model/reference is correct. The full new-kernel fixture matrix passes
memcheck/full leak, racecheck and synccheck: zero errors/hazards/leaks. Probe
v2 paired unit memory peak was 69.1 MiB, zero swap; every run was capped at
2 GiB. This is process-accounting data, not a claim that systemd accounts for
all unified GPU memory; the host reserve is checked separately.
An independent single-key identity test at the fresh v3/v4 roots passes all six
geometries: probability is one and output copies the corresponding value,
including grouped-query head mapping. This does not use the serial kernel as
an oracle. Invalid-row zero-output assertions are also independent.

The maximum selected-width case exposes remaining selection-validation cost;
this change does not claim optimal long-context scaling. Ordered denominator
and per-column PV folds are deliberately retained. Pooling/index and hyper
connection schedules, projections, and one-row prefill are not fixed here.

The compiler/frame path now consumes the new source, `(queries*heads,256)`
launch and owned scratch. The kernel entry has a new ABI-specific name because
the scratch argument was added: old AOT cannot silently run with the new frame.
Scratch is counted in the decoder/layer workspace and rejected before any
allocation when the budget is insufficient, including IndexShare consumers.
There is no steady-state allocation or runtime benchmark dependency.

Affected native tests pass: physical IR 36, generated selected source 3,
indexed source 2, sparse runtime fixtures 6, GLM integration 17; benchmark
runner tests 2. The GLM checkpoint native check passes using the repository's
existing migration warning exclusions (`-20-25-29-35-79-92`). The unexcluded
full check still reports unrelated toolchain migration warnings; this is not a
new clean full-suite claim. Unrelated dirty toolchain/TLS files are preserved.

The failed v1 probe compilation (missing newline between generated leaves) is
retained; source concatenation was corrected before the successful v2 run.
The AKO loop is retained, but its older `ako_trial.mbtx` wrapper performs
mandatory `sha256sum` scans. This task deliberately uses the bounded,
no-hashing paired runner instead, honoring the user's explicit instruction.

## Committed full-model export

Implementation `015722c7bff4ee992b50749cfe3460ec822f1e8a` is pushed to main.
Its clean, CPU-only ARM build passes on .178 (220 tasks, reported 160.3 MiB
process peak, zero swap, 2 GiB cap). Runtime/artifact root:
`/tmp/lunaflux-checkpoint-build-glm-sparse-retained-20261013-v1`.

The committed executable exported the real 45-layer GLM placement at physical
rows1/history64. Startup explicitly reports `payload_hashing=false`; existing
shard labels are reused, no payload digest is computed. Planned stage bytes are
102064287736 and 91516690644, both under the existing 103079215104-byte limit.
The actual stage sources contain six and five new
`*_attention_retained_scores_v1` entries respectively. Both full stage cubins
compile for sm121 under independent 2 GiB/no-swap CPU units and are downloaded
along with the binary and source without overwriting previous artifacts.
Existing CUDA unused-variable warnings remain visible. No full model was
loaded for these checks; this proves export/build propagation, not execution
of all 45 layers or end-to-end correctness.

## End-to-end comparison status

The unchanged six-prompt corpus and frozen vLLM GLM measurements are in
`BENCHMARK_SPARK_MODEL_COMPARISON_2026-10-12.md`: 0.385–0.675 seconds per
buffered HTTP request, 3–8 output tokens, C1, excluding loading. Those are not
comparable to the kernel-only numbers above. Native old 643/671-second examples
used different prompts/output lengths and are not a matched before/after.

.179 initially timed out during fresh SSH banner exchange. At that point its
previous SGLang process and memory release could not be verified, so no new
two-Spark full-model benchmark was launched. The subsequent recovery and actual
completed measurement below supersede that infrastructure blocker.
The benchmark runner accepts `--runtime NEW_RUNTIME` and retains that explicit
binary/AOT selection across prepare/start/collect, avoiding a silent benchmark
of the old hard-coded runtime. Fresh AOT from the committed executable is used
for the recovered run, preserving the exact six prompts/EOS/output cap.

### Recovery and selected full-model execution

On the subsequent requested retry, .179 accepted a fresh SSH login. Both hosts
had no GPU owners or running containers and approximately 117 GiB available.
The committed executable and fresh AOT above were staged on both machines;
no payload hashing or integrity scan was performed.

Native root `/tmp/lunaflux-model-bench-glm-retained-20261013-v1` completed all
six requests on the full 45-layer model, with source commit `015722c7`, fresh
retained-score AOT, PP2/TP1 and BF16 KV. Both user units exited successfully,
explicitly closed their CUDA contexts and used zero process swap. The monitor
retained the 16 GiB host reserve throughout. No profiler overlapped the run.

| Category | Input / output tokens | Request seconds | Finish |
| --- | ---: | ---: | --- |
| Fact | 22 / 4 | 45.141 | EOS |
| Arithmetic | 23 / 4 | 46.950 | EOS |
| Chinese | 22 / 8 | 52.419 | Length |
| JSON | 23 / 8 | 54.209 | Length |
| Code | 24 / 8 | 56.036 | Length |
| Retrieval | 36 / 3 | 68.720 | EOS |

Total measured request time is 323.475 seconds, excluding checkpoint loading.
These are resident CLI completion times, including single-row prefill,
generation, diagnostic completion-file output and request retirement—not
decode-only throughput. The native
first request has no excluded warmup. The 179 serial model steps recorded by
transport epochs imply 1.806–1.808 seconds of request time per step, including
host/pipeline overhead. This is a normalization of whole-request time, not an
independently measured per-kernel or decode latency.

The sparse fix demonstrably reaches full-model execution, but does not remove
the remaining slow decoder implementations or one-row prefill. Historical
643/671-second requests used different prompts and EOS settings; they still
cannot support an exact before/after speedup claim. The fresh vLLM comparison,
including all strategy, timing and output caveats, is recorded in
`BENCHMARK_SPARK_MODEL_COMPARISON_2026-10-12.md`.
