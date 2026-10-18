# GLM prefill projections, distributed activation scales, and serving

Status: eight full-model ablations completed; candidate numerical stability is
not established and full serving integration is incomplete. Baseline is
`a8d8fb87`; the prior deferred-HC
candidate did not demonstrate an end-to-end win and is not the new baseline.

## Experiment contract

Keep numerical programs unchanged. Finite nonnegative maxima admit exact
partitioning; floating-point sums and matrix publication retain their declared
laws. Physical IR owns partition coverage, scratch placement and producer/read
lifetimes. CUDA lowering alone owns workgroup and instruction details. No
comparison, profiler or payload hashing enters the compiler/request path.

The retained selected trace has one-workgroup input/activation scale scans and
slow BF16 prefill projections. These are hypotheses for the first-token gap,
not an additive attribution of the entire observed 11.2% completion deficit.

Budget: at most three distributed-maximum geometries and three projection
schedules in the first campaign. Measure complete producer/consumer chains,
decode as well as prefill rows, then separately benchmark each winning change
against the frozen native baseline and matched vLLM prompts. Retain losing
shapes and failures. Actual exported source, launch geometry, workspace owner
and executed symbols must agree before a path is considered selected.

## Maximum-scan design

Keep the two existing ordered launches. The first partitions the finite scan
and publishes private partial maxima beyond the numeric packed-buffer extent.
The pack launch reads all partials and derives the same scalar independently;
only its first lane publishes the scalar for subsequent matrix consumers.
No pack workgroup reads another pack workgroup's write. Scratch remains live
until every reader finishes, and is disjoint from packed payload and scales.
One-row work retains the single-partition arithmetic path. All allocation and
launch decisions are startup plans, with no per-token allocation or host scan.

## Serving boundary

The user now requests LunaFlux as the GLM serving engine, superseding the prior
instruction to restore vLLM afterward. Preserve the existing endpoint/model
contract and original container configuration as a recoverable fallback.
The original EXL3/DFlash service reports an 850000 context limit and exceeds
both the 100 GiB device budget and 16 GiB host reserve; a 64-context NVFP4 CLI
benchmark is not a drop-in replacement or proof of production readiness.
Inspect and test the actual resident HTTP/tokenization/cancellation path before
cutover, and disclose any model-format, context or API difference explicitly.

All GPU work is serialized per Spark, with 100 GiB/process and device ceilings,
zero experiment swap and at least 16 GiB host MemAvailable. Comparison stays
outside the compiler; no tolerance, reference or checkpoint changes are allowed.

## Isolated physical results (not end-to-end results)

Actual generated FP4 quantization at 64-row capacity was tested with input width
4096 and eight routed activation rows of width1024 per token. Complete scan+pack
chains use seven alternating CUDA-graph trials, not scan-only timing. Candidate
32 partitions retains two launches and adds128 scratch bytes per operand.

| Live tokens | Input chain old/new µs | Routed activation chain old/new µs |
|---|---:|---:|
|1|5.024 / 5.165|8.564 / 8.708|
|3|9.955 / 2.981|18.740 / 4.399|
|22|57.746 / 5.664|113.704 / 9.218|
|36|92.522 / 7.367|183.135 / 12.579|
|64|163.361 / 11.146|324.755 / 19.976|

Root: `/tmp/lunaflux-glm-distributed-scales-selected-20261018-v1`.
Numeric packed bytes/error results match, including zero/inactive/nonfinite and
capacity-boundary cases. Full leak/memory and race/synchronization checks pass
for32 partitions at rows1/3. This does not claim all64 rows under racecheck.
Sixteen partitions were slower in prefill;64 improve some prefill cases but
increase decode cost. Decode overhead remains visible, not waived.

Sparse projections had not forwarded matrix staging, so they always selected
synchronous copies even when recurrent projections were buffered. Independent
staging now reaches the sparse owner, all eight leaf projections and their
actual launch metadata. Same F32 outputs were checked across four projection
shapes and rows0/1/3/22/36/64/65.

| Projection N×K,22 rows | Sync µs | Buffered K32 µs | K64 µs | K128 µs |
|---|---:|---:|---:|---:|
|4096×4096|214.564|186.719|139.232|164.778|
|4096×8192|440.167|417.017|308.252|437.279|
|1536×4096|94.989|40.427|40.562|36.522|
|128×4096|93.776|40.430|41.010|36.549|

Root: `/tmp/lunaflux-glm-prefill-pipelines-20261018-v1`. Larger K is not a
universal improvement:4096×8192 at64 rows rises from431.951 µs(K32) to632.288
(K64) and884.551(K128). Resource/occupancy attribution requires counters;
timings alone do not prove the cause. The first full-model ablation therefore uses
K32 for sparse projections and leaves the already-buffered recurrent schedule
unchanged. K128 memory/leak/race/sync checks pass. The separate K32 gate at
`/tmp/lunaflux-glm-prefill-pipelines-k32-20261018-v1` also passes those checks
for rows1/22 on all four shapes.

The user subsequently withdrew the limited-service approval and requested a
full LunaFlux service comparable to vLLM. No C1/greedy-only HTTP wrapper is to
be deployed as that service. The original EXL3 containers remain stopped and
retained. Full-model ablations and service integration are separate gates.

## Full-service integration: concrete gaps, not HTTP wrapping

Reuse `service/online_tcp` / `service/openai_compat` / `service/online_session`:
they already own bounded concurrent connections, chat/responses conversion,
streaming, cancellation and scheduler integration. No alternate benchmark HTTP
server should duplicate these owners.

The current GLM execution path still has these demonstrated seams:

1. `SerialDecoderFrame::stage` accepts exactly one request and greedy mode.
   General serving needs per-request frame/state ownership, selected output rows
   and the existing sampling contract—not removal of its correctness checks.
2. `GlmDecoderStage` fixes one sequence/cache slot, and sparse layers retain a
   single contiguous history. Introduce slot/page-addressed sparse and recurrent
   state together, including cancellation, reuse and prefix lifetime tests.
   Merely raising HTTP concurrency would alias histories.
3. `ops/runtime_instance/runtime_route.mbt` has no GLM runtime recipe. Connect
   the native TP2 executor to the existing worker/service lifecycle, including
   startup, completion, cancellation/drain and failed-rank cleanup. Comparisons
   and diagnostic completion-file writes stay outside this path.
4. GLM text egress currently binds greedy candidate selection. Sampling needs
   the same request-owned temperature/top-k/top-p/RNG state and vocabulary-TP
   semantics as its declared serving policy, not greedy substitution.
5. Context and concurrency are capacity decisions constrained by actual weight,
   cache, recurrent-state, workspace and graph footprints on both ranks. The
   previous64-token diagnostic context and EXL3's advertised850000 are neither
   a service target nor evidence of memory feasibility for this checkpoint.

Acceptance must include matched prompt/tokenization tests, sequential and
interleaved clients, mixed prefill/decode, seeded sampling, streaming vs
nonstreaming agreement, cancellation/client disconnect, state reuse, long
context and exhaustion/recovery, and rank loss/drain. Monitor the100GiB/rank
ceiling and16GiB host reserve throughout. Full service is not yet implemented
or deployed; the two kernel changes do not establish those capabilities.

### Batch ownership implementation boundary

The first serving change is a model/backend-neutral `BatchedDecoderFrame`.
It consumes the existing immutable worker plan, assigns stable request slots
independently of the current batch order, and stages fixed-capacity ports for
mixed prefill/decode. New requests reset both state families. Continuations
must use the committed position; retirement is explicit and generation-aware.
State advances only after the whole GPU submission retires. Sampling retains
the original wire descriptors and output-to-plan mapping; no greedy fallback.
This is a runtime effect owner, not a new compiler numerical policy.

The host owner is implemented with nine passing native regressions covering
mixed plans, slot order, cancellation generations, partial-stage rollback,
failure retirement, output mapping, sampling preservation and context bounds.
The batched ABI includes per-sequence and per-token slot maps and absolute
positions. Existing single-history sparse kernels must not consume it until
their corresponding storage/append/attention plans have been generalized.
Adding this frame alone does not enable production concurrency or deployment.

The API audit also found streaming-only inbound support and startup-fixed output
budgets. Per-request output limits are now implemented: Chat accepts `max_tokens`
and `max_completion_tokens` (the latter takes precedence independent of key
order); Responses accepts `max_output_tokens`. Omission restores the startup
default, while explicit values may exceed that default within admitted engine
limits. Zero temperature maps explicitly to native Greedy, and top-k 0/-1 maps
to an absent filter. Invalid filters are still rejected before normalization.
Five new adapter regressions cover these behaviors and incremental work budgets.
This change does not implement non-streaming aggregation, GLM templates,
stochastic TP sampling or the remaining request fields.

### Cutover acceptance and implementation order

The accepted scope is a full resident LunaFlux service, not a C1/greedy wrapper.
The remaining sequence is:

1. Bind the batch owner to slot/page-addressed sparse state and recurrent state,
   then test mixed and reordered batches against independent request execution.
2. Connect vocabulary-sharded logits to the existing request-owned sampling
   contract. Test seeds, stop/EOS handling and cancellation without greedy
   substitution or a diagnostic completion-file handoff.
3. Add the GLM runtime recipe to the ordinary online worker lifecycle and API;
   complete non-streaming, streaming and template behavior in that shared path.
4. Establish a measured context/concurrency envelope, including exhaustion and
   recovery, with both ranks at or below100GiB and host reserve at least16GiB.
5. Run interleaved clients, disconnect/cancel/reuse, rank-loss/drain and matched
   prompt quality tests before pointing the existing endpoint at LunaFlux.

Those hardware memory bounds remain in force; withdrawing the limited-service
approval does not authorize exceeding physical memory or silently truncating
requests. EXL3's advertised850000 context is not an NVFP4 capacity measurement.
The original GLM containers are retained but stopped. No LunaFlux HTTP endpoint
has been deployed and there is currently no replacement inference listener.
The user explicitly chose to leave the endpoint offline until the full
LunaFlux service is ready; do not restore the old containers after this campaign.

### Full-model ablation caveat discovered during repeats

The isolated packed-byte equivalence checks are not a complete-model
determinism proof. The scale-only first run changed prompt0's output from eight
tokens to four. The combined reverse-order repeat also changed prompt1 and
prompt3 token IDs, as well as prompt0, relative to the retained control. These
are identical-input native runs; candidate promotion is withheld. Request0
was excluded from the fixed five-request timing cohort before repeats because
its control already had unequal work versus vLLM. A new mismatch within that
fixed cohort invalidates its comparable-work claim; do not remove more prompts
post hoc to obtain a favorable summary. Preserve the raw timing/token vectors.

The whole-model candidate is not yet proven, even though isolated scale and
projection outputs matched. Resolve the first actual divergent boundary rather
than assuming a quantizer, collective or schedule cause from elapsed time.

### Completed eight-run campaign

Source `cceebe88`, controller root
`/tmp/lunaflux-glm-prefill-ablation-20261018-v2`, completed all four alternatives
in forward then reverse order, each with one warmup and the same six literal
prompts. The original scale preparation failed before GPU launch because its
manifest predated the added selection fields; preserve the failed root and the
successful `glm-prefill-scales-matched-20261018-r1-retry1` separately.

The final control repeat retained every native token ID, as did both
projection-only runs. Scale-only r2 changed prompt2 as well as prompt0; the
combined r2 changed prompts0/1/3. Thus the observed instability is associated
with runs selecting distributed scales in this campaign, but the first
divergent arithmetic or resource-lifetime boundary has not been established.
Do not label an instruction, quantizer, collective or race as the cause yet.

The fixed cohort remains requests1..5 (31 generated tokens). Request0 is not
silently discarded from correctness evidence: its native control generates
eight tokens versus four in vLLM, so its raw timings and outputs remain in every
report but cannot represent equal-work acceleration. Do not remove additional
requests after observing candidate results.

| Alternative | Fixed-five completion, r1/r2 (s) | Mean first-token sum (s) | Mean after-first sum (s) | Native IDs across all six prompts |
|---|---:|---:|---:|---|
| Control |3.099 / 3.086|1.3160|1.7765|unchanged|
| Sparse projection K32 |3.100 / 3.048|1.2765|1.7975|unchanged|
| Distributed scales32 |not a comparable-work aggregate|not claimed|not claimed|changed|
| Both |not a comparable-work aggregate|not claimed|not claimed|changed|

The frozen vLLM fixed-five sum is 2.790644s. Control mean 3.0925s is 10.82% longer;
projection-only mean 3.074s is 10.15% longer. The latter is an observed 0.60%
reduction versus control, not a proven significant win with two repetitions.
Its 39.5ms lower first-token sum is partially offset by 21ms more time after the
first token. These are completion intervals, not an isolated GPU or decode
throughput attribution. The unchanged raw all-six vectors and invalid
candidate vectors are emitted by `scripts/summarize-glm-prefill-ablation.mbtx`;
it exits unsuccessfully when a fixed-cohort output changes instead of promoting
an attractive aggregate.

Native comparison is resident CLI completion (including diagnostic output-file
retirement); vLLM is HTTP completion. Both use the same NVFP4 model and TP2,
but native KV is BF16 and reference MLA KV is FP8. The reference is a performance
comparison, not a correctness oracle. No new hardware-counter attribution is
claimed. Host reserve stayed above 22GiB in native runs; all child units retired
successfully with zero experiment swap under their 100GiB caps. Existing system
swap was not zero and must not be confused with experiment swap.

After completion both GPUs had no compute owner. Explicit SSH checks confirmed
`glm53-exl3-head` and `glm53-exl3-worker` remain stopped, port 8888 has no listener,
and both hosts again have over 117GiB available. This follows the user's explicit
leave-offline decision, not a failed or completed LunaFlux cutover.

### Validation and committed serving prerequisite

`be153e0a` commits the batch frame owner and per-request API budgets/sampling
normalization. The affected native packages pass 52/52 tests; the full native
suite passes 5221/5221. Native checks and tests use the existing repository
migration exclusions `-20-25-29-35-79-92` with `--deny-warn`; no new exclusion
was added. A bare warning-denied invocation still reports those pre-existing
migration warnings and is not claimed to pass. Targeted `moon info` completed
with 797 existing warnings and zero errors; `moon fmt` and `git diff --check`
pass. The comparison-summary regression passes 1/1 and its actual campaign run
correctly rejects the changed-output aggregate.

The CLI now also has a header-only `inspect` mode for capacity planning. It
does not open CUDA, emit AOT, or scan/hash checkpoint payloads. Its estimate is
not a measured context capacity or a service readiness result. Unrelated
toolchain/TLS files were neither staged nor modified for this work.

### Distributed-scale investigation: defined bank bytes

The actual two-rank scale candidate was run under initcheck at
`/tmp/lunaflux-model-bench-glm-scales-initcheck-20261018-v1`. It reported
uninitialized one-byte reads in `glm_routed_3_prepare_bank`: the startup
permutation copied the entire aligned expert stride, although checkpoint
transfer initializes only payload, scale and global-scalar planes. Both exact
user units were stopped after this finding; this was not a completed model run
or a memory-limit failure. The local `initcheck-178.log` and
`initcheck-179.log` retain the reports. There is not yet evidence that these
padding reads caused the changed generated tokens.

The private lowering now derives readable spans from the immutable packed
layouts, permutes defined bytes and writes zero to alignment gaps. No payload
hash, additional full-bank scan, numerical-policy change or runtime validation
was added. An always-on source regression checks all three operand spans.

The new offline complete-chain probe uses each source's own launch geometry
and physical scratch size, independently captured graphs, varying live counts,
route occupancy and BF16 magnitudes. It compares input packing, gate/up,
activation packing, down and combination directly. The 44-step graph run at
`/tmp/lunaflux-glm-expert-scales-20261018-v4` (4096 hidden, 1024 local intermediate,
64-row capacity) is bitwise at all five frontiers; memcheck/full-leak and
initcheck pass. The input bank deliberately leaves alignment padding
uninitialized. The race checker hit its 2GiB process limit, not a numerical
failure; that result is preserved and not counted as a pass. A fresh v5 run
uses an 8GiB diagnostic cap and a 24GiB admission reserve. These are synthetic
component inputs, not whole-model correctness evidence.

The v5 campaign has now completed: memcheck/full leak, racecheck, synccheck and
initcheck all report zero errors. Racecheck peaked at 2.4 GiB and used zero swap.
The defined-bank fix is committed/pushed as `12df8a7c`.

The rebuilt full-model initcheck at
`/tmp/lunaflux-model-bench-glm-defined-bank-initcheck-20261018-v1` did **not**
finish. The watchdog stopped both ranks when `.178` MemAvailable reached
14,608,296 KiB, below the 16 GiB reserve. No initcheck errors were recorded
before that stop, but this is not a passing full-model sanitizer run. Both
ranks released their memory, with zero experiment swap. No limit was raised.

Two uninstrumented fresh runs at
`/tmp/lunaflux-model-bench-glm-defined-bank-scales-20261018-r1` and `-r2`
completed and closed both contexts. Only prompts 4 and 5 have identical output
IDs; prompts 0–3 differ. Therefore the padding fix does **not** resolve the
distributed-scale instability. Aggregate elapsed time is not comparable as
equal work. An explicit eager/no-shared-overlap control is being tested before
attributing the remaining failure to either scale arithmetic or execution order.

### Shared paged history and batched model ports

The generic immutable `PagedHistoryDomain` now separates per-request logical
context from shared physical history capacity. Scheduler page IDs and reuse
generations flow through the batched frame, sparse precision plan, private
CUDA append/pool/index/attention lowering and actual frame bindings. GLM
stage construction accepts batched capacities with this domain and forwards
slot metadata to recurrent layers. No per-request maximum-context allocation,
model comparison, hash scan or CUDA type is added to the semantic plan.

The GPU component at `/tmp/lunaflux-paged-sparse-20261018-v2` exercises five
mixed/chunked frames, reordered sequences, noncontiguous pages and physical
page reuse with a new generation. Appended values/stamps and ordered logical
selection are exact; all attention BF16 outputs match independently executed
contiguous histories on the same operands. Unused selection-slot holes may
differ and are not semantic indices. The v1 failure retained that distinction;
the corrected harness compares ordered visible selections, not neutral holes.
Memcheck/full leak, racecheck, synccheck and initcheck pass on `.179` at a 2 GiB
cap with no swap. This is bounded component coverage, not a whole-model or
performance claim.

The prepared batched rank owns metadata DMA and output/error retirement, while
whole-step commit stays with the coordinator. Its native transfer-double tests
cover memory rejection, teardown, peer-failure poisoning and zero warmed heap
allocation. Initial tests caught a test-fixture generation mismatch and an
allocating iterator in metadata staging; both are corrected rather than waived.
Paged history and rank ports are committed as `5dbfb418`. The generic batched
TP transport/coordinator now carries selected output vectors, waits for both
ranks and the commit receipt, and retires request identity plus generation.
Native tests include delayed local retirement, partial prefill, reordered
requests, reuse and stale cancellation; the complete affected transfer-double
package passes 129/129. These are not real two-Spark concurrent service tests.
Stochastic vocabulary sampling, buffered HTTP delivery and full-model service
acceptance are still outstanding. The public endpoint remains offline.

### Execution-order isolation after the defined-bank fix

Two fresh eager/no-overlap runs with the same distributed-scale32 source,
runtime and six literal prompts match 6/6 token sequences, also matching the
retained serial-scale control. Roots:
`/tmp/lunaflux-model-bench-glm-defined-bank-ordered-scales-20261018-r1` and `-r2`.
Complete six-prompt times are 3.983/3.997 seconds, versus matched vLLM 3.168596
seconds (25.7/26.1% more time). No changed-work speedup is claimed. Both ranks
release with zero experiment swap and the 16GiB reserve maintained.

This contrasts with 2/6 identical fresh outputs when graph replay AND expert
overlap are enabled. It narrows the defect to an execution-path interaction,
not yet a proof of which interaction. The first graph-only/no-overlap run also
matches all six controls; its fresh repeat is pending. Component graph
correctness alone was insufficient to demonstrate the full-model path.

The normal warning-denied native check (existing migration exclusions) and
19 affected source-package tests pass. The full service remains offline. The
batch frame is still a prerequisite, not GPU/service integration; sparse
history paging, batched model binding, sharded sampling and buffered HTTP
response delivery still need end-to-end validation before cutover.

### Overlap isolation and full-logit publication

The two graph-only/no-overlap runs at
`/tmp/lunaflux-model-bench-glm-defined-bank-graph-only-scales-20261018-r1`
and `-r2` now match all six literal output sequences and the retained control.
Their totals are 3.783/3.884 seconds. Conversely, the two eager/overlap runs at
`/tmp/lunaflux-model-bench-glm-defined-bank-overlap-only-scales-20261018-r1`
and `-r2` match only 4/6 sequences: prompts 0 and 3 differ between runs.
Graph replay is therefore not required to trigger the instability. This
isolates the failing configuration, not yet the faulty dependency or kernel.
Changed-work totals are not performance evidence. Both ranks released normally.

The new concurrent component control at
`/tmp/lunaflux-glm-expert-scales-20261018-v6` runs both actual generated expert
chains on independent streams with disjoint scratch, shared read-only bank,
and the existing 44 varying frames. It remains bitwise at all five frontiers.
This does not reproduce the whole-model shared/routed schedule and cannot
exonerate that binding. The preceding serial sanitizer results remain scoped
to their documented runs; the concurrent run is numerical, not a sanitizer pass.

`VocabularyPublication` now expresses the consumer requirement in parallel IR:
greedy candidates or all logits. Full-logit publication uses equal-width
rank-major transport, excludes uneven-shard padding, and reconstructs literal
row-major F32 logits. Precision effect plans and actual collective contracts,
allocations, launch dimensions and GLM batched bindings consume the same law.
Ordinary greedy publication remains unchanged. Sampling is outside model and
compiler semantics, using the existing seeded temperature/top-k/top-p sampler.
It cannot silently substitute greedy sampling for a stochastic request.

The actual CUDA pack/join probe at `/tmp/lunaflux-full-logits-20261018-v1`
checks a 17-token vocabulary over three uneven shards, live-output transitions
0/1/3/2/1, exact transport padding and exact reconstructed logits. Numerical,
memcheck/full-leak, racecheck, synccheck and initcheck checks pass on `.179`
under a 2 GiB/no-swap cap with a 24 GiB admission reserve. This is a component
test, not a complete model, throughput or HTTP-service claim.

Prepared host sampling budgets pinned DMA, staging, row logits, canonical
sampling scratch and per-request metadata at startup. The allocation regression
initially measured 48 heap allocations per two-output step. Generated native
code traced these to boxed mutable locals in the canonical sampler. Loop-carried
values and immutable RNG intermediates remove them without changing arithmetic,
candidate ordering or seeded selections. The repeated batched sampling test
now measures zero warmed allocations and no blocking synchronization; all
151 sampling/transfer-double tests pass. Full-model mixed batching, buffered
HTTP delivery and production service acceptance remain unproven. The public
endpoint remains offline, and no memory limit or numerical tolerance was relaxed.

### Matched retired-frontier investigation

The eager overlap run with HC control fusion disabled at
`/tmp/lunaflux-model-bench-glm-defined-bank-overlap-unfused-scales-20261018-r1`
still changes two of six control outputs. Fusion is not required to trigger the
failure. No changed-work timing is used as a speedup.

The next diagnostic captures actual per-layer full-stream attention publication,
normalized FFN input and combined FFN output after prefill retirement. These
are layer-private buffers, not overwritten residual ping-pong slots. Capture is
explicitly selected in the offline checkpoint CLI, bounded by the host reserve,
and excluded from benchmark timing claims. The compiler receives no reference
or comparison logic. Both ranks must be idle before a new probe is launched.

The paired capture is now complete: ordered
`/tmp/lunaflux-model-bench-glm-frontiers-ordered-scales-20261018-r2` versus overlap
`/tmp/lunaflux-model-bench-glm-frontiers-overlap-scales-20261018-r1`.
Both ranks exit successfully, release CUDA, and remain within 100 GiB/16 GiB
reserve with zero experiment swap. The first differing boundary is always the
combined MoE output, while that layer's normalized FFN input is still bitwise:

| Prefill | First layer | Different BF16 branch values |
| --- | ---: | ---: |
| Warmup | 22 | 29 |
| Fact | 29 | 130 |
| Arithmetic | 30 | 15 |
| Chinese | 7 | 12 |
| JSON | 19 | 43 |
| Code | 30 | 144 |
| Retrieval | None | All captured frontiers bitwise |

The literal-array report is `/tmp/lunaflux-glm-retired-frontiers-20261018-v1.tsv`.
This localizes the first divergence, not its cause: router arithmetic, routed
and shared producers, and TP publication must still be distinguished. Added
an opt-in retired MoE leaf observer for one selected layer, with named startup
buffer bindings and the same host budget. Packed scratch includes inactive
capacity; differences there alone are not mathematical output differences.
The observer does not insert device operations into the execution schedule.

### Full-service HTTP presentation

Non-streaming JSON now accumulates every visible token delta and final tail in
a startup-sized buffer. Omitted `stream` and `stream:false` select JSON; `true`
retains SSE. Both presentations use the same canonical internal token events,
request ownership, scheduler and compiler. The connection pool copies large
JSON bodies across bounded immutable output flights, retaining the semantic
terminal until its final chunk is copied. Buffer requirements include text plus
worst-case JSON escaping; no unbounded response builder is introduced.

The three affected HTTP/compatibility packages pass 134 tests, including exact
decimal Content-Length, bounded progress, deltas/tail/escaping and workspace
reuse. A real TCP test covers simultaneous JSON requests and subsequent SSE
reuse with 192-byte output cells. It is not yet a pass: this Mac cannot execute
the existing descriptor-pinned child path (Linux-only), so the test is moving
to a bounded CPU-only Spark run. Its child admission uses the existing
no-payload-read API, not the old test helper that computed an executable hash.
None of this establishes real-model service readiness or changes the offline
public-endpoint decision.
