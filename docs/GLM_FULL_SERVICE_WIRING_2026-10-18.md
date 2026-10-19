# GLM full-service wiring

## Context/concurrency extension, 19 October (in progress)

The public 2048-context/four-slot recipe tied physical history to
`slots * context`. `ResidentHistoryCapacity` now separates logical length,
resident recurrent-state slots and the shared physical page pool. The service
accepts explicit `history_pages`, forwards the same immutable domain to the
frame and decoder lowering, and gives the scheduler that exact physical pool.
One maximum-length request plus the decode reserve must fit. For example,
513 sixteen-token pages can serve one8192-token request, four2048-token requests
or sixteen512-token requests; sixteen slots do not promise sixteen8K histories.
The old omitted-field recipe preserves four-slot/2K allocation.

The first bounded campaign targets8K/16K/32K and concurrency1/4/8/16. CPU-only
`glm_service plan` reports actual checkpoint-bound device estimates before
export or GPU allocation. Reject plans above100GiB; a physical plan also needs
at least16GiB observed host availability throughout loading and inference.
Comparison stays in external MoonBit automation, uses matched prompts/output
budgets, and reports queueing, TTFT, completion, output counts and memory. No
payload hashes, changed arithmetic or compiler reference-engine dependency.

At the first live check both GPUs were already free: rank1 had aborted at
02:04:01CST and the manager stopped rank0 at02:04:03. The peer recorded SIGABRT,
not an OOM, with zero unit swap. The cause is not established and must not be
described as fixed by the capacity change. Existing public acceptance results
are historical, not evidence of continued service availability.

This continues the complete-answer campaign, not its replacement. Public HTTP
stays offline until real service acceptance on both Sparks. Keep the existing
canonical scheduler, tokenizer, request/event semantics and HTTP adapters;
do not wrap the serial CLI in a second admission queue.

### Capacity and host-memory findings (campaign still in progress)

The actual checkpoint-bound device plans, including the 2 GiB reserve, are:

| Logical context | Slots | Physical pages (16 tokens/page) | Planned bytes/rank |
| --- | ---: | ---: | ---: |
| 8,192 | 4 | 513 | 103,650,745,500 |
| 8,192 | 8 | 513 | 103,953,452,188 |
| 8,192 | 16 | 513 | 104,558,865,564 |
| 16,384 | 8 | 1,025 | 106,958,957,724 |
| 16,384 | 16 | 1,025 | 107,564,371,100 (over 100 GiB) |
| 32,768 | 4 | 2,049 | 112,667,262,108 (over 100 GiB) |

Device-plan admission is not host-memory qualification. The current expanded
DSA K/V representation is a real capacity cost; it must not be described as
equivalent to vLLM's compressed FP8 MLA cache. Sixteen short requests and sixteen
full-length histories are distinct workloads. Physical observations decide
which configuration can be installed.

Startup also exposed two host-plan integration defects. A fixed four-slot
admission-storage allowance could not instantiate the larger lane pool.
`LunaRequestPreparationStorageBudget::required` now computes the same immutable
cell requirements that the allocator validates. Separately, the tokenizer's
4,096-byte vocabulary *admission ceiling* was being used as the maximum decoded
piece, inflating every buffered HTTP response. The admitted vocabulary's actual
maximum is 512 bytes; output workspace validation and sizing now share that
bound plus stop-text and UTF-8 carry. The bound is retained at preparation,
not rescanned in token steps. The output-token allowance remains 8,192.

Admission lanes are independent of recurrent resident slots. The explicit
16-slot/16-lane candidate admits at most sixteen owned requests, not sixteen
active plus sixteen queued. Its preparation backing is 688,816,544 bytes and
buffered response backing 795,995,874 bytes, together below the 2 GiB reserve.
Object/allocator overhead and all other runtime memory still require physical
host-floor observation. Earlier startup failures remain evidence, not passes.

The first vLLM 8K/C16 attempt used 1,536 MiB KV memory and default graph captures
through 32. During graph warm-up, rank0 host availability reached 16,595,224 KiB,
below the required 16,777,216 KiB. The watcher stopped both ranks before any
timed requests. This is not an inference performance result or a license to
relax the reserve. A distinct retry uses 512 MiB KV and capture maximum 16.
Its actual capacity, answers and timings must be reported separately.

The second reference attempt also crossed the floor during capture:
16,654,912 KiB host availability at 94,594 MiB observed GPU residency. The
watcher again stopped both ranks; no timed reference inference completed.
The next prepared reference configuration has 256 MiB KV, 64-token prefill
budget and capture maximum16. It must qualify before yielding comparisons.

The 8K/16-slot LunaFlux candidate38d2d266 started and passed JSON/SSE/Responses
smoke checks, but its first timed cohort crossed the same host floor:
16,666,372 KiB at 98,528 MiB GPU residency. The paired owner drained and released
both GPUs with zero unit swap. This is a rejected deployment configuration, not
a successful16-slot qualification. An8K/eight-resident-slot/sixteen-admission-lane
candidate is being compiled; at most eight requests can own resident GPU state,
while additional admitted requests can queue. Preserve that distinction in
metadata and benchmark reports.

The full local native workspace suite passes5,293/5,293 and the native check
passes with the existing migration-warning exclusions20/25/29/35/79/92. The
workspace still contains unrelated migration changes; this is not a clean
commit-pinned ARM aggregate-test claim. The pinned38d2d266 ARM release build
passes. The earlier2GiB ARM aggregate compile was OOM-killed, and the interrupted
two-job local compile is retained; neither is described as a test pass.

The eight-resident candidate loaded within the memory limits and completed
the short C1/C4/C8 cohorts. The C16 burst exposed a separate admission defect:
all new requests enter Waiting before resident assignment, but the service
set waiting capacity to eight resident slots despite sixteen preparation
lanes. Seven requests received HTTP429 and the external client canceled its
remaining peers. This is not a successful C16 campaign. The scheduler policy
now sizes Waiting for the complete admission burst while leaving eight GPU
resident slots and the same physical history pool unchanged. A regression
checks that independent mapping. The old partial run is retained under
`/tmp/lunaflux-glm-service-benchmark-luna-20261019-v3`.

The queue repair is source-pinned at47a1f351. Its clean ARM release build
passes (1.1GiB reported build peak, zero unit swap); full local native tests
pass5,294/5,294 with the unchanged warning exclusions. Fresh export directly
matches the prior eight-slot CUDA sources, so AOT kernels are reused without
payload digests. No numerical kernel implementation changes in this repair.

Reference configurationv4 (8resident ceiling,64prefill tokens,256MiB KV,
capture maximum8) loads and serves within the resource envelope. However its
hybrid cache has only three blocks; at2K/C16 the observed active count is one
with fifteen waiting. It is not an adequate reference for effective batching.
The partial first-use campaign is retained under
`/tmp/lunaflux-glm-service-benchmark-vllm-20261019-v2`; it was deliberately
stopped after2K/C16 and before the longer cohorts to test a768MiB KV candidate,
not declared a full pass.
The baseline also emits substantially different reasoning/output lengths and
some repeated/truncated answers on identical serialized prompts. Compare
answer correctness and actual token counts separately; its longer completion
time is not proof of equal-work engine slowdown. First-use JIT warnings are
retained and require a warmed measurement pass before steady-state claims.

The external HTTP client additionally accepts `LUNAFLUX_BENCH_MAX_TOKENS=1`
for a separately labelled prefill-plus-one-generation control. It preserves
the literal prompts and all context/concurrency vectors; the normal128-token
complete-answer campaign is unchanged. The comparator checks matching output
budgets and records actual output counts. A one-token truncated response is
not a quality pass, and client event arrival is not a per-token GPU timestamp.
This control exists because naturally different reasoning lengths confound
equal-work completion comparisons; it does not modify either engine.
The additional `latency` mode runs C1 at all four prompt lengths, leaving the
complete-answer `all` mode's concurrency sweep unchanged.

Reference v5 (768 MiB KV, capture maximum eight) also failed the physical
host-reserve gate during graph warm-up: 16,455,900 KiB available on rank0 at
95,030 MiB GPU residency. The external guard stopped both containers; no timed
requests completed. Its logs are retained under
`/tmp/lunaflux-reference-bench-glm-context8k-20261019-v5`.
A distinct v6 configuration reduces capture maximum to four and KV to512MiB;
it is prepared, not yet qualified. Do not relax the memory floor or count a
loading failure as a performance result.

The corrected native47a1f351 run at
`/tmp/lunaflux-glm-http-context8k-c8-20261019-v3` completes the short C16 burst
without429s. All sixteen distinct labels are correct. Its ongoing external
campaign is `/tmp/lunaflux-glm-service-benchmark-luna-20261019-v4`.
The completed2,085-prompt-token cohorts take21.703s(C1),71.769s(C4),
141.091s(C8), and283.130s(C16). Longer cohorts, general service acceptance and
the matched reference are still pending; these are not full campaign passes.

Capacity is not throughput scaling. The observed native queue commonly has
one or two active requests during these short-answer prefills, while additional
requests wait. The selected eight-slot AOT source still has a single-thread
top-k index publication and repeated per-head duplicate-index checks, followed
by ordered sparse softmax/value folds. Those are literal implementation facts,
not a measured attribution of all elapsed time. No profiler has yet assigned
the new long-context latency to individual operations. The capacity work
does not claim to remove those costs or match vLLM's compressed-cache layout.

## GPU sampling

The full-logit path now has a backend-neutral total-order permutation plan in
elementwise physical IR. Every compare/exchange stage is a disjoint matching;
stage publication precedes the next matching. The existing stochastic final
program retains max-shifted temperature softmax, minimal top-p prefix,
ordered F64 mass folds and the request's counter-based draw. Ordering is by
descending logit, then ascending token ID. F32 input is never narrowed to BF16.

Private CUDA lowering emits initialization/finite validation, compare/exchange,
parallel probability evaluation, and final selection. A prepared integration
owner captures the whole fixed schedule at startup. Only sampling parameters
are uploaded and eight token/status bytes per live output are downloaded.
The request seed/index is independent of batch placement or retry. Workspace,
mailboxes, transfers and graph ownership are allocated once and explicitly
drained/closed. No runtime compilation, model payload hashing, reference-engine
dependency or host sampling fallback is introduced.

GLM full-logit rank binding requires the separately compiled sampling AOT. The
existing greedy-head path remains available. This is a capability requirement,
not a claim that the full service has been installed or qualified. The initial
bitonic schedule prioritizes complete semantics; its O(V log-squared V) ordering
cost and ordered mass folds must be measured, not advertised as optimal.

`cmd/full_sampling_probe` is an offline physical fixture using the actual
prepared owner and canonical request parameters. Its host sampler is an
independent test control, not a production dependency or a universal numerical
oracle. Cases cover unrestricted sampling, top-k/top-p, greedy ties, values
distinguished by F32 but not BF16, nonfinite rejection, varying/zero live rows,
and repeated request-index draws. Kernel sanitizer and actual-owner results
must be recorded before full-model use.

## Request execution and retirement boundary

The canonical HTTP worker now binds the real batched GLM TP owner. A terminal
request releases recurrent/KV slot ownership only after all submitted work
retires and both ranks acknowledge the exact request ID/generation. Requests
cancelled before their first GPU step have an explicit no-state retirement.
Absent rows in a batch are not evidence that a request has finished. Do not
reuse a slot by inferring liveness from the most recent batch.

Acceptance requires complete JSON and SSE answers, mixed concurrent
prefill/decode, seeded GPU sampling, stop/EOS/usage consistency, cancellation
and HTTP disconnect with reuse, saturation/capacity recovery and failed-rank
drain. Model tests, echo workers and transfer doubles alone cannot pass this
gate. Admission must respect the measured context/concurrency capacity under
100 GiB per Spark, 16 GiB host reserve and zero swap.

## GPU sampler qualification and service integration

`/tmp/lunaflux-full-sampling-20261018-v4` completes on both Sparks with the
actual prepared owner, 24 cases repeated twice at V=513 and V=155648, four
maximum output rows. Both geometries pass token checks and full leak/memcheck.
The tail geometry passes synccheck and racecheck restricted to the only
shared-memory kernel (initialization), with two sanitizer workers. This is not
full-vocabulary race coverage. Earlier unbounded race-worker attempts exceeded
their instrumentation cgroup memory caps; they are preserved, not passes.
All successful probe units close CUDA and report zero unit swap. Host-wide old
swap residency is distinct and is not reported as zero.

The new `cmd/glm_service` binds the canonical HTTP pool, tokenizer, admission
lanes and scheduler to a distinct GLM batched TP bootstrap recipe. Full-F32
sampling runs on the output rank. Retirement is an explicit ID/generation
receipt after both ranks release their state, including queued requests that
never submitted GPU work. Cancellation uses the original device generation.
Physical acceptance of this command is recorded below and in the complete-answer
report; it is not yet publicly installed.

HTTP integration adds a once-per-conversation document prefix and model EOS
union before the typed handoff declares its stop count. GLM explicitly admits
checkpoint special tokens through the generic tokenizer policy; existing
services retain their rejecting default. Worker idle waits have no poll-count
or wall-time lifetime cap; partial transfers retain their I/O deadline. The
HTTP Ready handshake waits for both ranks to finish preparation. Deployment
identity fields are supplied labels, never computed payload hashes.

The original wiring passed 376 affected native tests. Subsequent real HTTP
campaigns exercise both GPUs, rather than reusing the earlier CLI result.

## Lifecycle repairs and current acceptance

Full-service validation exposed two independent host-runtime defects after
successful model generation. Multi-request disconnect could pin another lane's
terminal event while retrying an earlier lane's abort. Retirement now consumes
that pinned event first and attributes it to its owning request. A separate
active-stop abort was reproduced without a model or GPU: async0.20.3 could
under-count live I/O after its signal task retired. The unmodified published
async0.22.4 dependency removes that signal-task accounting defect. Prepared-worker
ownership is retained across listener startup and protected cancellation cleanup.

These repairs affect explicit runtime effects, not numerical IR or compiler
comparison policy. There is no payload hashing, CPU sampling fallback, recurring
keepalive workaround or vendor-library patch. Actual pinned ARM signal-cleanup
and owned-worker startup fixtures each pass three repetitions. The latest
full-model campaign uses source71061272, context2048, four resident slots and
eight admission lanes under the100GiB/rank and16GiB host-reserve envelope.
The final HTTP, active-stop and peer-loss results belong in
`GLM_COMPLETE_ANSWERS_AND_SERVICE_2026-10-18.md`; none implies unrestricted
context, arbitrary concurrency or every feature of another serving product.

Public routing remains separate from private acceptance. The existing managed
route identifies the old EXL3 checkpoint; the tested checkpoint is NVFP4. Its
model metadata, worker ownership and forwarding credential must be connected
truthfully before cutover. A loopback acceptance listener with a fixture
credential and a bounded test lifetime is not a production deployment.

### Completed private acceptance

Pinned71061272 passes all nine real HTTP groups on the paired Sparks in v25.
The terminal observations show zero active requests, queue depth and KV usage,
with zero worker failures/restarts during that campaign. Twelve broader answers
finish naturally (4,007 tokens in319.579367 summed request seconds); content
review still identifies Markdown around requested JSON and factual caveats.
Those are not converted to unqualified correctness passes.

Coordinator-first `KillMode=mixed` units then pass v27 active shutdown with
four active/one queued and v28 deliberate peer loss. Actual exit receipts and
empty GPU-owner observations establish cleanup; default status of a collected
unit is insufficient. Both GPUs are released. Peak observed residency is
97,654/97,669MiB; the lowest host reserve across these accepted runs is
18,184,988/19,517,468KiB, with zero unit swap. The isolated full native suite
passes5,275/5,275 and the added transport regression passes8/8 affected tests.
Existing migration-warning exclusions are unchanged. Detailed positive and
negative evidence remains in the complete-answer report. Public integration
is still pending; the old EXL3 route must not misrepresent this NVFP4 runtime.
