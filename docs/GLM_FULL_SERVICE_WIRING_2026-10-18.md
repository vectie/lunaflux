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
