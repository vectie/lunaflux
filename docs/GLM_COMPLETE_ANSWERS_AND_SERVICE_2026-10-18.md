# GLM complete-answer and real-service acceptance

The previous six-prompt, eight-output-token timings are not a quality corpus.
This campaign uses independent expected answers/task checks, not reference
engine tokens. All evaluation remains outside the compiler and runtime.
No payload hashes, tolerance changes or production promotion are introduced.

## First campaign

Use both GB10 Sparks as TP2, 64-row prefill capacity, context1024 and at most512
new tokens. Re-export the actual AOT for that context rather than relabel the
old64-context artifact. Plan under100GiB per rank with its existing2GiB internal
reserve, externally cap each process at100GiB/no swap, and stop both ranks if
either host falls below16GiB MemAvailable. No overlapping GPU workloads.

Freeze a broader corpus before inference: factual short answers, multi-step
arithmetic, JSON/array formatting, code, multilingual translation, supplied-text
retrieval, longer distractor context, and reasoning. Retain complete raw tokens,
reasoning and final text, finish reason, input length and runtime selection.
An exhausted output budget is inconclusive; increase the admitted envelope and
rerun that case rather than count a truncated answer as correct. Do not silently
strip malformed framing or count reasoning-only text as the final answer.

Begin with the retained ordered-branch control, then the overlap candidate on
the identical corpus and envelope. Repeat relevant failures and isolate model
errors from request-state or decoder presentation errors. Stable wrong answers
are not fixed by copying another engine; semantic/state violations require a
source fix with regression tests. Code execution must be separately sandboxed
and bounded; never execute generated code inside the serving runtime.

## Service gate (not the CLI gate)

The resident serial CLI provides an initial model-output check only. The public
endpoint stays offline until the existing full HTTP/scheduler path is wired to
the batched GLM TP owner and validated on both GPUs. Required coverage includes
complete JSON/SSE answers and their agreement, multiple clients, mixed/chunked
prefill and decode, seeded sampling, cancellation/disconnect and slot reuse,
capacity rejection/recovery, and failed-rank drain. Echo-worker transport tests
and CPU tests do not establish this real-model acceptance.

The memory envelope must be measured for the actual context/concurrency plan;
the first1024-context diagnostic is not a declared production capacity limit.
Do not restore the old GLM containers or publish an incomplete replacement.

## Generation-control correction before running

The installed checkpoint declares EOS IDs154820,154827,154829. The old serial
CLI admitted only one stop ID and its inference limits allowed only one. It now
accepts an explicit bounded comma-separated set, preserving the old single-ID
and `-1` forms. The complete-answer harness supplies all three declared IDs.
Each retired result writes a finish reason and actual token count separately
from its raw tokens. This is a generation-control fix, not a numerical-policy
change or evidence that earlier answers actually crossed an ignored EOS.

Local warning-denied native check passes with existing migration exclusions.
The command package's3 tests and serial-generation package's11 tests pass;
the extended offline runner passes5 tests. Interface generation reports1096
pre-existing migration warnings and zero errors. These are preflight checks,
not physical-model or full-service acceptance.

## Source-pinned campaign and build compatibility

The executable/AOT for the first complete-answer run is pinned to598cf66c.
The first clean ARM build at82482cdc failed because the newer local formatter
rewrote `Int` to `Int32`, which the pinned Spark toolchain does not support.
598cf66c restores compatible source syntax; the fresh ARM build succeeds.
This was a build failure, not a GPU numerical failure. The failed build is
retained at`/tmp/lunaflux-checkpoint-build-glm-complete-answers-20261018-v1`.

The new AOT is at`/tmp/lunaflux-glm-expert-tp-complete-answers-20261018-v1`;
export and both CPU-only compilers complete. Planned device bytes per rank are
100898667632 (including the plan's reserve), below107374182400. Runtime process
and available-host-memory measurements are separate and remain required.

`scripts/assess-glm-complete-answers.mbtx` is an offline task assessor. It
retains raw text, requires explicit reasoning closure for this thinking prompt,
does not turn length truncation into a pass, and checks JSON/integer and
supplied-context tasks independently of another engine. Facts, translation,
summary, code and SQL require semantic/manual review in addition to framing.
Its2 tests pass. The command package's4 tests now include every declared EOS
as both the final-prefill token and a later decode token; all terminate without
another submission. The offline watcher retires both owned ranks on peer
failure or expiry, not only on a host-reserve violation.

## Concrete full-service gaps still visible in source

Do not equate `cmd/glm_checkpoint`'s serial TP generation with the canonical
HTTP service. The batched rank/pipeline and GLM page/slot bindings exist but
are not installed as the live canonical HTTP worker. In addition,
`integration/batched_logit_sampling` currently downloads full F32 logits and
samples on the host. It must not be presented as GPU sampling. The older
standalone CUDA sampler supports bounded restrictive top-k over BF16, not a
drop-in full F32-logit sampling replacement. The user's requested GPU execution
and full service acceptance remain unmet by those components alone.

A real adapter must retain canonical scheduler ownership, publish only after
both ranks retire, and release request-specific recurrent/KV state on the
canonical terminal/cancellation path. Do not bridge the CLI with a separate
HTTP queue and call it continuous batching. These are implementation/acceptance
requirements, not requests for more hardening or bit-identical tensors.

## First physical complete-answer results

Both TP2 campaigns used the same598cf66c executable/AOT and the literal15-case
corpus,512-token output allowance and1024-context envelope. The ordered-branch
and independent-expert-overlap runs each finished all15 measured requests by
declared EOS, without length truncation. Both ranks exited0, explicitly closed
their CUDA contexts and reported zero cgroup swap. Warmup is excluded below.

| Independent task | Ordered branches | Overlap |
| --- | --- | --- |
| Japan fact, including later repeat | Tokyo, correct | Tokyo, correct |
| Multi-step arithmetic / time conversion | 449 / 225 | 449 / 225 |
| Exact JSON object / sorted array / ordering logic | All exact task values | All exact task values |
| Chinese fact / English translation | Paris / cat under table, correct | Same meaning, correct |
| `unique_sorted` code | `sorted(set(values))`, manually reviewed correct | Same complete function |
| SQL aggregation | Query correct; **fails requested only-SQL format** (Markdown fence) | Same failure |
| Conflicting successive request notes | Mira then Jon, both correct | Mira then Jon, both correct |
| Longer supplied-context retrieval | lilac-81, correct | lilac-81, correct |
| One-sentence library summary | All four required facts, no invented fact | Different wording, all four facts |

Semantic/manual assessment is14/15 passing the full task instruction per run;
SQL is not counted as a pass simply because the query itself is correct.
This repeatable formatting error is observed model behavior, not a demonstrated
compiler arithmetic defect. No output sanitization, prompt rewrite or numerical
policy change is used to conceal it. The summaries' different wording does not
constitute a correctness failure. Code was reviewed, not executed in a Python
test runtime. This small corpus is evidence of these tasks, not general model
quality, concurrency correctness or stochastic correctness.

The ordered run produced212 measured tokens,3–37 per answer. Minimum sampled
host MemAvailable was22459732KiB (21.42GiB), overlap22518976KiB (21.48GiB).
Both stay above16GiB. A live overlap observation showed94676MiB resident GPU
memory on each Spark (92.46GiB). That single observation is **not** a measured
peak; the early watcher recorded cgroup and host memory, not GPU residency.
The extended campaign adds five-second GPU-residency observations and a100GiB
stop check. Cgroup peaks must not be substituted for unified GPU-memory peaks.

Retained roots:

- `/tmp/lunaflux-model-bench-glm-complete-answers-ordered-20261018-r1`
- `/tmp/lunaflux-model-bench-glm-complete-answers-overlap-20261018-r1`

The additional frozen corpus `glm_complete_answers_extended_v1.json` asks for
a16-object JSON array,220–260-word explanation, a complete interval-merging
function and the identical SQL prompt again. It addresses the first corpus's
short *actual* answers, not just its large permitted budget. Its run cancels
the unmeasured warmup after preparing its first step, drains that actual step
and releases both request owners before the four measured requests reuse
state. This is a resident TP-owner check, not an HTTP-disconnect test.

## Extended results and remaining correctness distinctions

`/tmp/lunaflux-model-bench-glm-complete-answers-extended-20261018-r1` completed
on both Sparks using the overlap path and the same source-pinned runtime.
The unmeasured warmup's real prepared step retired with
`request_cancelled=true`; all four following requests completed naturally.
The pipeline does not cancel a partially executed kernel: it drains the step,
suppresses its sampled token, releases both request owners, then accepts reuse.

| Task | Actual generated tokens (including framing/EOS) | Finding |
| --- | ---: | --- |
| 16 JSON objects with squares1–16 | 195 | Exactly matches every required key/value and array order; no omitted entries/fences |
| FIFO/LIFO explanation | 297 | 242 words, correct insertion/removal examples and uses; **one overbroad claim**: sorting does not universally require comparisons (counting/radix sorts are counterexamples) |
| Merge inclusive intervals without input mutation | 97 | Complete function, manually reviewed correct: sorted outer copy, fresh output inner lists, `start <= previous_end` merges touching endpoints |
| Identical SQL prompt | 30 | Correct aggregation/tie order, same Markdown-format failure |

All four finish with EOS, not the512-token limit. The generated SQL query was
also executed separately against an in-memory SQLite fixture, yielding
`B|10, C|10, A|8, D|-2` in the expected order. That checks SQL semantics only;
it does not turn the fenced response into compliance with the output format.
Python functions were manually reviewed, not executed; do not label them as
interpreter-tested. The explanation's inaccurate generalization is retained,
not edited into a passing answer. These answer-level issues are not yet
causally attributed to a LunaFlux execution defect or excluded as such by a
matched implementation control. Rewriting kernels to force these particular
answers would not be a correctness fix.

Observed maximum resident GPU memory was94676MiB (92.46GiB) on the extended
watch; minimum host MemAvailable was22430616KiB (21.39GiB). Both rank units
exit0, explicitly release CUDA, and have zero swap. The watcher samples every
five seconds, so this is the sampled peak, not a claim of continuous capture.
The runtime's100GiB allocation envelope and the16GiB host-floor checks remain
in force. No payload hashing or numerical-tolerance changes were performed.

Across the three runs,34 measured requests produced complete answers, plus
two completed warmups and one cancelled warmup. This proves neither arbitrary
prompt correctness nor concurrent HTTP service readiness. The canonical
batched frame/rank tests pass12/12; four focused transfer-double tests for
batched rank/control/TP retirement also pass. They are explicitly **not** GPU
model tests or network-client acceptance.

The endpoint remains offline. Real JSON/SSE agreement, concurrent mixed
prefill/decode, seeded GPU sampling, HTTP disconnect/cancellation, capacity
rejection/recovery and rank-loss handling are still unvalidated on the full
GLM service. The canonical worker bootstrap currently lists dense-model
recipes, not a GLM53 packed-model worker binding. Its GLM adapter must consume
the real model plan rather than masquerading as a Llama recipe. Together with
the missing GPU full-logit sampler and request-retirement wiring, this is
unfinished implementation—not a benchmark waiting for a switch to be flipped.

## Canonical HTTP integration, 18 October afternoon

The earlier missing implementation list above is now historical. Commits
`79c01cc2` and `fabc4701` introduce a dedicated GLM batched worker recipe,
canonical HTTP/scheduler binding, AOT full-logit GPU sampling, and explicit
two-rank request-retirement receipts. A terminal HTTP lane cannot be reused
until both GPU request owners acknowledge release. Cancellation preserves the
original device generation, including cancellation before the first frame.
This is not a serial CLI subprocess per request.

Sampling qualification root `/tmp/lunaflux-full-sampling-20261018-v4` contains
both-Spark correctness and memcheck/full-leak results for vocabulary513 and
the large synthetic155648 vocabulary. The loaded GLM checkpoint declares
154880; exact-checkpoint sampling qualification is recorded separately below.
Each normal probe covers24 cases twice. Tail
synccheck and init-kernel-only racecheck pass; **full-vocabulary racecheck was
not run**. Earlier capped racecheck OOM attempts are preserved, not converted
to passes. Sampling remains on GPU; only selected token/position results are
downloaded, not the vocabulary logits.

The first private HTTP integration attempts exposed two startup defects:

- Acceptance v1/v2: decoded-delta capacity4096 omitted the three-byte UTF-8
  carry required by a tokenizer admitting4096-byte pieces. The typed cause
  was `LunaOnlineAdmissionPreparationFailure(Invalid(OutputContract))`.
  `6a32cdcd` corrects the envelope to4099 and adds an exact-boundary test.
- Acceptance v3: the approved child launcher discarded the configured NCCL
  library/fabric environment. Rank1 observed rendezvous EOF. The fixed
  deployment library path had no NCCL installation, while the service's
  explicit runtime directory did. `4458d0d7` preserves the four explicit
  library/collective settings at launch. It also removes a rank0 startup
  print from the framed protocol channel. The environment change passes
  Linux ARM ASan/UBSan/leak, descriptor and signal probes on **both** Sparks
  at `/tmp/lunaflux-glm-worker-env-20261018-v1`, under2GiB/no-swap limits.

`52b80899` preserves preparation failure causes and applies the checkpoint
chat template to Responses string input. Required model EOS tokens remain
unioned with user stop tokens. The broader external HTTP acceptance runner is
`scripts/validate-glm-http-spark.mbtx`; its own parser regression is not a
full-model acceptance result.

Latest focused native tests pass264/264. Full native-suite execution is still
in progress at this record. All task commits are on main and pushed. No
checkpoint hashes or integrity scans were added. Direct small generated-source
comparison permits unchanged AOT reuse across these service-only changes.

The four-slot,64-row,1024-context plan estimates101957032556 bytes per Spark
(94.96GiB including reserves). This is a planning result, **not an observed
resident peak**. The two-rank acceptance watcher stops both owned units if GPU
residency exceeds100GiB, host MemAvailable drops below16GiB, or either rank
fails. All three failed private attempts are stopped. Public GLM remains
offline until complete HTTP behavior and answer-level acceptance pass.

### Startup acceptance v4 and lifecycle repair

The full native suite completed5252/5252, with the existing migration warning
exclusions unchanged. The subsequent worker/sampler changes also have focused
checks; this does not substitute for the two-GPU HTTP campaign.

Private v4 reached full checkpoint materialization but failed before HTTP
startup. Both owned units retired, unit swap stayed0, and no public endpoint
was opened. Source review found that the GLM adapter called `bind_sampling`
while the rank was still Prepared, whereas that operation requires Ready.
Executor binding must precede sampling attachment. The corrected order keeps
cleanup owned by the rank. A lifecycle regression rejects premature binding
and checks exactly-once sampling close.

The previous approved-child ABI closed stderr, so the original error was
lost behind a generic parent EOF and a peer cleanup Busy error. Stderr now
remains the parent's diagnostic descriptor, separate from framed descriptors
0/1. Startup diagnostics preserve the original preparation exception before
cleanup. Both-Spark ASan/UBSan/leak/descriptor/signal probes pass at
`/tmp/lunaflux-glm-worker-env-20261018-v2`. No request-time diagnostics, payload
hashes, or changed arithmetic were introduced.

Production readiness is now an explicit deployment boolean, false by default.
It does not bypass the pool's health/state readiness. The external acceptance
runner additionally covers capacity rejection/recovery, multi-turn messages,
stop strings and output-token budgets. These tests are not yet full-model
passes at this record.

### Eager-to-graph service transition (private acceptance v5)

The exact GLM vocabulary154880 sampler passes24 normal cases twice and
memcheck/full-leak checks on both Sparks at
`/tmp/lunaflux-full-sampling-glm-20261018-v5`. This is separate from the
earlier155648 synthetic coverage, not full-vocabulary racecheck.

Private HTTP v5 started successfully, but its first request received an empty
reply after4.304seconds and triggered worker recovery. Source isolation then
reproduced two failing pure regressions: both worker-process and service
telemetry owners wrongly treated selected graph path/shape as lifetime fixed.
The worker legitimately reports eager prefill(0hits,1miss), then captured
decode(1hit,1miss). Its cumulative telemetry was rejected as an invalid
completion. The corrected contract reports the latest step's path/shape and
monotonic cumulative counters across paths; sparse intervals may contain both.
Rollback, selected-counter progress and saturation behavior remain tested.
Rank-group telemetry now also permits path/shape changes between completions,
while requiring peer reports for the same completion to agree.

Observed v5 residency was95988MiB per GPU, with host available memory above
the16GiB reserve and unit swap0. Host swap already existed before this run.
The failed-peer parent drain exceeded its90second unit stop timeout; that
failure remains unresolved, not a clean retirement pass. The incomplete v6
trace run was deliberately stopped during loading because its syscall filter
missed send/receive transport; it is not an inference result.

The checkpoint chat template also requires `<think></think>` before historical
assistant content; the command now preserves that formatting. No arithmetic,
compiler comparison, hashing, or public route was added. Physical HTTP
acceptance must be rerun with these fixes before readiness promotion.

### Private HTTP v7: execution works; answer framing repaired next

Pinned48e7825e starts successfully and completes real two-Spark HTTP inference.
JSON/SSE return the same arithmetic result; Responses returns Paris. Four
concurrent requests return coherent photosynthesis answers. Three GPU-sampled
temperature0.7/top-k32/top-p0.9/seed12345 requests return the same sentence.
Six disconnect/reuse cycles complete, each followed by a correct2+2 response.
Evidence roots are `/tmp/lunaflux-glm-http-{smoke,concurrent,sampling,cancel}-20261018-v7`.
These are real canonical HTTP/TP2 runs, not standalone sampler results.

The raw responses expose a presentation defect: `323</think>323` and
`</think>Paris`. The prompt deliberately opens a reasoning section, but the
renderer previously treated every generated byte as answer content. The new
immutable output-framing policy is chosen by the model adapter and interpreted
only by the bounded OpenAI renderer. It recognizes the first closing delimiter
across token boundaries and preserves the following answer literally, for both
JSON and SSE. Generated tokens, stop behavior, usage and numerical IR do not
change. Tests cover split/overlapping delimiters, Unicode, final tails, reused
connections, and a token budget exhausted before any answer. The default plain
renderer remains unchanged. The smoke assertions now require exactly323/Paris,
so merely containing the expected answer no longer passes this presentation gate.

An additional replacement-worker telemetry regression accounts for a first
sample arriving after multiple unsampled steps: counters start at the child's
startup predecessor, not necessarily at global sequence1.

The full native suite run completed5259/5259; final framing changes additionally
pass129 focused tests and native check. No post-framing physical pass is claimed
yet. Capacity-only exports (no GPU execution) estimate106650380908 bytes per
rank at4096context/four slots;8192/16384 exceed100GiB and are rejected before
allocation. Host-reserve qualification is still required for any larger plan.

### Overload exposes missing recompute eviction

The first nine-client attempt exhausted SSH multiplexed sessions; it is not a
service capacity result. The second used a private local TCP forward and reached
the server: all clients lost their replies when the rank0 worker raised
`CapacityExceeded`. The watcher stopped both ranks. Memory remained within the
specified limits; this was not an OOM. Evidence:
`/tmp/lunaflux-glm-http-capacity-20261018-v7b`.

The scheduler preempts an aged waiter after two steps. Its recompute transition
released scheduler KV pages and active credit but returned only a boolean-like
`BuildPreempted`; the service discarded that result. The recurrent device frame
retained the victim's slot indefinitely. With four retained requests, the next
admitted replacement cannot reserve a slot. A focused frame regression
reproduces the failure and proves that eviction admits the waiter and permits
the original request to replay at position0 with the same generation and reset
state, without touching live neighbors.

The scheduler now emits an immutable request/generation eviction effect. The
service holds an explicit non-plan flight until the existing two-rank release
protocol acknowledges it; no successor plan is built in that interval. A
closed-child recovery consumes this device-only obligation without inventing a
submitted scheduler plan. Terminal device state is also retired before the next
plan, independent of slow HTTP terminal-event acknowledgment. Scheduling remains
backend-neutral; no model or CUDA detail enters the scheduler. The focused
native suite passes275/275. Physical overload retesting is still required.

The pinned ARM v9 build also caught a formatter-introduced range-pattern spelling
unsupported by its older compiler. The equivalent guarded byte comparison is
used instead; no toolchain migration or warning suppression was added.

### Recompute must make forward progress

A second, independent pure scheduler regression uses two finite requests,
one active slot and four output tokens each. Before the fix,80 scheduling
attempts emit only4 of the required8 tokens: each replay finishes, then is
immediately preempted before its next decode. The scheduler now retains one
preallocated decode-progress credit after completed replay. A request consumes
that credit on its next successful decode and is eligible for eviction again.
The policy changes no numerical computation, output-token budget or capacity;
it prevents fairness preemption from endlessly undoing useful replay work.
The previously failing test now finishes both requests and the213 affected
scheduler/worker tests pass. The preceding eviction revision's full native
suite passed5265/5265; this newer liveness change still needs physical overload
acceptance before publication.

### Filtered SSE tokens are not transport termination

Private v8 returns the exact JSON answer `323`, but the equivalent SSE closes
after its role event without `[DONE]`. The model and both ranks stay healthy.
The compatibility renderer correctly emits zero visible bytes for a reasoning
token; both TCP owners previously allowed such silent events only in buffered
mode and treated the SSE case as an invalid empty write. Both owners now
acknowledge silent nonterminal events without a socket write or connection cut.
Terminal events still require terminal framing. The failed raw SSE is retained
at `/tmp/lunaflux-glm-http-smoke-20261018-v8`; unit renderer success was not an
end-to-end transport pass. No public endpoint has been promoted.

The same v8 run correctly recalls `ORCHID-37` across a multi-turn prompt but
rejects a valid stop string before admission with503. The GLM startup envelope
reserved4096token-piece bytes plus3UTF-8 carry bytes, leaving no room for any
held stop prefix. The generic incremental decoder correctly rejects this
undersized envelope. GLM now derives delta capacity from4096piece bytes plus
the declared256stop bytes plus3carry bytes. This is a host output bound, not a
GPU or compiler-numerics change. The old failure and complete first broader
answer (truncated by512output tokens) remain under the v8 evidence roots.

### Private HTTP v9: framing/control passes, unexpected lifecycle exit

Pinned4580f324 uses freshly exported2048-context AOT on both Sparks. JSON and
SSE return exactly323, Responses returnsParis, multi-turn recall returns
ORCHID-37, the stop-string request publishes only`red `, and a four-token
request reports exactly4 completion tokens with finish_reason=length.
Evidence is under`/tmp/lunaflux-glm-http-{smoke,controls}-20261018-v9`.
Observed residency is97654/97669MiB with host reserve above16GiB and zero
owned-unit swap. The current full native suite passes5268/5268.

This is NOT acceptance: rank0 unexpectedly drains and exits0 after these
requests, before the subsequent capacity clients can connect. Their empty
replies at`/tmp/lunaflux-glm-http-capacity-20261018-v9` are not a valid overload
run. Rank1 remains resident. The preceding v8 stop also required the90-second
systemd kill timeout, and clean paired shutdown remains unqualified.

The pool previously discarded underlying coordinator/listener errors while
beginning drain. It now retains the first failure and its stage outside normal
token processing; the GLM command reports it after cleanup instead of calling
an error-triggered clean exit success. No speculative numerical fix, payload
hashing or public-route change was made. Physical reproduction with that
diagnostic build is still required.

### v10: reproduced overload credit and captured-collective teardown defects

Pinned dd38de21 repeats the smoke and all three control checks successfully.
The following nine-client overload attempt terminates rank0 with the retained
cause `take rejection: LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedLifecycle)`.
This narrows the earlier v9 unexplained exit to overload handling, not a
successful length-limited request or a memory failure. A taken rejection was
still reported as RejectionReady when a nonblocking error-response write had
not finished. The pool consequently attempted a second take. Both coordinator
drivers now report Waiting until the exact rejection credit is acknowledged.
The new regression fails before this change; the116 session/TCP tests pass
afterwards. Evidence roots retain the v10 smoke, controls and failed capacity
clients. The two-rank watcher now also retires a survivor after a clean peer exit.

Rank1 teardown again stalls. A scoped live GDB attachment to the exact owned
PID2917967 observes its main thread in `commDestroySync -> ncclCommPollCallbacks
-> ncclIntruQueueMpscDequeueAll(waitSome=true)`, waiting for captured graph
callbacks. The GLM command called communicator abort before the graph-owning
execution program could discard its graphs. The existing collective program
already has the correct discard-graphs/abort-communication/drain-stream order.
The stage now exposes execution-only abort through that owner before worker
DMA leases and model storage are released. No CUDA/NCCL ABI or kernel math
changes are required. Idle-captured and pending-execution teardown regressions
remain separate; real two-Spark shutdown must still confirm this repair.

Public GLM remains offline. This is a demonstrated lifecycle repair, not a
full-service acceptance pass or a throughput result. No payload hashes are used.

### v11: overload remains live; scheduler saturation was misclassified

Pinned70a5afe7 passes JSON/SSE/Responses smoke and remains healthy during the
nine-client overload. Four requests are rejected, but as503/service_error,
not429/capacity. Subsequent JSON/SSE/Responses and protocol checks pass. This
confirms the rejection-credit crash repair without calling overload accepted.
The client runner cancels its other clients on the first unexpected503; their
incomplete output is not a model correctness failure or a successful campaign.

The worker has eight publication lanes while the scheduler has four waiting
slots. A free worker lane therefore does not imply admission capacity. The
worker now translates scheduler waiting/request exhaustion into saturation,
releasing its temporary semantic retention. The session releases the consumed
claim and accepted-event reservation, and publishes a typed capacity rejection.
A regression with four lanes and only two waiting slots verifies saturation,
then successful retry after the scheduler activates the waiting requests.

Gateway compatibility now includes startup-owned `/v1/models` metadata and
the `/health` alias. The listing reports the admitted alias/context, not a
hard-coded model or an untested context promise. The embedded metrics endpoint
refreshes retained scheduler/graph counters on demand rather than exposing
stale zero gauges. None of these operations reads GPU data or hashes payloads.
The new changes still require their source-pinned two-Spark acceptance run.

The complete v11 HTTP answer campaign finished all12 prompts with natural stop,
using a1024-token ceiling. Evidence: `/tmp/lunaflux-glm-http-answers-20261018-v11`.
Reviewed answers include correct probability3/10, six chickens/four rabbits,
complete merge code and complexity, Chinese transaction examples, translation,
two-sentence summary, classifier limitations, concurrency and sort stability.
This is not twelve unqualified correctness passes: the JSON-only answer has
Markdown fences; the SQL answer's primary query is correct but its unsolicited
DENSE_RANK alternative includes customer_id in the ranking and thus does not
preserve equal-total ties as claimed. The floating-point answer also overstates
"never" using equality. These are retained answer-quality findings, not proven
execution defects. No response rewriting conceals them.

Paired v11 stop at18:40:54 CST completes on both ranks by18:40:57, exit0,
Result=success, no forced kill and zero owned-unit swap. Both GPU-owner queries
are empty afterwards. This physically confirms the captured-graph teardown
ordering repair; the preceding90-second shutdown hangs were real defects.
Latest local native suite:5273/5273 with the existing warning exclusions.

### v13: real queue pressure exposes a fixture-only fairness cadence

Pinned18fb1269 passes smoke, controls, model discovery and idle KV/request
retirement. The first observation client incorrectly expected HTTP200 for
private metrics; the existing control contract explicitly returns503 with
readiness=false while retaining its metrics body. The client now tests this
contract (including `/readyz`) rather than changing server readiness to pass.
The initial failed client output is retained.

Nine concurrent clients now produce four429 capacity responses, not503; four
requests run and one waits. The worker no longer exits. However, the GLM service
blueprint copied the scheduler fixture's two-step fairness threshold. With a
fifth request waiting it continually evicts/replays resident recurrent state.
At18:48:49 CST, prefix_tokens_computed reaches5330, graph misses321, and none
of these five512-token requests has completed. The live counters expose this
cost; the previously stale zero counters concealed it. This is a scheduling
policy error, not a numerical-kernel diagnosis.

The service now derives the age-only eviction window from the admitted context
(bounded by the existing scheduler validation ceiling). Existing bounded
waiting, page-pressure and deadline mechanisms remain in the generic scheduler;
no device/model numerical rule changes. A pure two-request regression proves
both policies finish, while the context-sized window avoids replay eviction.
The147 affected tests pass. The old-capacity experiment is deliberately stopped
after recording its counters; it is not a completed capacity pass. The new
policy requires a fresh physical replay of the unchanged clients.

### Active shutdown is distinct from the passing idle shutdown

Stopping v13 with five retained requests at18:49:21 releases both GPUs, but
rank0's HTTP parent remains in its reactor until the90-second stop timeout;
rank1 exits successfully. A scoped GDB capture of the owned rank0 parent shows
epoll/event-loop polling, not an NCCL wait. Three replacement attempts during
shutdown are also visible. This is a failed active-drain gate, not an idle-stop
regression or successful request retirement.

The repair propagates drain intent before retained tickets retire, preserves
terminal-recovery maintenance across repeated disconnect calls, consumes the
failure terminals after recovery is ready, and releases exact terminal lanes
after failed-child/device cleanup without sending retirement commands to the
dead child. Retired shared-stream epochs no longer leave HTTP/framed slots
waiting forever on an invalid route query. No GPU math or compiler comparison
policy changes. The191 affected native tests pass, including forbidden-restart
multi-lane retirement and early drain intent. The new active/failed-drain
process fixtures build locally but cannot start the native child on this Mac
(PhysicalUnavailable); this is not an end-to-end pass. The physical Spark
replay remains required before promotion.

The full local native suite subsequently passes5278/5278. Pinnedee79c390 ARM
build succeeds on178 under2GiB/no-swap. Both actual child-process fixtures,
`--openai-active-drain` and`--openai-failed-drain`, pass on that Spark, close
their HTTP pool with zero worker restarts and zero swap. Evidence is retained
in`/tmp/lunaflux-checkpoint-build-glm-http-20261018-v17/{active,failed}-drain.txt`.
These deterministic protocol children are not GPU-model acceptance; the next
GLM run must independently verify active shutdown with both TP ranks.

### v14: canonical HTTP, GPU sampling and retirement physical acceptance

Pinned `ee79c390` runs the actual GLM TP2 worker on both Sparks, with the
2048-context/64-prefill-row/four-resident-request AOT. This is a measured
deployment envelope, not qualification of arbitrary context or every vLLM API.
No numerical reference engine, payload hashing or response rewriting is in
the serving path. Public readiness remains false.

| External test | Observed result |
| --- | --- |
| Chat JSON / SSE / Responses | Exact arithmetic323, SSE terminal framing, Responses Paris |
| Generation controls | Multi-turn ORCHID-37, stop-string `red `, four-token length termination |
| Nine simultaneous capacity clients | Five complete, four HTTP429; subsequent request returns4 |
| Mixed JSON/SSE clients | Four coherent completed answers |
| Distinct prompt lengths and identifiers | All four exact request-specific identifiers; no observed cross-request routing |
| Seeded GPU sampling | Three temperature0.7/top-k32/top-p0.9/seed12345 responses agree |
| SSE disconnect / reuse | Six cancellations followed by successful reuse |
| Protocol / model discovery | Expected admission errors, Responses SSE and admitted model/context |
| Twelve broader answers | All natural stop,32–647 tokens each,4007 output tokens total |
| Active paired shutdown | Four active plus one queued; both ranks stop in about2seconds; GPUs released |

Before the intentional active stop, live counters show48admissions,
42completions,6cancellations, zero worker failures/restarts, zero active/queued
requests, and zero used KV pages. This includes request-retirement observation
after each test group, not only successful HTTP payloads. Buffered JSON early
disconnect detection is not established by the SSE cancellation test.

The context-sized fairness correction resolves the observed saturation replay
loop. The unchanged nine-client probe completes its first four512-token
requests in about73.2seconds and its queued request in112.9seconds, while
four clients receive429. During four-active/one-waiting execution, computed
prefix tokens stay284 and graph misses9 rather than continually replaying
prefixes. The v13 probe never completed, so no matched speedup factor is claimed.

The twelve sequential answers take321.034seconds in total, including HTTP and
prefill, or12.48output tokens/second. This is not isolated decode throughput or
a new comparison against vLLM. Manual review retains the JSON-only Markdown
failure, the incorrect unsolicited SQL tie-ranking alternative, and the
overbroad floating-point equality advice. The main SQL query, arithmetic,
translation, two-sentence summary and merge function are correct by inspection;
the generated Python has not been interpreter-tested. The concurrency example
also overstates that four tiny process-pool tasks must execute simultaneously.
Model-quality findings are not erased by service acceptance or attributed to
compiler arithmetic without a causal experiment.

Across249samples per Spark, maximum observed GPU residency is97654/97669MiB;
minimum host MemAvailable is18209088/19540112KiB (17.37/18.64GiB), above the
16GiB reserve. Both owned units report zero swap. Sampling is not a proof of
an unobserved instantaneous peak; the100GiB process cap and paired memory
watch remain in effect.

At19:13:52CST both units receive stop with four active and one waiting request.
Their actual journals record successful stop at19:13:54, with no timeout,
forced kill or replacement process. Both GPU-owner queries are empty. A
garbage-collected transient unit's default `ExecMainStatus=0` alone is not used
as proof; the journal and GPU release are checked separately. Interrupted
clients are retained as interrupted, not counted as successful answers. This
repairs the real v13 active-drain hang. Unexpected peer death is a separate
remaining physical gate, not implied by this cooperative stop.

Evidence roots are `/tmp/lunaflux-glm-http-{MODE}-20261018-v14`, with modes
`smoke`, `controls`, `capacity`, `concurrent`, `routing`, `sampling`, `cancel`,
`protocol`, `answers`, and `active-stop`. Group observations use
`/tmp/lunaflux-glm-http-{MODE}-20261018-v14-retired`; the separately added routing
observation uses `/tmp/lunaflux-glm-http-observation-routing-20261018-v14`.
Service/memory evidence
is `/tmp/lunaflux-glm-http-acceptance-20261018-v14`.

### Public cutover is not the private acceptance listener

Read-only inspection finds the LunaNexa route still names the old EXL3 runtime
and forwards to `glm53-exl3.lunanexa.svc.cluster.local:8899`; its endpoint is
absent. It does not yet own this NVFP4 TP2 process pair or supply its bearer
credential. Existing external-route metadata cannot truthfully be reused for
a different checkpoint/runtime. The current loopback listener, acceptance
credential and three-hour transient units are not a production deployment.
The user has been asked whether to expose a separately authenticated standalone
service first or wait for the managed LunaNexa integration. No public route has
been changed while that deployment choice is pending.

### v15: unexpected peer loss still blocks public release

The same `ee79c390` executable passes a fresh three-route smoke on both GPUs.
With four active requests and one queued, the test kills only the exact owned
rank-1 main process at 19:18:09 CST. The paired watcher detects the failure and
starts rank-0 stop at 19:18:10. Both GPU allocations are released, but the HTTP
parent does not finish retirement; systemd kills it at its 90-second timeout.
This is a failed physical gate, not a pass because GPU memory was reclaimed.

A scoped administrator GDB attachment to that exact parent, PID548705, shows
`epoll_pwait(timeout=0) -> EventLoop -> run_forever`; its other threads wait
for async-pool signals. No child or GPU owner remains. The single-request
failed-worker process fixture did not cover this multi-request failure timing.
The private run is retained at
`/tmp/lunaflux-glm-http-peer-failure-20261018-v15`; no successful completion
marker is written. The watcher reports the intentional peer failure rather
than hiding it as a successful service exit.

Cold teardown diagnostics now report retained request/event ownership once
after prolonged drain. They do not execute on normal token progress, inspect
GPU data, or include prompts/credentials. The 121 affected session/TCP tests
pass; the diagnostic is evidence collection, not a claimed repair. Public
cutover remains blocked by this failure independently of the route choice.

### Fault-fixture correction and repeated physical failure

The first two-client CPU fixture never reached its admission precondition.
`cmd/worker_echo_fail` was reading the startup-attestation frame as if it were
the first plan. It now consumes that frame before Ready and decodes a real
plan before injecting loss. The corrected two-admitted-request ARM run at
`/tmp/lunaflux-checkpoint-build-glm-http-20261018-v18/multi-drain-v2.txt`
exits zero in 30 ms, closes the pool and reports zero restarts. The failed
earlier fixture is retained; this CPU result does not cover the TP timing.

The v16 physical run passes smoke, then fails peer-loss cleanup with SIGABRT
instead of v15's timeout. Both GPU owners retire; the HTTP parent does not
exit successfully. A separate v17 attempt fails before inference because the
control listener address remains in TIME_WAIT after the failed pair. It is
not an additional numerical or GPU failure. The new listener binding enables
address reuse without port-sharing/load-balancing, and closes the first
listener if binding the second fails. Both cases have native socket tests.
The repeated peer-loss failure remains under investigation; no public route
or production-ready flag is changed.

The session regression now reproduces a distinct ordering defect: in
multi-request disconnect, a later lane's terminal event can hold the single
publication slot while the coordinator keeps retrying abort on an earlier
active lane. The regression fails before the repair. Disconnect now consumes
the pinned event first, assigns retirement to that event's route (not whichever
ticket was last polled), and distinguishes terminal events from accepted/token
events. All 123 session/TCP tests pass. This is not yet proof that the physical
peer-loss failure is resolved; the source-pinned replay remains required.

The v19 live snapshot confirms this ordering is the actual stalled state:
five tickets, maintenance kind 4, no outstanding semantic credit/ACK or
observation, recovery already complete, and `FailureUsageEvent` pinned to
lane 6 while lanes 3/4/5/7 remain active. No GPU owner survives. The scoped
parent GDB trace captures the later abort in the async event loop; it is not
a CUDA/NCCL wait. Evidence is under the v19 acceptance and peer-failure roots.
The complete local native suite after the route repair passes 5,283/5,283
with the existing migration-warning exclusions unchanged. No physical repair
pass is claimed before the new `64f144c8` build/replay.
