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
