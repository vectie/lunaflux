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
