# Decoder numerical-program migration

The migration preserves the selected numerical program. It does not select
arithmetic by matching a reference, claim that agreement proves correctness, or
add payload hashing. Model adapters keep their model semantics; scalar rounding,
fold topology and state effects belong to compiler plans; CUDA spells the
private implementation. Quantized encodings and block matrix primitives retain
their existing typed precision laws rather than being flattened into scalar
BF16 arithmetic.

## Work and validation ledger

| Family | Expression/effect migration | Validation |
| --- | --- | --- |
| Text and learned heads | Previous committed work | 103 affected tests; bounded Spark head fixtures |
| Learned compressor | Previous committed work | Typed folds/effects and actual prepared schedule |
| Rotary product | Expression consumed by tensor/cache lowerings | Separate and fused programs remain distinct |
| HC nonlinear controls | Affine, sigmoid, root, softmax publication, collapse and both post-mix expressions consumed | Existing explicit control/tree laws retained; F32 post-mix fixture passes independent oracle and sanitizer checks; whole-control quality is separate |
| Dense SwiGLU | Explicit projection/activation/product rounding; effect-derived frame order | Native plan/source checks and complete GLM checkpoint execution; isolated GPU oracle coverage not added in this campaign |
| Recurrent delta | Explicit square/dot folds, normalization, decay, state update | Exclusive-slot ownership retained; three physical shapes pass oracle/state and sanitizer checks |
| Causal convolution | Ordered tap fold and activation expression | Raw-history publication retained; three physical shapes pass oracle/chunk/reset and sanitizer checks |
| Query/output/expert low-bit paths | Existing typed quantization/scaled-dot laws plus matrix publication, clamp, activation and route expressions | Physical low-bit primitives stay in private lowering |
| Sparse/window attention, indexing, routing/combine | Probability storage, normalization/division, scores, activation and combine expressions consumed | Sparse pool→index→attention frame order now derives from effect IR |

Estimated division is a distinct primitive from rounded division. The exact
rewrite gate rejects replacing one with the other. This retains the existing
GLM sparse probability law, rather than silently changing it during migration.
Rounding at BF16 product, route weighting, state and output boundaries is explicit.

### Architectural scope

This extends the existing precision IR, not a second model-specific compiler.
Model builders still describe DeepSeek/GLM topology and choose declared numeric
contracts. Pure expression/fold construction describes arithmetic; the existing
effect IR orders materialized producer/consumer stages. Private CUDA lowering
renders those primitives and binds them to the existing AOT frames. Quantized
matrix fragments, physical reduction trees, cache ownership and recurrent-slot
lifetimes retain their existing typed plans rather than being flattened into a
scalar expression or an untyped global configuration.

An explicit alternative (FMA contraction, reciprocal square root, approximate
division, a different reduction tree or a different rounding boundary) remains
a different numerical program. A comparison score cannot turn it into an exact
rewrite. GPU fixture comparisons, retired-logit analysis, independent engines,
and held-out quality belong to offline evaluation. There is no reference-engine
import or empirical-correctness constructor in the compiler.

The ledger describes the selected decoder families, not a claim that every
historical compatibility source generator has been deleted, every CUDA statement
has become scalar IR, or every nonlinear operator has a formal error proof.
The migration itself preserves numerical policy and does not claim a speedup.
Matrix fragments and reduction trees remain typed operations, not scalar folds.

Working-tree warning-denied native check passed. Initial full suite: 5,088/5,099;
the failures were source-spelling/count and source-digest snapshots. They were
replaced by explicit arithmetic/source assertions, not numerical tolerance
changes. Full rerun: **5,101/5,101**. This is the working-tree suite, not a claim
that unrelated toolchain/TLS edits were committed. The later observer check also
passes. `moon info` completed with 3,656 existing migration warnings and zero
errors. Compiler dependency closure: 11 packages, no evaluation dependency.

## Physical and integration checks

Migration checkpoint `fc3d4e1c` is pushed. Clean source-pinned native ARM builds
for both checkpoint executables passed, followed by fresh two-stage CUDA AOT
export/compilation. No prior cubins are used as evidence for changed lowering.

`/tmp/lunaflux-decoder-numeric-20261012-v2` exercises freshly generated recurrent,
causal-convolution and F32 hyper-connection kernels on Spark179, serially under
2 GiB/no-swap cgroups. Recurrent small/medium/64-head×128-key×128-value shapes
preserve output/state bits against the retained serial implementation, with a
separate F64 oracle check. Convolution small/tail/8192-channel shapes preserve
history and prefill-versus-decode chunking, with independent numerical checks.
The F32 hyper fixture checks control order, transposed residual orientation,
inactive rows and repeated execution. All seven cases pass memcheck/full leak,
racecheck and synccheck. These are fixtures, not full-model quality results.
The initial v1 export was stopped while waiting on the local full-suite build
lock; no GPU work had started. v2 uses an isolated build directory.

GLM now has an explicit `egress-logits` diagnostic: borrow the existing head
allocation, prepare a bounded transfer before execution and copy once after
EOF. Normal generation adds no observer, transfer or per-token comparison.
The resident runner's `logits` mode captures request 1's final selected row,
predicting generated token index 7 after consuming seven generated tokens.
Offline analysis must use that literal prefix, not call it initial-prefill
logits. Access before preparation is rejected and covered by the stage test.
The observer native check, stage test and eight runner tests passed; targeted
interface generation retains 936 existing migration warnings, zero errors.

## Whole-model execution, not automatic quality acceptance

DeepSeek root `/tmp/lunaflux-dspark-run-numerical-ir-text-20261012-v1` uses
`fc3d4e1c`, fresh AOT, all 43 layers across 27/16-layer stages, six literal
one-row prefill steps, and eight ordinary-target generated tokens. Both ranks
exit zero and explicitly close CUDA; peak cgroup memory is 39,817,031,680 and
47,402,897,408 bytes, with zero cgroup swap. Reported generation duration is
31,526 ms, excluding startup. This reference schedule is not a tuned serving
benchmark. The mode name `generate-reference-dspark` means the ordinary native
target without speculation, **not an independent reference implementation**.

A separate fresh-request one-token capture returns the same first token
(528), and its raw selected vocabulary row contains 129,280 finite F32 scores;
the argmax reproduces that emitted token. Its duration is 14,565 ms. This checks
the capture's row/sampler wiring, not correctness by self-agreement.

The literal raw input `The capital of France is\n` produced an HTML-like
continuation, not an acceptable answer. Preserve it as a negative content check.
The installed original `generate.py` uses `encode_messages(..., thinking_mode=
"chat")`, adding BOS/User/Assistant/`</think>`; our raw `text-frames` API explicitly
does not apply a template. A subsequent single-user smoke uses that documented
format. GLM's installed `chat_template.jinja` similarly supplies gMASK/sop,
low-effort system text, user/assistant markers and `<think>`. Template expansion
is offline model input preparation, never a numerical compiler pass. These
single-question smokes cannot establish held-out quality or cross-engine parity.

In GLM's first templated request, the first three generated IDs are
`154842,59190,154827`: `</think>Paris<|user|>`. The installed
`generation_config.json` lists EOS IDs `[154820,154827,154829]`; therefore the
answer before the observed EOS is Paris. The bounded diagnostic explicitly
uses `stop_token=-1` and continues to eight tokens for the retired-head capture;
the subsequent role text is post-EOS continuation, not part of the answer.
Keep all eight literal IDs for position-correct offline analysis rather than
silently truncating the numerical capture to the displayed answer.

### GLM complete checkpoint result

`/tmp/lunaflux-glm-sequence-numerical-ir-chat-20261012-v1` uses source
`4cc199fb`, fresh AOT, all 45 GLM-5.3-Flash decoder layers split 25/20, and two
requests on the same resident pair. Request 0 has 22 input tokens; request 1
has 21. Each generates eight tokens with stopping intentionally disabled.
Request 0 retires at epoch 30 and request 1 at epoch 58; both ranks exit zero
and close their CUDA contexts. Peak cgroup memory is 49,161,658,368 bytes on
178 and 43,547,713,536 bytes on 179, with zero cgroup swap. This does not claim
that the machines' pre-existing host swap was empty.

The second output is `</think>One plus one equals **2**`, with IDs
`154842,3966,5519,825,16801,3070,17,334`. Both simple answers are correct. The
second response reaches the eight-token diagnostic limit rather than EOS;
no claim is made about later continuation. This covers the complete text
decoder of the Flash checkpoint, not GLM's larger Full checkpoint or vision
encoder, nor a held-out quality suite.

The retired capture has 154,880 finite raw F32 logits (BF16-rounded by GLM's
declared head program). Row 0's argmax is 334, matching the final emitted token;
top score is 29.125 and runner-up is 18.5. It predicts generated index 7 after
consuming the 21 input tokens and seven generated tokens: absolute prediction
position 27. All eight generated IDs, the literal input, and `logits-scope.txt`
are retained for an offline teacher-forced comparison. This is not a comparison
against an independent model implementation.

Reported request durations are 671,411 ms and 643,002 ms, excluding initial
checkpoint loading. This is a slow serial correctness-oriented execution path,
not a tuned serving benchmark or a performance gain from this refactor.

### DeepSeek complete checkpoint chat result

`/tmp/lunaflux-dspark-run-numerical-ir-chat-20261012-v2` uses `fc3d4e1c`, the
fresh AOT compiled from that same migration source, all 43 DeepSeek-V4-Flash-
DSpark decoder layers split 27/16, physical capacity six rows and logical
one-row prefill. The literal chat prompt contains 14 tokens, including the
checkpoint's BOS/User/Assistant/closing-think markers. The ordinary native
target produces IDs `51119,16,1`: `Paris.<｜end▁of▁sentence｜>`, and stops at
the configured EOS rather than continuing to the eight-token ceiling.

Both ranks exit zero and explicitly close CUDA. Peak cgroup memory is
50,781,921,280 bytes on 178 and 45,287,469,056 bytes on 179; both report zero
cgroup swap. Reported generation duration is 38,807 ms, excluding initial
checkpoint loading. Both GPUs are released before any later workload.
The three-token result is not a throughput benchmark, DSpark speculative
acceptance test, or an independent original/vLLM/SGLang comparison.

The raw-prompt and templated results are retained together. The simple chat
answer is correct, showing that the earlier HTML-like raw continuation cannot
by itself establish a decoder arithmetic defect. It does not resolve the
previous long-stream numerical differences or establish broad model quality.
The initial chat preparation v1 failed before execution because its local
literal input file had not yet been created; v2 preserves the corrected
preparation and completed run without overwriting v1.

Comparison and teacher-forced evaluation stay in offline runners. Full-model
validation must include final logits and token/position semantics, not just
hidden-state similarity. These smokes do not establish held-out model quality,
cross-engine parity, or the absence of every numerical defect. Historical
compatibility generators and broader numerical proof/coverage remain separate
from the selected decoder-family migration described here.

The user explicitly requested stopping MiniMax to use both Sparks. Its exact
user unit was stopped; systemd reports timeout during termination, MainPID=0,
and no remaining GPU process. This is an interrupted experiment, not completed
media validation. Logs and partial outputs were preserved. Both Sparks are now
available for serial, bounded two-rank validation.

The GLM runner now exports and compiles fresh AOT from the source-pinned binary.
Reusing the old AOT would not exercise migrated kernels. Existing declared shard
labels are copied as metadata; no payload hashes are computed.
