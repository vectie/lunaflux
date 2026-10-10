# DeepSeek literal-text frontend — 2026-10-10

The original installed DeepSeek-V4 Flash DSpark tokenizer now loads and executes
through the shared native checkpoint text frontend. This is text-input
functionality, not an independent reference comparison or GPU throughput result.

## Implemented semantics

The empty normalization sequence is identity. Three isolated Split stages run
in declared order: Unicode numeric triplets, CJK/kana ranges, then the declared
word/mark/symbol/whitespace expression. Later stages and BPE cannot rejoin an
earlier isolated fragment. The implementation streams scalar boundaries without
heap allocation in the tokenizer worker's progress path. Unicode punctuation
and symbol classification comes from Unicode 17.0.0 general categories.

The actual JSON contains 46 ordinary added tokens with `normalized=true`.
Identity normalization makes their raw and normalized matching domains equal;
both use literal matching. Unsupported stripping and single-word matching are
not silently ignored. Ordinary added tokens are not classified as special and
are preserved when decoding with special-token skipping.

`text-frames MODEL_ROOT ROWS HISTORY TOKENIZER_LABEL INPUT_UTF8_FILE NEW_OUTDIR`
adds no BOS/EOS or chat template. It emits prompt execution frames, token IDs
and decoded literal text, retaining one context position for generation.
The supplied tokenizer label is not a checksum. This mode opens no CUDA context,
reads no weight inventory or shards and performs no payload authentication.

## Actual-checkpoint text checks

The original 6,367,146-byte tokenizer was used, with its original model config.
All five CLI cases exited 0 with empty stderr and exact decoded text:

| Case | Token IDs |
| --- | --- |
| `Hello` | `19923` |
| `Hello, 世界! مرحبا 123456 e◌́ 🐈\n` | `19923,14,223,3427,3,112125,53067,223,6895,18009,312,17793,7351,241,233,201` |
| `abc中文def カ・ナ　 abc €abc` | `32372,21134,3465,223,15961,4825,27071,18524,71490,21820,32372` |
| Explicit BOS + `Hello` + EOS | `0,19923,1` |
| Explicit user/assistant markers | `128803,19923,128804` |

Evidence: `/private/tmp/lunaflux-deepseek-text-live-20261010-v3`.
Earlier failed v1 evidence remains preserved. The compatibility fix has no model
name branch in the tokenizer worker; JSON selects an execution-semantics profile.
Actual IDs are reported, not claimed to match an independent tokenizer yet.

Native tokenizer/frontend, prompt-frame and three checkpoint command tests passed
93/93. The one-off original-file parser diagnostic was removed after passing;
the committed regression suite has no dependency on a local checkpoint path.

## Prepared two-Spark literal replay

Commit `fca565544b345d1c8c9f2015f7214bbb66061037` built on .178 with
236 successful ARM native release tasks, under the existing 2-GiB/no-swap build
unit. CPU-only preparation on that binary used the original installed tokenizer
and emitted one six-row-envelope frame for the literal user/assistant-marker
input (`128803,19923,128804`), with exact decoded input and empty stderr.

Local preparation: `/tmp/lunaflux-dspark-run-text-20261010-v1`.
The two hosts use the corresponding new
`/tmp/lunaflux-dspark-real-text-20261010-v1` directories. Unchanged device code
reuses the already-compiled six-row/context-256 AOT modules from
`/tmp/lunaflux-dspark-real-verify-20261010-v1`; this is not a claim that those
modules were freshly compiled from the frontend commit. This first prepared
replay was not executed; the reference-rendered second replay below was chosen
after the actual-caption MiniMax joint request released .178.
The runner consumes the prepared prompt directory and frame count
instead of hard-coding one BOS frame, and checks both GPUs before starting
either owner.

A second replay preparation uses the exact non-thinking single-user prompt
string from the checked-out vLLM regression
`tests/tokenizers_/test_deepseek_v4.py`:
`<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>`.
The original tokenizer emits `0,128803,19923,128804,128822`, one frame, with
empty stderr and exact decoded text. This checks a reference-rendered string,
not independent tokenizer-ID or model-output parity. It is prepared at
`/tmp/lunaflux-dspark-run-text-20261010-v2` and the corresponding remote
`/tmp/lunaflux-dspark-real-text-20261010-v2` roots. The active execution sequence
waited for the existing MiniMax unit's terminal success and GPU release before
starting speculative then ordinary target generation on this identical prompt.
No existing model owner is restarted. Runner prompt/output regressions and the
27-script no-hashing gate pass; the token-step scan/copy/readback gate also passes.

## Completed literal-prompt two-rank execution

Both modes completed successfully, with sixteen identical generated tokens:

```text
19923,3,1730,588,342,1694,440,4316,33,1,380,9259,223,8842,2878,1863
```

The original tokenizer's CPU-only output decoder produces, preserving special
tokens:

```text
Hello! How can I help you today?<｜end▁of▁sentence｜> PPL 模型微调
```

Token 1 is the actual checkpoint's EOS. This deliberately fixed-length diagnostic
sets stop token to `-1`: the final six tokens are generation *after EOS*, not part
of a normal assistant answer. Matching them checks internal speculative/ordinary
state behavior, not conversational quality. A stop-aware request must terminate
at EOS; it must not silently present these extra tokens as a response.

| Mode | Diagnostic generation time | Sixteen output tokens / second | Peak .178 / .179 bytes |
| --- | ---: | ---: | --- |
| DSpark target verification | 31.364 s | 0.510 | 48,947,175,424 / 33,515,851,776 |
| Ordinary greedy target | 39.206 s | 0.408 | 51,757,350,912 / 34,091,294,720 |

Speculative elapsed time is 20.0% lower for this one short literal-prompt sample;
the older BOS-only sample was 30.3% slower. These are not repeated or
counterbalanced trials, not independent vLLM/SGLang comparisons and not general
DSpark acceleration claims. The timer excludes model preparation/loading but
includes prompt frame execution and diagnostic IO as in the earlier report.
Speculative output reports six verified blocks, 24 submitted input rows,
15 committed input rows and four accepted-prefix replays. Component-time
attribution remains unmeasured.

All four user units report success, exit status zero, retained exited state,
and zero measured swap peak. Both post-mode GPU process receipts are empty;
the speculative request exited before ordinary model loading began. Config,
checkpoint, binary and CUBIN payloads were not hashed. The diagnostic cgroup
peaks do not replace the separately bounded driver/unified-memory plans.

Raw mode journals, terminals, prompt and both token vectors remain under
`/tmp/lunaflux-dspark-run-text-20261010-v2`. Offline decoded output is saved at
`/private/tmp/lunaflux-deepseek-decoded-output-20261011-v1`; its decoder change
does not alter the committed `fca56554` GPU run binary or AOT modules.

The existing NVIDIA vLLM container on .179 is
`0.13.0+faa43dbf.nv26.01`, and lacks the newer `vllm launch render` component.
A GPU-less, network-disabled 4-GiB CLI probe failed device inference before
model loading. It did not produce an independent tokenizer or model reference.
Full independent numerical parity, longer/wrapped contexts, cancellation,
multi-request serving and matched external-framework benchmarks remain open.

## Completed stop-aware follow-up — 2026-10-11

`prepare-deepseek-text-replay.mbtx` optionally accepts a maximum generated-token
count and stop token. The runner retains `16/-1` for old fixed-length diagnostics,
but now verifies the requested bound, the terminal `Length`/`StopToken` reason,
and that no published token follows the first configured stop token. The actual
number of committed speculative input rows must equal output count minus the
initial seed, rather than hard-coding fifteen rows. Regression cases cover both
old diagnostics and shorter stop-terminated output.

The unchanged actual chat prompt, binary and device modules are prepared anew
at `/tmp/lunaflux-dspark-run-text-eos-20261011-v1` with a 64-token maximum and
the original checkpoint's `eos_token_id=1`. Preparation opens no GPU and does
not overwrite the completed fixed-length run. Both physical modes now complete
with the exact ten-token prefix ending at EOS:

```text
19923,3,1730,588,342,1694,440,4316,33,1
```

The original tokenizer decodes both outputs identically:
`Hello! How can I help you today?<｜end▁of▁sentence｜>`.
Neither path publishes a token after EOS; both report `generated=10
finish=StopToken`, not a 64-token `Length` completion.

| Mode | Diagnostic generation time | Ten output tokens / second | Peak .178 / .179 bytes |
| --- | ---: | ---: | --- |
| DSpark target verification | 12.502 s | 0.800 | 58,502,074,368 / 35,242,106,880 |
| Ordinary greedy target | 24.658 s | 0.406 | 58,457,829,376 / 36,137,172,992 |

Speculative execution uses two verified blocks, nine committed input rows,
twelve submitted rows and one accepted-prefix replay. Its completion time is
49.3% lower in this single EOS-limited sample. The different fixed-length/BOS
results above remain valid; this is not a repeated performance estimate or an
independent framework comparison. The ten-token numerator includes EOS.

All four user units report successful zero-status termination and zero measured
swap peaks. Both modes release both GPU owners, observed by empty post-mode
compute-process queries; ordinary loading starts only after speculative release.
Both journals report `payload_hashing=false`. The decoder is CPU-only and does
not change the `fca56554` GPU binary or reused AOT modules. Its exact decoded
outputs and empty stderr are retained at
`/private/tmp/lunaflux-deepseek-decoded-eos-20261011-v1`.

This closes the actual prompt's EOS-termination and internal differential check,
not independent model parity, wrapped-context correctness, cancellation,
multi-request serving, or production performance.

## Original sliding-window wrap — 2026-10-11

A fresh literal prompt repeats `Hello ` 136 times, asks `Please answer hello.`
and uses the same original non-thinking chat presentation. The native original
tokenizer emits 144 input tokens, divided into 24 six-row prefill frames. This
crosses the checkpoint's 128-token sliding window while staying within the
unchanged 256-position AOT/context envelope. The two modes use the same
`fca56554` binary, original weights and AOT modules, with maximum output 16
and EOS 1. No new compilation, payload hashing or concurrent GPU workload is
introduced by this case.

Both modes produce exactly these ten IDs and terminate `StopToken`:

```text
19923,3,1730,588,342,8233,440,4316,33,1
```

The original tokenizer decodes both into the exact 61-byte string
`Hello! How can I assist you today?<｜end▁of▁sentence｜>`; decoder stderr is
empty. The speculative route executes three verification blocks, nine committed
input rows, eighteen submitted rows and three accepted-prefix replays.

| Mode | Generation including 24-frame prefill | Peak .178 / .179 bytes |
| --- | ---: | --- |
| DSpark target verification | 87.772 s | 50,100,387,840 / 26,728,951,808 |
| Ordinary greedy target | 92.038 s | 49,474,068,480 / 27,703,451,648 |

The speculative completion is only 4.6% shorter in this single longer-prompt
case, versus 49.3% in the short EOS-limited case above. Prefill and replay are
part of the measured generation interval; these totals are not isolated decode
throughput or a general speedup. All four user units terminate successfully
with status zero and measured swap peak zero. All four post-mode GPU-owner
query files are empty, and the ordinary mode starts only after speculative
release. The 96-GiB/no-swap external limits remain in force on both nodes.

Evidence: `/tmp/lunaflux-dspark-run-text-wrap-20261011-v1`; original-tokenizer
decoding: `/private/tmp/lunaflux-deepseek-decoded-wrap-20261011-v1`.
This proves internal speculative/ordinary equality across this actual sliding
window transition, not independent upstream model parity, maximum supported
context, cancellation, multi-request serving or external-framework performance.

## Independent original-tokenizer corpus — 2026-10-11

The native frontend now matches an independent vLLM 0.26.0 HF renderer on
all twelve actual-checkpoint corpus cases, for both exact token IDs and exact
decoded text. The renderer uses the installed original model/tokenizer files,
`--tokenizer-mode hf`, and `add_special_tokens=false`; neither side inserts a
chat template, BOS or EOS. The corpus includes the actual five-token chat
prompt and 144-token wrapped-window prompt above, English, CJK/kana/Hangul,
composed/decomposed Unicode, mixed numeric scripts, whitespace/CRLF, emoji,
original added/special tokens, source code, multilingual text and punctuation.

Reference token counts in that order are
`5,144,20,24,15,41,11,31,7,42,22,33`. All twelve ID comparisons and all twelve
decode comparisons pass. This closes independent frontend parity for this
corpus, not arbitrary-input completeness or independent real-weight logits.

The existing H3 ARM image supplies vLLM 0.26.0, Transformers 5.14.1 and
tokenizers 0.22.2. `vllm launch render` loads no model weights. Its GPU exposure
only supplies platform identity; before/after compute-owner queries are empty.
The isolated loopback-only container has a 4-GiB memory ceiling and no swap;
measured cgroup memory peak is 2,332,917,760 bytes and swap peak is zero.
It stops successfully after collection. No payload hashing or production
service change is involved, and these are not inference-throughput results.

The first renderer failed when a CGo dependency exhausted its 128-task limit.
Its terminal logs remain preserved. The successful retry limits CPU affinity
to four cores and library thread pools to two, with a 512-task ceiling; it is
not a model/code compatibility workaround.

Evidence: `/private/tmp/lunaflux-reference-tokenizer-corpus-20261011-v1`;
startup/failure capture: `/private/tmp/lunaflux-reference-tokenizer-20261011-v2`.
Requests, native IDs/decoded text, raw reference responses, terminal state and
memory readings are retained separately without overwriting earlier attempts.

## Real generation cancellation — 2026-10-11

The `9dbb0b6b` checkpoint executable adds terminal-only explicit context-release
telemetry and an unbuffered, one-time generation-start marker. It does not change
CUDA arithmetic, the six-row/context-256 AOT or token-step validation. The test
uses the actual original five-token chat prompt, maximum output 251 and EOS
disabled, so the complete requested capacity is exactly 256 and the request
cannot finish normally before interruption. The generation marker follows a
real GPU prefill whose first sampled token is 19923.

`scripts/capture-dspark-cancellation.mbtx` reuses the normal two-rank launcher
with fresh output directories and 96-GiB/no-swap user units. After generation
starts, it waits one second and sends exactly one SIGINT to the selected rank.
No SIGKILL escalation, payload checksum or additional GPU workload is used.

Both directions pass. In each, the signalled rank records successful explicit
context close; its peer sees `RemoteChannelError.Disconnected` and also closes
its context. CUDA context close refuses outstanding child resources, so these
markers are stronger than merely observing vanished processes. All four
post-test GPU compute-owner queries are empty. Neither case publishes a normal
generation receipt; additional read-only checks find no `output-run/tokens.txt`.

| SIGINT target | .178 cgroup peak bytes | .179 cgroup peak bytes | Both swap peaks | Both explicit contexts closed |
| --- | ---: | ---: | ---: | --- |
| Ingress | 48,170,549,248 | 25,569,525,760 | 0 | Yes |
| Egress | 48,673,447,936 | 43,985,432,576 | 0 | Yes |

These are cgroup observations, not total physical device-memory accounting or
performance measurements. The ordinary-success launcher returns nonzero as
expected and retains both terminal journals; its failed result is not rewritten
as a successful generation. The cancellation capture reports the narrower
cleanup outcome separately.

The first ingress attempt remains a diagnosed capture failure under
`/tmp/lunaflux-dspark-run-cancel-ingress-20261011-v1`. Both contexts actually
closed, but the harness incorrectly required a userspace exit code from the
signalled process. MoonBit async unwinds owners, flushes stdio and then re-raises
the original signal; its C runtime explicitly calls `raise(sig)` after
`fflush(0)`. Therefore successful cooperative SIGINT cancellation appears as
`ExecMainCode=2`, `ExecMainStatus=2`, while the disconnected peer exits with
code/status 1. The corrected harness requires the role-specific close marker
and correct expected exit for each side, and still rejects SIGKILL, OOM, swap,
missing/running units and successful generation publication. It does not accept
a SIGINT result alone as cleanup evidence or rewrite the first failed capture.

The passed ingress rerun is retained under
`/tmp/lunaflux-dspark-run-cancel-ingress-20261011-v2/cancellation`; the egress
case is under `/tmp/lunaflux-dspark-run-cancel-egress-20261011-v1/cancellation`.
Separate user units and remote roots use the same suffixes; no evidence is
overwritten. The runtime binary is built from `9dbb0b6b`; the corrected capture
classifier is committed as `673e14fd`. Subsequent capture code also checks
tokens-file absence directly after both contexts retire. For the two executed
captures, this check was independently run after collection and returned 0.

Affected native checks and formatting pass with existing migration warning
exclusions `-20-25-29-35-79-92`. The five-package native matrix passes 25/25;
the complete packed-execution fake-device regression package passes 114/114.
Capture self-tests reject SIGKILL, missing close markers, successful output,
swap, OOM and missing/running units. The 28-script no-hashing and token-step
scan/copy/readback checks pass. `moon info` completes with no errors and the
existing toolchain-migration warnings. No native ABI or kernel changed, so
these are not new arithmetic sanitizer or performance results.

This is process-level cancellation of this attached single-request DSpark
diagnostic, not reusable request cancellation with retained weights. It does
not close bounded multi-request serving, other model cancellation paths,
independent model numerics, or performance comparison.

## Resident ordinary request sequence — 2026-10-11

The corrected `6894b72e22308009ea4a7ba0340bf1015163ead0` ARM release binary
processes two different original-text requests in one pair of processes, without
reloading weights, modules or connections. CPU preparation retains both prompts
before device startup under one aggregate byte budget. The original five-token
chat prompt is followed by the 144-token sliding-window prompt; both request
limits are maximum 64 new tokens and EOS 1. Six-row/context-256 AOT kernels are
unchanged and reused; this is a session-lifetime test, not new kernel validation.

| Request | Input tokens | Output tokens including EOS | Request lifetime incl. prefill | Next transport epoch |
| --- | ---: | ---: | ---: | ---: |
| 0 | 5 | 10 | 24,587 ms | 11 |
| 1 | 144 | 10 | 91,800 ms | 44 |

Both downloaded outputs are byte-identical to the corresponding earlier
isolated ordinary-target token files. They are not compared to an independent
upstream model in this test. The timing excludes initial model loading and is
one diagnostic sample, not a repeatable throughput or speedup claim.

Both user units exit 0 (`active/exited` because `RemainAfterExit=yes`), with
explicit ingress/egress context-close markers and empty terminal GPU owner
queries. Cgroup memory peaks are 48,372,363,264 bytes on .178 and 31,970,054,144
on .179; both unit swap peaks are zero. These cgroup values do not include every
physical CUDA allocation on unified memory. Process limits remain 96 GiB/no-swap
and both machines had over 125 GB available before startup.

Raw local captures and downloaded outputs:
`/tmp/lunaflux-dspark-run-sequence-20261011-v1`.
Remote input/output roots: `/tmp/lunaflux-dspark-real-sequence-20261011-v1`.
User units: `lunaflux-dspark-sequence-20261011-v1-sequence-ingress.service` and
`lunaflux-dspark-sequence-20261011-v1-sequence-egress.service`.

The next speculative-session source change moves verifier ownership from a
single generation helper into the outer rank session. It preserves explicit
final drain/release while individual requests release only their KV/frontiers.
The new fake-device/socket regression exercises a distinct second prefill and
rejected-prefix replay using the same resource owners. This source work does
not itself prove physical speculative queue reuse, retained-weight cancellation,
concurrent service, independent numerics or full-model performance parity.

## Resident speculative request sequence — 2026-10-11

`09cb4e77c334edb7258c7ad3d6e2b6a340156f0a` subsequently passes the same two
literal requests using `generate-sequence-dspark`. Its `v4` exact-source ARM
release build succeeds in 238 tasks, with a 649.8 MB CPU-build peak and zero
swap. The preceding `f3254a29`/`v3` build is preserved but never physically run:
source review found that verification request release left the ordinary prefill
coordinator's publication frontier live. The corrected source locally retires
that frontier after the shared rank receipt, without another network release.
The strengthened regression also stages a third distinct zero-position prefill.

| Request | Verified blocks | Submitted input rows | Committed input rows | Prefix replays | Lifetime incl. prefill | Next epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0: five input tokens | 2 | 12 | 9 | 1 | 11,585 ms | 5 |
| 1: 144 input tokens | 3 | 18 | 9 | 3 | 86,793 ms | 35 |

Both requests emit ten tokens including EOS 1. Downloaded token files are
byte-identical to the respective ordinary resident outputs and, transitively,
their isolated ordinary-target runs. This proves reuse, reset and actual
verification/replay on this checkpoint; it does not prove independent upstream
model equivalence. Lifetimes exclude initial loading and are single diagnostic
observations, not repeatable throughput or vLLM/SGLang comparisons.

Both user units exit 0 with explicit ingress/egress context-close markers;
terminal GPU owner queries are empty. Cgroup peaks are 53,723,009,024 bytes on
.178 and 24,815,456,256 on .179; both unit swap peaks are zero. During loading,
.178 reported 86,487 MiB (84.46 GiB) of GPU allocation and 34 GB of system
available memory, while .179 had approximately 54 GB available. Cgroup peaks
are not complete physical CUDA-memory accounting. Both runs use the stated
96-GiB/no-swap process limits and no concurrent model workloads.

Raw captures: `/tmp/lunaflux-dspark-run-sequence-spec-20261011-v1`.
Remote roots: `/tmp/lunaflux-dspark-real-sequence-spec-20261011-v1`.
Units: `lunaflux-dspark-sequence-spec-20261011-v1-sequence-spec-ingress.service`
and `lunaflux-dspark-sequence-spec-20261011-v1-sequence-spec-egress.service`.
New output paths preserve earlier captures; no payload hashes were computed.

The affected native matrix passes 120/120 after removing unused fixture payload
hashing; the focused verification/CLI matrix passes 11/11 and standalone runner
tests pass 3/3. Native checks/formatting, generated API review, and no-hashing/
token-step developer gates pass with existing migration warning settings. The
only added public API is local `retire_publication()`; no CUDA arithmetic or
native ABI changes were required. Independent full-model references, GPU
maximum-capacity execution, retained-weight cancellation, concurrent serving
and matched repeatable external-framework performance remain open.

## Exact 256-token request envelope — 2026-10-11

The same `09cb4e77` binary physically completes the 144-token literal prompt
with exactly 112 generated tokens in both speculative and ordinary modes.
Downloaded output files are identical across all 112 IDs. EOS stopping is
deliberately disabled, so output after EOS is a capacity diagnostic, not a
normal assistant answer or independent model reference.

Speculative execution uses 29 verification blocks, 173 submitted continuation
rows, 111 committed input rows and 17 rejected-prefix replays. Its final
transport epoch is 71; ordinary execution retires at 136. Both finish `Length`.
Prompt plus emitted output equals 256, but the last emitted token does not
require a further KV input: committed history ends at 255. The corresponding
software regression checks final-block clipping and rejects +1 output before
device preparation. This does not validate the checkpoint's model-maximum
context or arbitrary long-context numerics.

Request lifetimes including prefill, excluding initial model loading, are
207,224 ms speculative and 350,885 ms ordinary. These are single diagnostic
samples, not repeatable throughput or a vLLM/SGLang comparison. The previously
validated six-row/context-256 AOT is unchanged.

Both modes exit 0 with explicit ingress/egress context-close markers; all
terminal GPU owner queries are empty. Speculative cgroup peaks are
54,929,371,136/.178 and 31,328,419,840/.179 bytes; ordinary peaks are
56,123,568,128/.178 and 27,066,699,776/.179 bytes. All four unit swap peaks are
zero, within the existing 96-GiB/no-swap unit limits. Cgroup accounting does not
cover every CUDA allocation on these unified-memory devices.

Local captures: `/tmp/lunaflux-dspark-run-capacity-20261011-v1`.
Remote roots: `/tmp/lunaflux-dspark-real-capacity-20261011-v1`.
Units use `lunaflux-dspark-capacity-20261011-v1-{ingress,egress}.service` and
`lunaflux-dspark-capacity-20261011-v1-reference-{ingress,egress}.service`.
No payload hashing or authentication scans are added. Independent references,
retained-weight cancellation, concurrency and matched performance remain open.
