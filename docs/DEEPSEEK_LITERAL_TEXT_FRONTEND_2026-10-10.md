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

## Stop-aware follow-up preparation

`prepare-deepseek-text-replay.mbtx` optionally accepts a maximum generated-token
count and stop token. The runner retains `16/-1` for old fixed-length diagnostics,
but now verifies the requested bound, the terminal `Length`/`StopToken` reason,
and that no published token follows the first configured stop token. The actual
number of committed speculative input rows must equal output count minus the
initial seed, rather than hard-coding fifteen rows. Regression cases cover both
old diagnostics and shorter stop-terminated output.

The unchanged actual chat prompt, binary and device modules are prepared anew
at `/tmp/lunaflux-dspark-run-text-eos-20261011-v1` with a 64-token maximum and
the original checkpoint's `eos_token_id=1`. This preparation opens no GPU and
does not overwrite the completed fixed-length run. Its speculative/ordinary
physical completion is not yet claimed.
