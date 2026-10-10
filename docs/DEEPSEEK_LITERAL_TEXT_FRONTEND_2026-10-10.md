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
modules were freshly compiled from the frontend commit. No GPU request has
started in this replay: .178 is still owned by the actual-caption MiniMax joint
request. The runner now consumes the prepared prompt directory and frame count
instead of hard-coding one BOS frame, and checks both GPUs before starting
either owner. Matched speculative/ordinary GPU results remain pending.

A second replay preparation uses the exact non-thinking single-user prompt
string from the checked-out vLLM regression
`tests/tokenizers_/test_deepseek_v4.py`:
`<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>`.
The original tokenizer emits `0,128803,19923,128804,128822`, one frame, with
empty stderr and exact decoded text. This checks a reference-rendered string,
not independent tokenizer-ID or model-output parity. It is prepared at
`/tmp/lunaflux-dspark-run-text-20261010-v2` and the corresponding remote
`/tmp/lunaflux-dspark-real-text-20261010-v2` roots. The active execution sequence
waits for the existing MiniMax unit's terminal success and GPU release before
starting speculative then ordinary target generation on this identical prompt.
No existing model owner is restarted. Runner prompt/output regressions and the
27-script no-hashing gate pass; the token-step scan/copy/readback gate also passes.
