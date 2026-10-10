# Generic tokenizer frontend: actual GLM configuration

## Implemented

The shared tokenizer adds the no-normalizer ByteLevel pipeline with digit
groups of one to three Unicode numeric scalars. Its policy is selected by the
actual tokenizer schema, not the model name. The existing immutable BPE tables,
bounded worker and whole-pretoken `ignore_merges` lookup remain authoritative.
Added tokens may append outside the base vocabulary and retain their distinct
`Added` versus `Special` decode behavior; no fixed checkpoint IDs are imposed.
Unsupported normalization/regex behavior is not silently replaced.

`cmd/glm_checkpoint` now exposes `text-frames`, converting already rendered
UTF-8 prompts to canonical serial prompt frames. Its limits are 64 MiB tokenizer
JSON, 1 MiB text, model vocabulary size and context minus one output token.
The label is supplied by the caller and is not authenticated. The mode reads
no weights and performs no cryptographic payload scans or CUDA calls.

## Actual-file checks

Local capture root:
`/private/tmp/lunaflux-glm-tokenizer-live-20261010.eUawaScr`.
Its `model/tokenizer.json` is the original 20,217,442-byte file downloaded from
the installed GLM checkpoint; its JSON was not rewritten for these tests.
The `check.mbtx` automation captures output/error streams under separate,
non-overwriting names. `template-final` and `multilingual-final` are successful
runs of the final local release binary;
the earlier local path probes are preserved too. On macOS `/tmp` is a symlink:
the filesystem reader requires the canonical `/private/tmp` path.

| Check | Observed result |
| --- | --- |
| Original template, rows=1/history=64 | 13 tokens and 13 serial frames |
| Frame comparison against the prior GPU-smoke inputs | all header/payload bytes match except removed checksum slot 60..63, now zero |
| Chinese, Arabic digits, accented/decomposed Unicode, newline, emoji | 22 tokens/frames; exact original-byte decode |
| Original template, history=5 | rejected token overflow before output directory creation |
| Successful run stderr | empty |
| Native tokenizer/JSON/file/GLM-command tests | 65/65, existing migration-warning exclusions |
| No-hashing developer regression | passed, 26 campaign scripts plus loader checks |
| Token-step scan/copy/readback gate | passed without a checksum scan |

Original template:

```text
[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>a<|assistant|><think>
```

Actual IDs:

```text
154822,154824,154826,25062,287,29905,371,25,12035,154827,64,154828,154841
```

These match the prior checkpoint-vocabulary/merge diagnostic and the prompt
used in the two-host smoke. `compare-frames.mbtx` compares all thirteen frames
against downloaded originals without hashing. Their sole byte differences are
the old four-byte checksum field, which the current encoder deliberately zeros.
That diagnostic is a narrow template reference,
not an independent upstream arbitrary-text tokenizer implementation. Multilingual
roundtrip proves byte preservation, not full upstream token-boundary parity.
One separate local template run took 0.52 s wall time and 229,654,528 bytes
maximum RSS under `/usr/bin/time -l`, with zero swap. It includes tokenizer
JSON parsing/table construction and prompt encoding, not GPU inference; no
baseline or speedup claim follows from that single cold run.

## Remaining model-level work

Independent upstream tokenizer/logit parity, checkpoint chat-template rendering,
generated-text integration and complete model serving remain open. The earlier
GLM GPU smoke used pretokenized frames and an older binary. No new GPU workload,
kernel performance result or production-readiness claim is made by this change.

## Independent original-tokenizer corpus — 2026-10-11

Twelve real original-tokenizer cases now match vLLM 0.26.0's HF renderer,
including exact token IDs and exact decoded text. The original thirteen-token
GLM template remains unchanged. The other cases cover a 140-token literal
prompt, English, CJK/kana/Hangul, composed/decomposed Unicode, mixed numeric
scripts, whitespace/CRLF, emoji, original special tokens, code, multilingual
text and punctuation. Reference token counts in that order are
`13,140,20,27,14,38,10,26,8,38,34,31`; all 24 comparisons pass.

The existing reference image's Transformers 5.14.1 does not recognize
`glm5_next`, so its original-model renderer attempt fails before tokenization.
For this strictly tokenizer-only comparison, the renderer uses its supported
DeepSeek configuration as an API driver and explicit
`--tokenizer /reference-tokenizer --tokenizer-mode hf` pointing to the original,
unmodified GLM files. There is no model inference, no substituted vocabulary,
and no claim that this image can execute GLM. Requests explicitly disable
automatic special-token insertion. The unrelated API-driver configuration
does not establish GLM model or chat-template support.

The loopback-only renderer has 4 GiB/no-swap limits, measured memory peak
2,454,654,976 bytes and swap peak zero. It exits successfully after collection;
the GPU compute-owner query is empty. The original files remain read-only.
Failed attempts retain their logs: default UID 65532 could not read the GLM
files, UID 1000 had no passwd entry in the image, and the readable original
model then exposed the unsupported architecture. The successful isolated
reference container runs as root without changing host-file permissions.

Evidence: `/private/tmp/lunaflux-reference-glm-corpus-20261011-v1` and
`/private/tmp/lunaflux-reference-glm-20261011-v4`.
This supersedes the missing independent tokenizer comparison for this corpus
only. Real-weight numerical parity, generated-text GPU integration,
cancellation and multi-request serving remain open.
