# Bounded two-rank DeepSeek base checkpoint diagnostic

This native executable composes the existing model-neutral rank control,
plaintext activation TCP/pinned DMA, prompt frames and greedy generation with
checkpoint-backed DeepSeek base stages. It plans compact weights plus state,
workspace, I32 token tables, residuals and a caller reserve before device upload.
Rank request metadata includes absolute positions. No TLS or runtime JIT is added.

Inspection reads only bounded headers; upload reads selected tensor slices.
Neither hashes checkpoint payloads. Inventory IDs are declared labels, not
runtime-verified integrity claims. Both old sha256sum records and explicit
`label:HEX  filename` inventory records are accepted.

`config MODEL_ROOT` inspects the actual configuration without loading weights.
`token-frames MODEL_ROOT ROWS HISTORY COMMA_SEPARATED_TOKEN_IDS NEW_OUTDIR`
creates bounded canonical prefill frames (no guessed tokenizer/chat template).

`preflight-generation MODEL_ROOT ROWS HISTORY MAX_NEW_TOKENS STOP_TOKEN_OR_MINUS_ONE
FRAME_ROOT FRAME_COUNT PROMPT_BYTE_BUDGET` reads only the model config and bounded
prompt frames, without weight-shard inspection or CUDA. It rejects requests whose
entire prompt plus generation cannot fit, missing/malformed/out-of-order chunks,
or inconsistent request identities. The DSpark two-host runner uses this before
starting either rank. Normal generation also performs this planning before
opening CUDA or creating output; it retains those same immutable prompt bytes
and closes their file authority rather than reopening files during prefill.
Rebuild the runner's checkpoint executable to obtain this command. An older
prepared executable cannot satisfy preflight; the runner stops before launching
either GPU rank rather than silently bypassing capacity planning. Existing
diagnostic evidence remains unchanged.

The remaining modes share this prefix:

```
MODE MODEL_ROOT SHARD_SHA256 ROWS HISTORY BUDGET0 BUDGET1 RESERVE
```

Append these arguments:

- `export NEW_OUTDIR`: exact two-stage source and placement sizes, no GPU upload.
- `egress AOT_ROOT AOT_FILE CONTROL_LISTEN ACTIVATION_LISTEN`
- `ingress AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT FRAME_COUNT NEW_OUTDIR`
- `generate AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT PREFILL_COUNT MAX_NEW_TOKENS STOP_TOKEN_OR_MINUS_ONE NEW_OUTDIR`

Build exported sources offline for the actual CUDA target. Run ranks under
separate no-swap process limits below available unified memory. Reserve runtime,
module, protocol, pinned transfer and reader metadata memory outside stage weights.
Each rank uploads only its assigned interval. Close activation/pipeline owners
before ranks/stages, modules, streams and contexts.

The native async runtime handles SIGINT/SIGTERM by cooperative cancellation.
`generation_started` marks entry into generation after prefill; it is emitted
once, not per token. `rank_cleanup: role=... context_closed=true` is emitted only
after explicit context close succeeds during reverse-order teardown. Context
close rejects live child resources. This terminal marker is not successful
request completion: a cancelled request must not publish a normal tokens file.

The un-suffixed modes execute only base layers, even for a DSpark checkpoint.
`export-dspark`, `egress-dspark` and `generate-dspark` accept the same respective
arguments and additionally plan and execute the actual three-block predictor.
The egress budget includes predictor banks, local embedding, captures and result
buffers; the vocabulary is borrowed from base egress. Target verification
executes the two-rank decoder placement; prediction state and its local
embedding/head are owned by egress.

After each successful whole-base commit, prefill chunks prime main KV; decode
steps execute prediction. Commit acknowledgement waits for the prediction queue
to retire before request metadata can be reused. Terminal diagnostic output
contains the dependent draft blocks and target-controlled tokens. The
`generate-dspark` route performs greedy target verification with state
backup, rollback/replay and accepted-input commit before publishing output.
`generate-reference-dspark` retains the same checkpoint and prepared geometry
but runs ordinary greedy generation without speculative verification.

Native compile and component GPU results are not proof of complete-checkpoint
numerical correctness or throughput. The attached route has not yet passed an
independent upstream parity check. Real two-rank checkpoint execution and
internal greedy parity are recorded in
[the real-checkpoint report](../../docs/BENCHMARK_DSPARK_GREEDY_VERIFICATION_2026-10-10.md)
and [the literal-text report](../../docs/DEEPSEEK_LITERAL_TEXT_FRONTEND_2026-10-10.md),
not claimed as optimized serving throughput.

## Literal text frontend

`text-frames MODEL_ROOT ROWS HISTORY TOKENIZER_LABEL INPUT_UTF8_FILE NEW_OUTDIR`
loads the original tokenizer JSON through the common native frontend and writes
the same prompt frames as `token-frames`, plus token IDs and decoded text. The
input file is literal already-rendered text: no implicit BOS/EOS, chat template
or normalization is added. The installed identity-normalizer ordered numeric,
CJK/kana and word/symbol Split sequence retains all earlier split boundaries
during BPE. The 64-hex label is caller supplied, not a computed checksum. This
mode reads no weight inventory/shards and opens no CUDA context. Token output
reserves one context position for generation; overflow is rejected.

`decode-tokens MODEL_ROOT TOKENIZER_LABEL COMMA_SEPARATED_TOKEN_IDS NEW_OUTPUT_FILE`
decodes generated output through the same original tokenizer envelope, preserving
explicit special tokens. It reads only config/tokenizer metadata, adds no template
and never loads weights or opens CUDA. Output is raw decoded bytes; arbitrary
partial token vectors are not guaranteed to form complete UTF-8 text.
