# Bounded two-rank DeepSeek base checkpoint diagnostic

This native executable composes the existing model-neutral rank control,
plaintext activation TCP/pinned DMA, prompt frames and greedy generation with
checkpoint-backed DeepSeek base stages. It plans compact weights plus state,
workspace, I32 token tables, residuals and a caller reserve before device upload.
Rank request metadata includes absolute positions. No TLS or runtime JIT is added.

`config MODEL_ROOT` inspects the actual configuration without loading weights.
`token-frames MODEL_ROOT ROWS HISTORY COMMA_SEPARATED_TOKEN_IDS NEW_OUTDIR`
creates bounded canonical prefill frames (no guessed tokenizer/chat template).

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

This executes base layers even when the checkpoint contains DSpark extensions.
It does **not** execute DSpark prediction/MTP, and must not be presented as full
DSpark qualification. Native compile and component GPU results are not proof of
complete-checkpoint numerical correctness or throughput.
