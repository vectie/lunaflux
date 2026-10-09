# GLM checkpoint pipeline diagnostic

Native executable for the installed GLM-5.3 Flash NVFP4 checkpoint. It reads
`config.json` and a sha256sum-style safetensors inventory, authenticates
the shards through the existing streaming reader, and plans all 45 layers using
actual tensor storage plus state/workspace/text costs. It never expands packed
weights into a host model arena or runs JIT in the request path.

`config MODEL_ROOT` admits only the bounded real configuration without reading
shards or touching CUDA. This is a cheap compatibility check before full startup.

`token-frames MODEL_ROOT ROWS HISTORY COMMA_SEPARATED_TOKEN_IDS NEW_OUTDIR`
constructs canonical contiguous prompt chunks from already-tokenized input,
without scanning checkpoint shards or opening CUDA. Only the last chunk samples;
sequence/model/request identities match `generate`. It is an offline diagnostic
producer, not live scheduler admission, a chat template or a tokenizer. The
model's actual tokenizer must supply these IDs. GLM's non-normalizing ByteLevel
regex, three-digit groups, ordinary added tokens and `ignore_merges=true` are
not yet supported by the existing Qwen tokenizer profile.

Common arguments, all budgets in bytes:

```
MODE MODEL_ROOT SHARD_SHA256 ROWS HISTORY BUDGET0 BUDGET1 RESERVE MODE_ARGS
```

`SHARD_SHA256` may be relative to `MODEL_ROOT` or an absolute file in a separate
metadata directory. Neither mode writes to the model source. All shard locators
inside the inventory remain relative to the read-only model root.

- `export NEW_OUTDIR`: write the exact planned `stage-0.cu`, `stage-1.cu` and
  placement receipt without GPU allocation. Compile these offline for each
  assigned device before execution; older arbitrary stage halves are not valid.
- `egress AOT_ROOT AOT_FILE CONTROL_LISTEN ACTIVATION_LISTEN`: stream/upload the
  last stage, then listen for the two plain-TCP connections. Wait for its loaded
  message before starting ingress. No TLS requirement is introduced.
- `ingress AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT FRAME_COUNT
  NEW_OUTDIR`: stream/upload the first stage and execute bounded canonical wire
  frames `plan-0.bin` through `plan-(FRAME_COUNT-1).bin` from one contiguous
  request. Write each canonical completion without overwrite, then release
  request state. Frames use model generation 1 and greedy sampling. The current
  command accepts pretokenized frames, not a text HTTP serving endpoint.
- `generate AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT
  PREFILL_COUNT MAX_NEW_TOKENS STOP_TOKEN_OR_MINUS_ONE NEW_OUTDIR`: consume
  prompt-only `plan-i.bin` chunks ending in one final-prefill frame, then feed
  each actual terminal-rank token into successive canonical decode steps.
  `-1` disables token stopping; otherwise use the tokenizer's real stop ID.
  Length/context/EOS bound continuation. After both ranks release the request,
  write generated IDs to `tokens.txt` without overwrite. The decode loop does
  no evidence rendering/filesystem writes. Prompt frame input is still the
  diagnostic entry point; this is not text HTTP serving or tokenizer support.

Use the same checkpoint inventory, geometry, budgets and reserve on both ranks.
Set budgets from current free unified memory, leaving OS/other-process headroom;
launch within an externally enforced memory/time budget. `RESERVE` covers module,
request ports, DMA/control and process overhead; it is not extra GPU capacity.
The executable has bounded file/header/chunk/protocol reads, but is not itself
a replacement for a host cgroup memory ceiling. Start with one-row/history-64
correctness, not long-context/concurrency campaigns.

This is a diagnostic execution entry point, not a production-ready admission or
a positive real-model correctness/performance claim. Exact checkpoint loading,
GPU numerics and two-host generation must still be tested. Self-feeding greedy
continuation is implemented, not a positive actual-checkpoint result. It uses the existing
correctness-first reference lowering, not the throughput Qwen path. Startup
uploads use an explicit scoped authenticated-handle session: each
referenced shard is reauthenticated once for the whole stage upload, rather
than once per layer/weight owner. Initial inspection and the upload session
remain separate passes; neither is repeated in token execution. Cross-shard
duplicate-name detection is indexed rather than quadratic. These are source
improvements, not a measured full-checkpoint startup timing claim.

The per-shard tensor bound equals the bounded total population (500,000), not
an assumed weight-shard distribution. ModelOpt may concentrate all 37,152
input scales in one small auxiliary shard. File/header byte limits remain
independent. Header overlap validation uses sorted ranges; startup reports the
authentication and binding/placement phases. Capture both output streams in
the service journal so a terminal startup error is retained.
