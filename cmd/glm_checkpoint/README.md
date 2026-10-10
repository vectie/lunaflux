# GLM checkpoint pipeline diagnostic

Native executable for the installed GLM-5.3 Flash NVFP4 checkpoint. It reads
`config.json` and a safetensors inventory, inspects bounded shard headers
through the streaming reader, and plans all 45 layers using
actual tensor storage plus state/workspace/text costs. It never expands packed
weights into a host model arena or runs JIT in the request path.

`config MODEL_ROOT` admits only the bounded real configuration without reading
shards or touching CUDA. This is a cheap compatibility check before full startup.

`preflight-generation MODEL_ROOT ROWS HISTORY MAX_NEW_TOKENS STOP_TOKEN_OR_MINUS_ONE
FRAME_ROOT FRAME_COUNT PROMPT_BYTE_BUDGET` resolves the entire request using only
config and prompt bytes before either rank starts weight loading. Prompt length
plus generation must fit the context; chunk positions, order, final sampling and
request identity must agree. No hashes, shard reads or CUDA are involved. Normal
generation likewise retains an immutable, byte-budgeted prompt replay before
device preparation/output creation and closes input-file authority; files are not
reopened during prefill. This uses the same model-neutral plan as DeepSeek.

`preflight-generation-sequence` uses the same arguments with `REQUEST_COUNT`
instead of `FRAME_COUNT`. The common `integration/serial_prompt_file` adapter
reads `frame-counts.txt` once and prepares every `request-i/plan-N.bin` under one
aggregate frame-byte budget before either CUDA context opens. Capacity failure
in a later request rejects the whole queue before model/device preparation.

`token-frames MODEL_ROOT ROWS HISTORY COMMA_SEPARATED_TOKEN_IDS NEW_OUTDIR`
constructs canonical contiguous prompt chunks from already-tokenized input,
without scanning checkpoint shards or opening CUDA. Only the last chunk samples;
sequence/model/request identities match `generate`. It is an offline diagnostic
producer, not live scheduler admission, a chat template or a tokenizer. The
model's actual tokenizer must supply these IDs.

`text-frames MODEL_ROOT ROWS HISTORY TOKENIZER_LABEL INPUT_UTF8_FILE NEW_OUTDIR`
reads the installed `tokenizer.json` with the common tokenizer reader and emits
the same prompt frames, plus `tokens.txt` and `decoded.txt`. The input file must
have an absolute canonical path and contain the already rendered prompt/chat
template. This mode does not invent a chat template or add normalization beyond
the installed tokenizer's declared pipeline. The
generic no-normalizer digit-triplet ByteLevel profile supports the actual GLM
regex, ordinary/special added tokens and `ignore_merges=true`, without a
model-name branch or rewriting its JSON. Tokenizer JSON is bounded to 64 MiB;
text to 1 MiB; token output reserves one context position for generation.
`TOKENIZER_LABEL` is a caller-supplied 64-lowercase-hex association label, not a
computed or verified checksum. This offline mode does no payload hashing,
weight-shard reads or CUDA work. It is not a text HTTP serving endpoint.
Text encoding and little-endian ID serialization are owned by the shared
`integration/checkpoint_text_input` frontend; GLM retains ownership of prompt
chunking and its execution-frame protocol.

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
  diagnostic entry point; `text-frames` supplies tokenized input, but this is
  still not text HTTP serving.
- `generate-sequence` uses the same nine mode arguments as `generate`, with
  `REQUEST_COUNT` instead of `PREFILL_COUNT` and the queue layout above. Both
  rank/weight/channel owners remain resident through the queue; each request
  retires on both ranks before the next starts at position zero. Transport
  epochs increase across requests, rather than restarting at one. Completed
  outputs live under `NEW_OUTDIR/request-i/`. This finite sequential diagnostic
  is not concurrent batching, an HTTP service or a physical qualification claim.

Use the same checkpoint inventory, geometry, budgets and reserve on both ranks.
Set budgets from current free unified memory, leaving OS/other-process headroom;
launch within an externally enforced memory/time budget. `RESERVE` covers module,
request ports, DMA/control and process overhead; it is not extra GPU capacity.
The executable has bounded file/header/chunk/protocol reads, but is not itself
a replacement for a host cgroup memory ceiling. Start with one-row/history-64
correctness, not long-context/concurrency campaigns.

This is a diagnostic execution entry point, not a production-ready admission or
a positive real-model correctness/performance claim. Exact checkpoint loading,
independent GPU numerical parity remains to be tested. The real two-Spark
13-token prompt/eight-output smoke now completes with released GPUs and no
swap, documented in `docs/BENCHMARK_GLM_NOHASH_SMOKE_2026-10-10.md`. That binary
predates the `text-frames` frontend. It uses the existing
correctness-first reference lowering, not the throughput Qwen path. Startup
uploads use an explicit scoped file-handle session. Header inspection and
upload perform no checkpoint payload hashing. Existing sha256sum inventory
records are declared labels only; `label:HEX  filename` records explicitly
carry a supplied checkpoint label, not a shard checksum. Cross-shard
duplicate-name detection is indexed rather than quadratic. These are source
improvements, not a measured full-checkpoint startup timing claim.

The per-shard tensor bound equals the bounded total population (500,000), not
an assumed weight-shard distribution. ModelOpt may concentrate all 37,152
input scales in one small auxiliary shard. File/header byte limits remain
independent. Header overlap validation uses sorted ranges; startup reports the
header-inspection and binding/placement phases. Capture both output streams in
the service journal so a terminal startup error is retained.
