# MiniCPM-V 4.7 resident text executable

This local diagnostic executable joins startup preparation, the resident owner,
request execution and final text decoding for the selected flagship model. A
queue of repeated requests uses one weight load and one prepared execution
session. The `serve` mode exposes buffered/streaming chat over that
same resident session. Separate `inspect-media`/`run-media` modes use the combined
text/vision resident. It sends no robot commands. Continuous batching remains
separate work. `serve-media` adds inline image input in either delivery mode.

Physical status,2026-11-17: the full selected checkpoint runs on the supplied
GB10 with all nine AOT modules. Four text diagnostics match34 upstream greedy
decisions, while measured logits differ. Real image execution returns `Red`
and `Blue` for synthetic squares and reuses text state correctly. Actual JSON/
SSE/auth/media serving and MoonRobo's two-host recorded-shadow integration pass.
See the [physical evidence ledger](../../docs/MINICPM_V47_SPARK.md). Historical
software-smoke results below retain their narrower scope; broad visual/planner
quality and sustained serving remain open.

The model root must contain the pinned `config.json`, `tokenizer.json`,
`model.safetensors.index.json` and its actual 16 BF16 shard files. Use revision
`98e79ba972b586ac6146d3505771ce1467e199ba`. The CUBIN root contains the seven modules
produced by the existing offline AOT build helper. No Python runtime, JIT
compilation, weight checksum or automatic smaller-model substitution is used.

Inspect model/header compatibility, tokenizer prompts and payload budgets with
no GPU access or weight-payload reads:

```sh
moon run cmd/minicpm_v47_generate --release --target native -- inspect MODEL_ROOT cmd/minicpm_v47_generate/requests.example.json NEW_INSPECTION_DIRECTORY
```

After the real checkpoint and matching AOT artifacts are staged on the supplied
Spark, explicitly run generation (device ordinal 0 and a 100,000,000,000-byte
allocation ceiling shown as an example):

```sh
moon run cmd/minicpm_v47_generate --release --target native -- run MODEL_ROOT CUBIN_ROOT cmd/minicpm_v47_generate/requests.example.json NEW_RESULT_DIRECTORY 0 100000000000
```

The run mode checks observed BF16 `sm121` capability and the supplied ceiling,
then performs one bounded streaming upload using 4 MiB host scratch. The initial
explicit weight/storage/output payload budget is 69,473,047,868 bytes; module,
driver and queue overhead is additional. Device allocation may still fail and
retains cleanup authority. The actor process is independent of this command.

The fixed text profile is single-flight greedy, 32 rows/chunk and 4096 context
tokens. It selects the same full model for text as for planned vision support.
Stop IDs come from the pinned model's `<|endoftext|>` / `<|im_end|>` markers.
Unsupported media/sampling options are rejected. The upstream generation file
uses stochastic sampling; this executable deliberately requests greedy results
for first execution and comparison. It is not a claim of equivalent task quality.

All output paths are new. `plan.json` is a preparation-only snapshot and always
records no GPU execution. Successful run startup adds `startup.json`; each
completed request adds `result-N.json` with emitted IDs, decoded text, usage and
finish reason. A truncated or invalid UTF-8 byte sequence uses replacement text
with `text_utf8_valid: false`; the exact token IDs remain available. Result
decoding and file writes occur after request completion,
outside token steps. A failure returns nonzero and may leave partial receipts;
missing result files must not be interpreted as completed requests. Numerical
and device qualification remain false even when generation completes.

The local software smoke uses actual config/tokenizer/headers but **sparse zero
payloads**, and invokes only `inspect`:

```sh
moon run scripts/test-minicpm-v47-bootstrap.mbtx /absolute/path/lunaflux ACTUAL_ASSET_DIRECTORY NEW_FIXTURE_OUTPUT
```

The smoke's `sparse-software-fixture` directory is not a model checkpoint for
inference. Its successful plan is retained locally at
`target/minicpm-v47-bootstrap-smoke-v1/inspection/plan.json`. No run-mode GPU
execution, actual checkpoint upload, NVCC compatibility or model numerics have
been tested here.

## Chat serving

After staging the real checkpoint and matching AOT modules on the supplied
Spark, set `LUNAFLUX_API_KEY` in the environment and start the resident endpoint:

```sh
moon run cmd/minicpm_v47_generate --release --target native -- serve MODEL_ROOT CUBIN_ROOT 127.0.0.1:8000 0 100000000000
```

The existing bounded LunaFlux ingress requires a configured bearer policy; no
secret is printed. `LUNAFLUX_REQUEST_TIMEOUT_MS` optionally sets the request
timeout (default 30000, maximum 300000). The endpoint accepts
`POST /v1/chat/completions` with `Content-Type: application/json`, an explicit
`Content-Length`, bearer authorization, model `MiniCPM-V-4.7-35B-A3B`, greedy
temperature, and either `stream:false`/omitted for JSON or `stream:true` for SSE.
It uses the pinned system/user text template.
Stop-token/string handling and UTF-8 framing reuse the canonical output owner.
Truncated output returns finish reason `length`, which MoonRobo's planner rejects.

For an interactive streaming request:

```sh
curl -N --fail-with-body http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $LUNAFLUX_API_KEY" -H 'Content-Type: application/json' --data '{"model":"MiniCPM-V-4.7-35B-A3B","temperature":0,"max_tokens":256,"stream":true,"messages":[{"role":"user","content":"Describe the next task."}]}'
```

One request executes at a time; concurrent arrivals get 429. Timeout/disconnect
retires model work before reuse. Streaming uses `text/event-stream`, token deltas,
terminal usage and `[DONE]`; after headers a timeout/failure closes the stream
without a second HTTP response. The endpoint currently has no `/v1/models` or
health routes. It does not publish qualification or performance
claims. The initial local coverage uses canonical HTTP with the scripted native
driver; actual selected-model serving evidence is recorded above.

## Local image queues

Use the full selected model with local PNG/JPEG images and mixed text requests:

```sh
moon run cmd/minicpm_v47_generate --release --target native -- inspect-media MODEL_ROOT MEDIA_ROOT cmd/minicpm_v47_generate/media-requests.example.json NEW_INSPECTION_DIRECTORY
moon run cmd/minicpm_v47_generate --release --target native -- run-media MODEL_ROOT CUBIN_ROOT MEDIA_ROOT cmd/minicpm_v47_generate/media-requests.example.json NEW_RESULT_DIRECTORY 0 100000000000
```

The media CUBIN root includes all nine v19 modules, adding `vision_encoder.cubin`
and `image_scatter.cubin` to the seven text modules. The combined explicit
allocation payload is 71,008,215,248 bytes before module/driver/queue overhead.
The AOT geometry remains 16,384 raw patches, 8,192 patches/crop and 4096 context
tokens. All queue images must fit these fixed capacities.

The example paths `rgb.png` and `baseline.jpg` refer to caller-provided media
under `MEDIA_ROOT`. The small synthetic examples live in `media/rgb8/fixtures`;
they exercise data preparation, not useful robot observations. The queue format
supports a system string and an ordered `user` array of `{type:"text",text:...}`
or `{type:"image",path:...}` objects. Paths resolve through an approved local
root. There is no implicit remote URL download or base64 schema. The typed
[media preparation API](../../integration/minicpm_v47_media/README.mbt.md) also
accepts decoded camera frames without encoding them as files.

Each encoded image is at most 32 MiB and 16,777,216 pixels; the queue's retained
prepared BF16 pixels are at most 256 MiB. Resize working memory is admitted
separately. PNG/JPEG conversion uses the vendored memory-only RGB8 decoder;
its EXIF orientation and color/precision limits are documented in the
[RGB8 contract](../../media/rgb8/README.mbt.md). Prompt/image ordering and layout
are derived together before CUDA initialization.

`inspect-media` writes a media plan receipt with token IDs, image/crop/feature
counts, prepared pixel bytes, and combined allocation budget. It records no
weight payload reads or GPU execution. Run mode loads one combined resident,
waits for vision completion, binds device features, then drives the existing
decoder; later text requests clear the image map. It writes startup/results in
the same new-directory pattern as text mode. Cancellation/active teardown are
owned by the resident. The existing `serve` mode still accepts text only.

The real CLI inspection smoke at `target/minicpm-v47-media-inspection-v2/plan.json`
uses actual config/tokenizer/header metadata and explicitly labelled sparse-zero
shards. Its PNG/JPEG pair prepares 2,080 raw patches/130 feature rows and a
180-token prompt, followed by a text request. It is not an inference checkpoint.
Actual combined upload, run-mode generation and Spark/NVCC execution now have
the sampled physical evidence above. Broader visual quality remains open.

## Image serving

After staging the actual checkpoint and all nine matching CUBINs on Spark:

```sh
moon run cmd/minicpm_v47_generate --release --target native -- serve-media MODEL_ROOT CUBIN_ROOT 127.0.0.1:8000 0 100000000000
```

It uses the existing `LUNAFLUX_API_KEY` bearer policy and
`LUNAFLUX_REQUEST_TIMEOUT_MS`. The route remains `POST /v1/chat/completions` with
JSON `Content-Length`, greedy sampling and either JSON or SSE delivery selected
by `stream`. Text strings and
ordered text/image content lists are accepted; image URLs are inline PNG/JPEG
data URIs. `media-chat.example.json` contains small synthetic fixtures:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $LUNAFLUX_API_KEY" -H 'Content-Type: application/json' --data-binary @cmd/minicpm_v47_generate/media-chat.example.json
```

One optional nonempty system message precedes one user message. Outer message
whitespace is trimmed; absent system messages stay absent. `detail:auto` and
`detail:high` both use the selected processor's full profile. Remote/file URLs,
low-detail mode, tools and video are rejected. The full HTTP body, including
all base64 inputs, is at most 16 MiB. Pixel, crop, context and working-memory
ceilings remain those of the prepared runtime. Decoding/resizing happens before
token execution, and features stay on device during decoding.

The shared HTTP exchange authenticates before model admission. The normalizer
keeps generation fields for canonical validation and retains the original HTTP
lease while admitting normalized placeholder JSON. Text/image replies share
UTF-8 framing, stops and usage. Timeout/disconnect cancels pending vision or
decoder work before reuse; another arrival receives 429. This endpoint uses one
request at a time and sends no robot commands. The original `serve` still loads
the seven-module text session.

Release build and native scripted-driver loopback tests pass. Actual Spark
serving also passes the separate colored-square JSON/SSE/reuse diagnostic;
model/visual quality remains broader work. The tiny synthetic example supplied
in this directory remains a protocol fixture, not a grounding test.
