# MiniCPM-V 4.7 chat session adapter

This adapter joins a prepared full flagship text runner to the shared session
HTTP transport. It borrows model/tokenizer/runner state and owns incremental
output buffers. It adds no model weights, native context, actor or robot control.

Text and image services share model-owned message normalization, including the
pinned Python/Jinja Unicode/control whitespace trim. The text endpoint accepts
string/text-list content with a 1 MiB body ceiling and rejects image parts before
decoding. The original authenticated HTTP lease remains attached through
canonical admission. All normalization happens before device/token execution.

The existing OpenAI ingress applies the pinned no-thinking template and publishes
a canonical text handoff. The backend tokenizes that rendered prompt once, builds
positions and reserves the requested output allowance within the 4096-token
session context. It does not apply a second template or encode during decode.
Greedy generation supports both buffered JSON and streaming SSE delivery;
stochastic sampling is rejected. Required EOS IDs are joined with caller stops. The model remains
MiniCPM-V 4.7 35B-A3B with all 40 decoder layers.

The caller constructs `Backend::new(model, tokenizer, resident.runner())`, then
`backend.service(policy)` and retains both through `service.serve(listener)`.
Stop the service before closing the resident session and its context. The default
published model alias is `MiniCPM-V-4.7-35B-A3B`; the bearer policy is the existing
LunaFlux HTTP policy, not a new deployment approval layer.

```mbt check
///|
fn _service(
  model : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  runner : @request.Runner,
  policy : @auth.LunaApiAuthPolicy,
) -> @session.Service raise {
  let backend = @minicpm_v47_chat.Backend::new(model, tokenizer, runner)
  backend.service(policy)
}
```

Canonical token events use the existing incremental decoder and stop matcher.
They preserve split UTF-8 and stop prefixes across tokens; the renderer either
buffers the final HTTP choice or writes each SSE chunk with bounded write
backpressure. The canonical engine handoff remains event-streaming even
when HTTP delivery is buffered. Token-limit completion discards an incomplete
UTF-8 suffix under the existing output contract. Usage counts sampled tokens,
including a sampled stop ID; visible text excludes matched stops.

Timeout, disconnect or callback failure invokes the runner's cancellation barrier
before request state can be reused. A failed barrier leaves backend readiness
false. Successful cleanup restores readiness without reloading weights. A failed
request never selects a CPU, smaller-model or alternate-runtime fallback.

The native software-driver fixture uses the actual tokenizer and real prepared
40-layer frame, but scripted token IDs and synthetic weights. It exercises EOS,
length/string stops in both delivery modes, sampling rejection, enqueue
failure/recovery and seven successful 924-launch requests across real loopback HTTP. Prompt usage agrees
with the existing text-template path for Unicode-trimmed strings, text lists and
absent system messages. This is binding/lifecycle evidence, not
model arithmetic, quality or physical CUDA execution. The queue/request allocation
gate remains separate; no end-to-end HTTP allocation or throughput claim is made.

## Image chat

`Backend::with_images` borrows a prepared combined image execution and its
vision program. Its service accepts the same route, streaming choice and controls, with
one optional system message followed by one user message. User content
may be a string or a list of text/image parts. Image parts use
`{type:"image_url",image_url:{url:"data:image/png;base64,..."}}` or JPEG. Optional
`detail` is `auto`/`high`, both using the pinned full processor profile; `low`
and remote/file URLs are rejected. No filesystem or outbound network authority
enters image admission. The full HTTP body is bounded to 16 MiB, including all
base64 images; existing processor/crop/context budgets also apply.

```mbt check
///|
fn _image_service(
  model : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  vision : @vision.Program,
  execution : @image_runtime.Execution,
  policy : @auth.LunaApiAuthPolicy,
) -> @session.Service raise {
  @minicpm_v47_chat.Backend::with_images(model, tokenizer, vision, execution).service(
    policy,
  )
}
```

The model-owned normalizer derives native placeholders and a typed RGB sidecar,
preserving all generation options for the canonical parser. Duplicate keys are
rejected before map-backed normalization. The parser retains the original
authenticated HTTP lease through admission, then publishes the usual handoff.
Image preparation tokenizes once using the pinned trim/newline/optional-system
profile. The shared output driver handles text and image sessions with the same
incremental UTF-8, EOS/string-stop and usage laws. Vision completes before
decoder submission, and its features stay on device. Later text requests disable
the image map. Media parsing/decoding/resize finish before token execution.

The native loopback fixture uses actual tokenizer assets and PNG/JPEG inputs.
It checks image replies/usage, no-system text reuse, 401/400/429 and enqueue
failure/recovery. A held completion event verifies that pending vision does not
start decoding, and that timeout or peer disconnect retires it before reuse.
Streaming image output/usage also match the buffered path. Disconnect after
streaming Accepted headers while vision is pending retires native work before
releasing request credit. Once headers are sent, an interrupted stream ends
without a false `[DONE]`; canonical completion/failure produces one terminal
marker through the shared renderer.
This is scripted-driver evidence, not actual model arithmetic or grounding.
