# MiniCPM-V 4.7 text/decoder native ABI fixture

Run locally from LunaFlux:

```sh
moon run tests/minicpm_v47_decoder_abi --release --target native --deny-warn --warn-list '-79-29-25-20'
```

The fixture uses the existing `device_worker_alloc` software driver and
allocation counter through compile-time redirects. No real CUDA device or
checkpoint payload is loaded. Synthetic immutable weight allocations may share
storage across layers; their zeroed backing remains demand-paged. The fixture
validates extents, not weight contents or GPU arithmetic.

The combined image runtime fixture prepares all nine module owners against
the same scripted driver and borrowed synthetic weights. It enforces vision
completion before decoder submission and prevents feature reuse while decoding
is pending. A 67-token image prompt runs three prefill chunks and one decode
step: 318 vision launches plus 3,699 decoder launches. Decoder steps retain
zero heap/native-resource allocation, ten metadata uploads and producing-only
four-byte readback. Tests cover image-to-text reuse, cancellation and enqueue
failure recovery in either branch, partial preparation, and close during active
vision or decoder work. Both weight sets remain borrowed; a real combined
checkpoint upload and GPU numerical behavior are not exercised here.

It also prepares the real `Storage` owner, checks its 22 mutable allocations,
and verifies that insufficient budget is rejected before resource creation.
The typed metadata/state/buffer getters supply the forward frame; cleanup covers
both that storage owner and the frame's private output allocations.

It prepares the complete text forward owner, including all 40 decoder layers,
untied embeddings/head and selected-row final norm/logits/greedy selection.
Preparation creates 776 function handles plus two egress workspace allocations.
One prefill and two decode frames submit 924 launches each, 2,772 in total. Every
launch passes through the existing region/argument validator. The second decode
reuses/reset the prepared executor. Mutable metadata changes between steps are
staged by the caller outside the measured queue-execution scope.

The allocation counter has an explicit positive control. Queue execution must
create no heap objects, device resources, copies or readbacks. Cleanup must
balance native creation/close counters with no live children. The fixture also
checks that branch/MoE arena aliasing is rejected before preparing resources,
because the latter retains the startup expert map.

The fixture additionally prepares `minicpm_v47_request.Runner` and tests a
33-token prompt in two prefill chunks, followed by two decode frames. A scripted
greedy-output test hook supplies IDs 100, 101 and EOS; it does not compute logits.
Those four frames submit 3,696 additional launches. The broader measured step
now includes metadata encoding, ten live input uploads, queue submit and event
retirement. It allocates no heap/device resources; only the final prefill and
decode frames read one four-byte token. Two static page/slot uploads happen at
prepare. Both blocking wait and poll retirement are exercised. Further fixtures
cancel an in-flight prefill, reuse the prepared queues, reject an invalid sample,
and recover after a failed enqueue through the cancellation barrier. Resource
counters still balance after all requests.

The fixture now exercises `minicpm_v47_runtime.Execution` as a complete resource
owner. An insufficient mutable budget creates no native resources. Invalid layer
bindings fail after module/storage acquisition and cleanup restores the prior
live-resource count. A valid session owns its own modules, stream, buffers and
queues; two requests use it, then closing during an active prefill aborts work
before releasing resources. Its borrowed synthetic weights remain live. This
tests the execution lifetime separately from the full resident weight upload,
which would stream 69 GB and remains untested here.

Set `LUNA_MINICPM_V47_ASSET_DIR` to the resolved actual tokenizer assets to
also run `chat_fixture.mbt`. This executes the real model/chat adapter through
canonical HTTP/OpenAI ingress and response rendering, using the same scripted
driver. Seven successful 924-launch requests cover buffered/streaming EOS,
length/string stops and
reuse after an injected enqueue failure. A stochastic request is rejected before
launch. Prompt usage matches the existing template path. This is a software
transport/native binding check; it neither computes logits nor measures
end-to-end HTTP heap allocation or throughput.

The image HTTP fixture checks both delivery modes with actual PNG/JPEG inputs
and tokenizer assets. Streaming Accepted can reach the client while vision is
pending; disconnect there must cancel native work before the next request.
Synthetic weights and scripted token IDs still provide no GPU arithmetic or
visual-quality evidence.

`vision_fixture.mbt` now covers the independent native vision owners. A real
318-stage frame binds all 25 functions and replays twice without heap/resource
allocation. The 4x variant uses 305 stages. Input submission copies 13 packets
(10 for 4x), with no readback; feature rows become available only after
completion. Tests cover mutable/weight alias rejection, insufficient budgets
before resource creation, closed-weight partial startup, repeated requests,
cancellation, failed enqueue/reuse and active close.

The vision loader also streams all 459 selected tensors from sparse zero
fixtures into final allocations and verifies exactly 1,105,972,064 copied bytes,
using one 4 MiB host scratch. It binds/replays the same native frame with those
allocations, then checks partial-transfer-startup cleanup. No model arithmetic
occurs in this scripted driver; zero payloads are never model weights.

The multimodal fixture uses the same allocation counter. It binds completed
vision features to two mapping uploads, then drives three 925-launch prefill
chunks and one unchanged 924-launch decode through the existing request owner.
Zero-allocation/copy-count checks, cancellation, text reuse, partial bridge
startup and failed map-copy recovery pass. Fake kernels compute no embeddings
or logits; the generated-source host probe separately checks the word-copy law.

The initial full-step measurement found iterator allocations in packet lookup;
indexed traversal removes them. This is release-build allocation evidence,
not a latency or token-throughput benchmark.

An AddressSanitizer smoke run also passes for the generated MoonBit C and all
four test stub objects. Its local log is
`target/minicpm-v47-vision-native-asan.log` for the extended real-tokenizer HTTP
and vision fixtures
(`LUNA_MINICPM_V47_ASSET_DIR` set). It retains the normal prebuilt MoonBit
runtime/support libraries and mimalloc, with `detect_leaks=0`; explicit fake
resource counters cover close balance. It does not establish physical CUDA
sanitizer or GPU leak results.

Reproduce the ASan smoke with the default `_build` location:

```sh
LUNA_MINICPM_V47_ASSET_DIR=RESOLVED_ASSETS moon run scripts/test-minicpm-v47-native-asan.mbtx ABSOLUTE_REPO MOON_ROOT NEW_OUTPUT
```

This does not prove compiled-source compatibility, numerical results, real
checkpoint loading, CUDA graph capture, GPU leaks, token quality or performance.
Those remain target qualification work.

The reused driver allocation helper uses `calloc` instead of eager malloc plus
memset, preserving its zeroed-storage semantics without touching unused large
weight regions. Its original release allocation fixture remains reproducible:

```sh
moon run tests/device_worker_alloc --release --target native --deny-warn --warn-list '-79-29-25-20'
```

The `serve-planner-fixture READY_JSON PROPOSAL_JSON` mode exposes one loopback
image-chat request for the real MoonRobo client. It loads the actual tokenizer
and full selected model program, borrows synthetic weights and supplies a
bounded sampled-ID sequence through the test driver. When the reply completes,
the service retires before native owners close and the process asserts zero
live resources. It is a software fixture, never a model implementation or GPU
backend. The ordinary ABI run also checks multi-token script consumption across
producing prefill/decode steps; this path passes generated-C/native ASan.

MoonRobo's `scripts/test_lunakine_visual_planner.mbtx` orchestrates the real
planner/task processes with this server and checks the accepted fixture reference.
No actor or robot writer is loaded. Its report explicitly records scripted
tokens, absent GPU arithmetic and unqualified visual quality.
