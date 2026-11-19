# Selected robot model: MiniCPM-V 4.7 35B-A3B

The selected DGX Spark target is **`openbmb/MiniCPM-V-4.7-35B-A3B`**, revision
**`98e79ba972b586ac6146d3505771ce1467e199ba`**. The user prioritizes the strongest
MiniCPM capability rather than a small model chosen for easier integration.
For robot vision and planning, this is the current flagship V variant; that
does not assert that one model wins every text/audio/task benchmark.

Sources: [official model card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B),
[pinned config](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/98e79ba972b586ac6146d3505771ce1467e199ba/config.json),
and [OpenBMB release chronology](https://openbmb.cn/docs/en/changelog).

## Current physical deployment, 2026-11-17

The target `192.168.2.179` is reachable through cluster SSH port32123. It reports
NVIDIA GB10, AArch64 Ubuntu24.04.2, driver580.178.04 and NVCC13.0.88. All nine
v19 CUDA modules compile for `sm_121`. A real-driver probe loads all nine and
checks6,144 output words after1,000 image-scatter launches. The native LunaFlux
release CLI builds using the host's October23 toolchain. Normalization retains
the same FP64-to-FP32 rounding through `Float::from_double`; three obsolete
test-only `try?` forms were replaced. The affected local suite passes42/42.

The isolated host root is `/home/wlc004s/lunakine-deployment-20261117`.
`scripts/stage-minicpm-v47-checkpoint.mbtx` supports resumable pinned HF/mirror
acquisition or ModelScope payload transport. ModelScope downloads must match
the pinned manifest's sixteen SHA-256 values once at acquisition; metadata
comes from the pinned HF assets. Hashing never enters model startup or token
execution. Acquisition is complete: all sixteen shards match the pinned
manifest. Previously verified receipts are reused only for caller-owned staged
shards assumed immutable after acquisition. The full native text resident
loads and completes two greedy requests with30/58 input tokens, returning
`ready` and `4` and stopping on EOS. These are actual CUDA results on GB10;
upstream numerical, vision, serving and performance acceptance remain open.
Historical software-only evidence below retains its original scope. The
[full-model offline reference](../tests/minicpm_v47_model_oracle/README.md)
uses unchanged Transformers5.18.0 in an isolated GPU environment and compares
matching token inputs outside production inference.

The extended native diagnostic completes four requests and retains34 full
vocabulary rows, including EOS, outside the production token path. Actual
combined vision/text execution also completes: identical color questions for
lossless448x448 red/blue-square images produce `Red` and `Blue`, then a text-only
request produces `ready` with the same resident. Each image contributes64
feature rows to a108-token prompt. These initial grounding/reuse results do
not qualify broad visual accuracy or upstream numerical equivalence. Receipts
are retained in MoonRobo's
`output/lunakine-host-verification-20261117/spark-native-vision-v1` and raw logit
files remain on the Spark deployment root.

The actual upstream GPU reference now completes with Transformers5.18.0 and
PyTorch2.14 eager arithmetic, using CPU-backed BF16 loading followed by moving
the whole unchanged model to CUDA. All four greedy sequences and all34 sampled
argmax decisions match, including EOS;8,436,896 finite logit values are compared.
Maximum absolute logit difference is1.53125, so numerical equivalence remains
unproven. CUDA allocator peak counters are unavailable with caching disabled;
zero values in that reference receipt do not measure zero memory use.

Actual `serve-media` passes authenticated JSON/SSE text/image requests,
unauthenticated401 rejection and text reuse after images. The live HTTP verifier
is `scripts/verify-minicpm-v47-live-http.mbtx`; it accepts endpoint/media/output
paths and reads the bearer from the environment without logging it. These tests
buffer the SSE body and do not establish first-token latency. MoonRobo's real
coordinator also completes text-only and image-assisted planning across the LAN
to the Ascend recorded-feature shadow actor. LunaFlux imports no robot/runtime
code for this integration. The temporary listeners were stopped after testing.
Broader numerical/visual/planner quality and sustained serving remain open.

MiniCPM file/HTTP ingress now uses EXIF-aware decoding. The separate upstream
image-loader reference passes126 orientation cases and72 full PNG cases;
54 JPEG cases retain sampled channel difference up to3 after the progressive
entropy-buffer correction. Full JPEG/color-profile parity and effects on
model outputs remain open. The26 affected tests, release build and native
codec ASan/UBSan probe pass. See the
[ingress reference](../tests/minicpm_v47_image_load_oracle/README.md).

## Architecture implications

This is not the Llama-based MiniCPM5-2B. Its configuration declares:

| Component | Selected semantics |
| --- | --- |
| Outer model | `MiniCPMV4_7ForConditionalGeneration`, `minicpmv4_7`. |
| Text decoder | Qwen3.5-family hybrid MoE, hidden 2048, 40 layers. |
| Attention | Three linear layers followed by one full layer: 30 recurrent and 10 full layers, with output gating. |
| Full attention | 16 query heads, two KV heads, head width 256, partial interleaved MRoPE. |
| Linear attention | Convolution width 4; 16 key heads and 32 value heads, widths 128; FP32 recurrent state. |
| Experts | 256 routed experts, top eight per token, intermediate 512, plus a shared expert. |
| Vocabulary | 248144; token IDs and visual framing differ from the existing pinned Qwen artifact. |
| Vision | 27-layer encoder, hidden 1152, patch size 14, GELU-tanh; visual compression inserts at layer 6. |
| Image preparation | Slice-aware processing, canvas MRoPE, 4x/16x compression; pinned default 16x. |
| Initial precision | BF16 weights; preserve FP32 recurrent state and explicit accumulation rules. |

Reuse LunaFlux's tokenizer, hybrid-attention, MoE, matrix, scheduling and resource
owners where their semantics match. Existing `qwen35_moe_spec` is an exact FP8
text-only profile with vocabulary 248320; its constructor does not represent
MiniCPM BF16. Do not rename the outer config, change its vocabulary, discard
vision or pass it to the dense Llama builder to make admission succeed.

## Implementation sequence

Vision preparation is owned by `model/minicpm_v47_processor`, independently of
the decoder/request owners. Its first interface accepts decoded interleaved
RGB8, plans the default 448-resolution thumbnail and up to nine detail slices,
and packs normalized BF16 `[1,3,14,sum(H*W/14)]` pixels in upstream crop order.
The geometry and placeholder fragment feed the existing canvas position owner.
Use the Transformers v5.18.0 PIL processor as the initial resize reference;
Torchvision pixel parity, compressed-image decoding, video sampling, the vision
encoder and embedding insertion remain separate follow-ups. Preparation runs
once before prefill, never in the actor loop or decoder token step.

Implemented software evidence: eight processor package tests cover geometry,
nearest-even ties, slice rows, normalization, patch order, bounds and 4x/16x
token counts. Real-tokenizer tests join the generated body and local image ID
to the existing canvas owner; the combined focused suite passes 18/18. The
offline Pillow12 C oracle matches all 12,305,664 BF16 pixel words across five
nonuniform images, including ten-crop input. A separate full PIL pipeline
reference now invokes unchanged Transformers5.18.0 preprocessing, geometry,
backend and transform methods. All twelve decoded-RGB8 cases match22,682,688
BF16 words, target sizes, grids, crop counts and output shapes, including slice
limits1/2/7/9 and the448-area slicing boundary. Its report is
`target/minicpm-v47-pil-pipeline-v2/comparison/report.json`. These sampled PIL
results do not establish torchvision parity, vision execution or model quality.
See the [processor guide](../model/minicpm_v47_processor/README.mbt.md).

The encoder now has a model-owned packed-crop layout plus generic
segmented BF16 attention primitives. Preserve 72-wide heads and the eager
reference's BF16 score, scaled-score and normalized-probability boundaries.
The existing vision conditioner's online-F32 attention and bilinear position
lookup have different semantics and cannot serve as substitutes. Reuse its
norm, biased-linear/GELU and residual operators where their contracts match.
Nearest position IDs, crop attention spans, 2x2 window order and final merger
order are prepared once per media request; AOT kernels consume integer buffers
rather than generating CUDA from request-specific geometry.

The complete AOT vision program is prepared:318 stages/25 entrypoints with all
459 vision tensor names joined to actual headers (1,105,972,064 BF16 bytes).
At16384 raw-patch capacity its eight scratch regions total405,274,624 bytes;
pixels, features and integer metadata are external. The focused foundation and
program/native-plan suite passes115/115; existing vision-conditioner reuse tests pass10/10
with the preexisting dependency's warning92 additionally excluded for that
regression command. A CPU-only compilation/probe of the generated CUDA spelling
passes ASan/UBSan for patch/window/position/mean and head72 attention cases.
That probe uses host CTA barriers and does not qualify NVCC or device execution.
Native vision weight/frame ownership and independent execution are now
implemented. At16384 patches, complete mutable storage is429,178,928 bytes in15
allocations. The scripted native driver uploads all459 tensors from sparse zero
fixtures and checks exactly1,105,972,064 transferred bytes. Both318-stage16x and
305-stage4x paths, zero-allocation queue replay, completion/cancellation/reuse,
partial startup and active close pass. The generated-C/four-stub ASan smoke
passes with prebuilt runtime/mimalloc retained and leak detection disabled.
These tests compute no GPU arithmetic. Media serving and upstream
position/numerical comparison remain open; native embedding insertion and
combined residency are implemented below. See the
[vision program guide](../integration/minicpm_v47_vision/README.mbt.md).

The embedding bridge is owned by `integration/minicpm_v47_multimodal`. Prompt
preparation renders the pinned stateless content-list template (newline-joined
parts and local image IDs), tokenizes once, and joins explicit image plans to
canvas coordinates and a causal-position-to-feature map. A generic typed BF16
word-scatter program inserts completed device features after token embedding
and before decoder prefill. Decode uses the unchanged text queue. The map and
feature count upload once per request; no feature readback or image recompute
belongs to a token step. Request queue preparation accepts explicit prepared
launch spans, keeping image-specific code out of the existing host controller.
Combined residency, local media and buffered HTTP images are implemented below.

This bridge is now implemented and checked: typed image/native-token prompts,
multi-image NaViT packing and causal feature maps feed one raw BF16 scatter
after token embedding. Prefill uses925 launches; decode stays924. The map/header
occupy16,388 bytes at context4096 and upload twice per image request, with no
feature readback or extra request-step transfer. `clear_prompt` disables stale
image mapping for later text requests. The affected-package suite passes50/50
with actual tokenizer assets. The scripted native pipeline covers three
prefill chunks, decode, zero-allocation steps, cancellation/text reuse, partial
startup and failed map-copy recovery. Its generated-C/four-stub ASan smoke
passes with prebuilt runtime/mimalloc retained and leaks disabled. The actual
generated scatter also passes the CPU ASan/UBSan source probe; no CUDA arithmetic
is claimed. v18 adds the ninth AOT module/64th entrypoint; the preceding eight
source files remain byte-identical to v17. See the
[multimodal bridge guide](../integration/minicpm_v47_multimodal/README.mbt.md).

Combined residency is now owned by `integration/minicpm_v47_image_runtime`.
It composes the existing text/vision weight loaders, frames and request driver;
both execution paths reuse `ModuleSet` for the seven decoder modules. Image
startup reads nine bounded CUBINs. `begin_image` enqueues vision asynchronously;
completion binds device features before decoder submission. The owner rejects
reuse while vision or its decoder request is pending, clears the map for text
requests and retires both queues on cancellation. One upload scratch buffer is
reused sequentially for both weight sets. `bootstrap.prepare_images` prepares
both branches from one config/tokenizer/header inspection.

At 16,384 raw patches, the combined explicit allocation payload is
71,008,215,248 bytes, including 582,463,600 mutable/output/metadata bytes;
driver/module/queue overhead is additional. The affected suite passes 63/63
with actual tokenizer assets. Scripted native and generated-C/four-stub ASan
checks cover complete execution ownership, vision-to-decoder ordering,
zero-allocation token steps, image/text reuse, cancellation/failure recovery,
partial startup and active close in either branch. ASan retains the prebuilt
runtime/mimalloc and disables leak detection; native resource counters balance.
No model arithmetic or full checkpoint upload is exercised. v19 records the
combined resident, preserves all nine v18 CUDA sources and has a nine-command
AOT dry run. Local and buffered HTTP media are implemented below; actual
upload/NVCC/Spark execution and upstream quality acceptance remain open. See the
[image runtime guide](../integration/minicpm_v47_image_runtime/README.mbt.md).

Local PNG/JPEG ingress and execution are now connected. `media/rgb8` supplies
immutable decoded frames or direct camera RGB; `integration/minicpm_v47_media`
derives image plans, prompt/layout/pixel packing together. The executable's
`inspect-media`/`run-media` modes consume explicit mixed image/text queues and
the nine-module resident. Inspect mode passes with actual config/tokenizer/
headers and labelled sparse-zero shards, with no GPU access or tensor payload
read. Its PNG/JPEG example yields 2,080 raw patches, 130 feature rows and a
180-token prompt, then reuses the decoder for text.

The affected media/processor/runtime suite passes 45/45 with actual tokenizer
assets. Native scripted/ASan coverage continues from encoded files through the
processor, multi-image vision, decoder/EOS and text reuse, with zero-allocation
token steps. The standalone decoder ASan/UBSan probe also checks RGB/alpha/
grayscale, baseline/progressive JPEG, 16-bit/size rejection and every truncated
fixture prefix; codec allocation counters balance after each call. Full native
ASan retains the prebuilt runtime/mimalloc; both probes disable leak detection. This verifies
software data flow and lifetime, not GPU model arithmetic or visual grounding.
Full JPEG/color-profile parity, video/camera SDK adapters, full checkpoint
upload and Spark/NVCC numerical/quality acceptance remain open. See the
[media guide](../integration/minicpm_v47_media/README.mbt.md) and
[executable commands](../cmd/minicpm_v47_generate/README.md).

Buffered HTTP image admission is now implemented through `serve-media` and
`Backend::with_images`. The shared exchange exposes a model-owned handler;
MiniCPM normalizes inline PNG/JPEG data URIs into native placeholders plus typed
RGB content. Generation fields remain validated by the canonical parser, which
retains the original authenticated HTTP lease through normalized admission.
The full body is bounded to the existing 16 MiB ceiling. Text/image replies
share incremental UTF-8, EOS/string stops and usage; optional system presence,
outer trim and content-list newlines follow the pinned profile. No URL/file
fetch, codec or media work enters token scheduling.

The affected suite passes 98/98 with actual tokenizer assets. Real loopback
HTTP over the scripted native driver checks PNG/JPEG replies, token usage,
no-system text reuse, 401/400/429, enqueue failure/recovery and pending-vision
timeout/disconnect cleanup. The held vision event prevents decoder submission
until completion. Generated-C/five-native-object ASan passes with the prebuilt
runtime/mimalloc retained and leak detection disabled; native resource counters
balance. The shared allocation-driver regression and release CLI build pass.
These are software protocol/ownership checks; full checkpoint upload, actual
Spark/NVCC serving and visual/numerical quality remain unverified. Video/E1
camera SDK ingress and metadata-transform parity remain separate follow-ups.

MoonRobo's real image planner client is now connected in a cross-repository
software smoke. Labeled PNG/JPEG observations cross LunaFlux's real authenticated
HTTP/media/full-program request path, a bounded scripted sampled-ID sequence
returns a complete skill proposal, and MoonRobo's real task supervisor accepts
the expected fixture reference. The one-request model fixture closes all native
resources before exit. The multi-token test hook and unchanged driver allocation
fixture pass generated-C/five-object ASan/regression checks, retaining prebuilt
runtime/mimalloc with leaks disabled. MoonRobo's focused planner/task/skills/
runner suite passes 11/11 and its release CLI builds. This is software-boundary
evidence, not GPU arithmetic, visual reasoning, actor execution or robot control.
See `moonrobo/scripts/test_lunakine_visual_planner.mbtx` and its local
`output/lunakine-visual-planner-smoke-v2/report.json` in the sibling repository.

The deployment audit fixed aggregate AOT receipt bookkeeping: successful builds
now require every requested module, rather than exactly seven. The focused
`scripts/test-minicpm-v47-aot-receipt.mbtx` regression passes for seven/eight/nine
modules, dry runs and failed compiler output using the existing explicit fake
compiler. `target/minicpm-v47-aot-receipt-test-v2/report.json` records the scope;
its dummy artifacts are not CUDA code or deployment inputs. No actual NVCC
compilation is established by this test.

The earlier local Darwin/ARM64 compiler limitation is superseded by the actual
Spark compilation and module/scatter checks recorded above. The cached CUDA
redistributables and failed temporary compiler VM are historical preparation
only. Full-model numerical/performance qualification still requires the actual
checkpoint and upstream comparisons on the target.

1. **Artifacts and contract:** pin config, tokenizer, processor, chat template,
   generation policy and tensor index; map actual tensor names and shapes.
   MiniCPM owns its outer multimodal specification and vocabulary. Extract shared
   hybrid geometry only where both families consume it. Metadata inspection is
   separate from execution readiness.
2. **Text decoder:** bind BF16 storage, exact recurrent/full-KV lifetimes,
   convolution, gates, partial MRoPE, routed/shared experts and logits. Reuse
   operators without inheriting the pinned Qwen FP8/vocabulary assumptions.
   Prove ordinary decoding before MTP/DSpark or expert-prefetch optimization.
3. **Tokenizer/chat:** admit actual assets and special tokens; implement the
   pinned instruct/thinking policy. Compare upstream token IDs and logits before
   skill quality. The small model's BOS/empty-think template is not transferable.
4. **Vision:** match slicing, resize/normalization, patches, encoder, early
   compression, image framing, visual insertion and canvas positions. Compare
   embeddings and outputs. Prepare images asynchronously; media ingress stays
   above the model-independent scheduler.
5. **Spark:** build exact GB10 `sm121` artifacts, load pinned weights and validate
   memory, logits, generation, stops/cancellation and resource balance. Evaluate
   planning and visual grounding on recorded scenes.
6. **Robot task integration:** use MoonRobo's existing planner/task contract and
   LunaKine actor. Start with text requests to this same flagship decoder, then
   add visual scene inputs. MoonRobo owns references, outcomes and robot authority.

## Implemented software and remaining work

The independent coordinate check now covers complete unpadded canvas sequences,
not just the linspace primitive. A MoonBit automation extracts the unchanged
four position/grouping methods from pinned Transformers 5.18.0, preserving their
license/header; a small offline C++/CPython adapter invokes those methods with
CPU Torch and the actual selected configuration. Corpus construction/comparison
stay MoonBit. All 154 cases match 751,830 prompt coordinates, 1,848 decode
coordinates, `rope_delta` and 32-row prompt staging. The report is
`target/minicpm-v47-canvas-upstream-v2/comparison/report.json` (CPU Torch 2.9.1).
Cases cover mixed image/video labels, adjacent/separate groups, slices, six image
geometries and 4x/16x profiles. Crop grids are inputs to both canvas implementations;
the separate full PIL pipeline reference above checks twelve geometry/pixel
cases. Masked/batched padding and physical CUDA parity remain open. No
Python/Torch or template interpreter enters production.

The template check uses the checkpoint's actual Jinja source with an
offline Jinja2 CLI. Its supported profile is optional system + one user, no
tools, no reasoning history, generation cue enabled. MoonBit's default trim
only removes four ASCII characters; the pre-fix run reproduces a vertical-tab
mismatch in `target/minicpm-v47-chat-template-v1.log`. The model adapter now owns
the pinned Python/Jinja trim rule, shared by native prompt construction and
text/image HTTP normalization. All 128 supported-profile cases pass, comparing
24,870 bytes across both paths; the report is
`target/minicpm-v47-chat-template-final/report.json`. The offline CLI uses
Jinja2 3.1.6 via jinja2-cli 1.0.1; its final source LF is removed only to reproduce
Transformers' `keep_trailing_newline=false`, as recorded in the report. No
template interpreter enters production execution. The 70 affected tests and
native ASan fixture pass (prebuilt runtime/mimalloc retained, leak detection
disabled; explicit native resources balance). Text HTTP string/list/no-system
usage matches the trimmed prompt; the text endpoint rejects image parts before
decoding. Full model logits, broader history/tools/thinking profiles and actual
CUDA execution remain unverified. The native release serving CLI also builds;
that is not NVCC compilation or GPU execution.

An independent software check now compares the selected `tokenizer.json`
against Hugging Face tokenizers 0.23.2, loaded directly by an isolated Rust test
oracle. Corpus construction and comparison stay in MoonBit; the oracle only
calls upstream encode/decode. This adds no production dependency and does not
establish upstream Jinja-template or model-logit equivalence. All 1,797 distinct
strings and 28,043 token IDs match; decoded text matches as well. The report is
`target/minicpm-v47-tokenizer-upstream-v2/report.json`; the corpus includes
multilingual text, NFC/combining marks, digit runs, whitespace, special markers,
skill JSON, optional system prompts and image placeholders.

The local numerical check now covers canvas resampling. An exploratory native
LibTorch 2.9.1 CPU probe found that exact rational nearest-even coordinates do
not always equal upstream's FP32 `linspace(...).round().long()` (for example,
15 samples over extent118 place sample7 at59 rather than58). The position owner
now preserves the FP32 step and symmetric endpoint calculation. A separate
offline ATen oracle compares the public prompt-position API; neither ATen
nor LibTorch becomes a production LunaFlux dependency. CPU parity is scoped
separately from still-required CUDA/Spark parity. The pre-fix run reproduces
58 versus59 at `target/minicpm-v47-positions-aten-v2.log`; the corrected report at
`target/minicpm-v47-positions-aten-final/report.json` passes 8,196 cases
and 533,500 coordinates on both axes. The affected positions/processor/tokenizer/
multimodal suite passes 23/23 with the pinned assets and existing warning policy.
It does not qualify full processor output.

- `model/minicpm_v47` parses the actual selected BF16 outer/text/vision config.
  `model/hybrid_moe` owns shared geometry, also consumed by the existing pinned
  Qwen profile without transferring its FP8 or vocabulary assumptions.
- `model/minicpm_v47_weights` binds all **1,152 actual tensor headers** across
  16 shards, totaling **70,425,751,648 BF16 payload bytes**. Tests include missing
  shards, duplicate tensors, wrong packed-expert axes and wrong control dtype.
- `integration/minicpm_v47` joins the actual 248144-entry tokenizer and model
  markers and provides the initial no-thinking text planner prompt/profile.
  Real tokenizer round trips pass. `tokenizer.json` uses the NFC single-digit
  regex; its `tokenizer_config.json` describes a different regex. Execution follows
  the former. The isolated tokenizers 0.23.2 reference matches all 28,043 IDs and
  decoded text over 1,797 distinct corpus strings. The separate Jinja oracle
  verifies 128 initial-profile cases. Broader chat profiles, model logits and
  arbitrary-Unicode equivalence remain unqualified.
- `model/gated_delta_reference` implements FP32 scalar-decay state and causal
  convolution with grouped heads, reset and continuation. This is a numerical
  reference, not a production fallback. Convolution can preserve the BF16
  projection boundary from upstream's unfused fallback; a regression fixture
  distinguishes it from FP32-through-SiLU fusion.
- The existing typed recurrent CUDA lowering now accepts grouped Q/K heads,
  scalar decay and packed input row strides. Ordinary arithmetic is reused.
  Chunk lowering rejects the new layouts; it does not silently reinterpret them.
  Source tests pass, but compilation, GPU comparisons and performance are pending.
- `cmd/minicpm_v47_prepare` joins source shard-size metadata, headers and the
  tensor index, then exports seven text-path CUDA files, an unbound vision module and a metadata-only
  preparation report. It loads no weight payload or device.
- `model/minicpm_v47_operators` selects the decoder/query/key centered norms,
  packed query/gate views, BF16-stage SiLU-gated recurrent norm, full-softmax
  top-eight router and recurrent controls, alongside convolution/delta plans.
  The command consumes this builder; generic IR and kernel packages carry no
  checkpoint identity or model-family branches.
- New normalization, routing and control source generators lower typed numeric
  expressions. Decoder gamma addition stays FP32; recurrent normalized and
  weighted values each round to BF16 before the FP32 SiLU gate. Routing retains
  full 256-expert FP32 softmax before selection/renormalization and BF16 output
  weights. Controls preserve BF16 beta and FP32 unclipped logarithmic decay.
  Offline references test these boundaries, IEEE BF16 rounding, packed views,
  extreme routing logits and reusable scratch. They do not certify GPU execution.
- Routed experts reuse `ExpertMlpPrecision` and its existing three-stage CUDA
  lowering, with explicit `Bf16EachAdd` combination in ascending expert-ID order.
  Full bank ownership and local down publication are required for this mode;
  tensor partitioning rejects it. Existing F32-combine plans retain their defaults.
  Router weights still round to BF16, then widen exactly into the existing F32
  route-weight ABI. This does not remove the BF16 rounding stage.
- The shared expert reuses dense staged BF16 SwiGLU. Router/shared-gate projections
  reuse matrix lowering; a small generic operator rounds the shared sigmoid,
  shared product and final routed/shared addition. A composed offline test covers
  routing through both expert branches and final output. Another fixture proves
  that BF16 each-add `256 + 1 - 256` is zero, while F32 combination gives one.
- `integration/minicpm_v47_moe.bind_expert_weights` binds packed checkpoint axes
  to the existing streaming expert uploader. A real safetensors byte-transfer
  test with three-byte scratch checks all destination matrices and padding.
  Actual selected-model headers separately verify byte ranges across early,
  full-attention and final layers. Preparation emits all 30,720 startup ranges
  in `expert_uploads.json` without downloading the payload.
- The MoE integration now owns one immutable nine-stage program and a native
  frame binder. The program supplies combined `moe.cu`, stage ABIs and workspace
  layout to both the runtime and preparation command. The frame loads AOT
  functions and binds borrowed weights/input/output plus one caller-owned arena;
  its launches feed the existing ordered executor. It owns only its function
  handles. Close the executor before the frame, then release borrowed resources.
  Tests check ABIs/layout/source, but actual device prepare/execute/cleanup is
  still untested. The report explicitly marks native frame execution false.
- Full-attention plans now bind all four BF16 projection shapes to actual headers,
  centered Q/K normalization, split-half rotation of the first 64/256 channels,
  interleaved 11/11/10 T/H/W frequencies and head-interleaved output gating.
  Generic typed rotary/gate lowering preserves BF16 coefficients, products,
  sigmoid and publication rounds. CPU fixtures distinguish intermediate product
  rounding and verify the unchanged tail and strided gate view.
- `integration/minicpm_v47_attention` reuses the existing paged-attention compiler
  for width-256 prefill/decode. The ABI now admits that width; specialized
  lowerings keep their own restrictions. The declared unqualified sm121 software
  profile emits scalar schedules 100/400 with no tuning records or hardware
  observations. At 4096 tokens, each full layer's K/V planes occupy 8,388,608 bytes.
  Its full-attention program now owns 13 ordered stages per phase, including
  bit-preserving QKV packing and the existing paged KV writer before read-only
  attention. The native frame binds that same program; actual CUDA preparation,
  execution and cleanup qualification remain pending.
- A real compiler coverage gap is fixed: when an unmeasured static winner has no
  CUDA emitter, the backend filters emitted legal variants and the common compiler
  reuses its resource scoring. Unsupported explicit/measured selections still
  fail. Tests cover width-256 source construction, selection coherence and
  preservation of resource constraints. This is not a performance qualification.
- `model/minicpm_v47_positions` now owns compacted prompt canvas coordinates and
  decode `rope_delta`. It handles thumbnail/detail grids, slice marker corners,
  row newlines, image/video groups and per-call spatial merge. Allocation-free
  chunk/decode staging keeps causal cache addresses separate. The actual
  tokenizer/text template now prepares tokens and positions through one bridge.
  Analytic position fixtures and the FP32 rounding regression pass. The offline
  ATen CPU oracle matches 533,500 coordinates through the public prompt API;
  the separate full-canvas oracle also matches 751,830 coordinates over 154
  unpadded sequences against unchanged upstream methods. The full PIL pipeline
  reference separately checks geometry/pixels on twelve images. Masked/batched
  padding and physical CUDA parity remain open.
- `integration/minicpm_v47_recurrent` now owns all nine recurrent-branch stages
  and its native frame: QKV/z/a/b projections, convolution, controls, grouped
  recurrence, gated normalization and output projection. It reuses the existing
  numeric kernels and supplies the same source/stage/region plan to preparation.
  K/V pointers are views into convolved packed rows; no second packed bank is
  introduced. Persistent BF16 history and FP32 state stay outside its reusable
  1,845,248-byte workspace. Actual device execution remains untested.
- `integration/minicpm_v47_decoder` now connects all 40 prepared branches/MoEs
  with centered input/post-attention norms and two BF16 residual updates per
  layer. Its 920-launch phase queues borrow three activation banks and four
  disjoint persistent state arenas. Attention/recurrence share one scratch arena;
  MoE keeps its own shared arena so the startup expert map survives other work.
- `integration/minicpm_v47_text` now wraps that stack with the existing text I/O
  source/frame owners: embedding, selected-row centered final norm, vocabulary
  projection and greedy selection. Embedding and language head remain untied.
  One selected output row requires 996,672 bytes of preallocated output workspace;
  each complete text phase has 924 launches. The generic text plan gains an
  explicit normalization weight law, defaulting to direct gamma for existing
  callers and selecting FP32 one-plus-gamma here.
- The native ABI fixture prepares that complete text owner with synthetic
  weights and the existing software driver. It checks 776 function handles plus
  two output allocations, argument regions, 2,772 submitted launches, reuse/reset
  and balanced explicit cleanup. Queue execution allocates no heap objects or
  resources and performs no copies/readbacks. Caller metadata staging is outside
  that measurement. This validates bindings/lifetimes, not GPU arithmetic.
- `integration/minicpm_v47_weights` now binds the text checkpoint to final
  allocations and owns bounded startup transfer/cleanup. The pinned headers map
  693 source tensors into 653 allocations totaling 69,319,779,584 bytes; vision's
  459 tensors remain unloaded. Ordinary tensors preserve exact BF16 bytes and
  rank; expert slices reuse the shared transfer session/bank layout. One caller
  scratch buffer serves all allocations. The real-header test uses sparse zero
  payload holes, while a small byte fixture checks actual reads/scatter/padding
  and session cleanup. Real CUDA weight upload remains untested.
- `integration/minicpm_v47_text.Storage` owns the complete mutable allocation
  set. It admits 152,271,612 bytes before allocation and provides typed state,
  buffer, metadata and token objects to the frame. The existing native ABI probe
  now prepares this owner too, verifies 22 allocations and pre-allocation budget
  rejection, and closes it after the forward frame. Request metadata/reset
  initialization remains the caller's responsibility.
- `integration/minicpm_v47_request` now supplies that initialization and a
  single-flight greedy request driver over the same full flagship text frame.
  The host owner stages chunked prefill, decode inputs, canvas/causal positions,
  exact context limits and final-row selection. The native owner prepares two
  reusable queues, uploads static page/slot metadata once, and commits progress
  only after completion. A 33-token scripted request adds 3,696 launches to the
  original 2,772-launch queue-only probe. Complete request steps allocate no
  heap/resources and perform ten metadata uploads, with one four-byte readback
  only on producing frames. EOS, token/context limits, cancellation/new-request
  reset, invalid samples and failed-enqueue recovery pass. An ASan smoke also
  passes with prebuilt runtime/support libraries and mimalloc retained and leak
  detection disabled. This validates control flow and resource lifetimes;
  scripted IDs do not establish GPU arithmetic or model token agreement.
- [`integration/minicpm_v47_runtime`](../integration/minicpm_v47_runtime/README.mbt.md)
  now composes complete resident resource ownership. `Execution` owns seven AOT
  modules, a stream, storage/frame and request queues while borrowing weights.
  `Resident` joins it to the existing streaming weight owner; both retain
  cleanup authority after partial preparation. The explicit initial allocation
  budget is 69,473,047,868 bytes, with driver/module/queue overhead additional.
  The seven-CUBIN reader runs only at startup. A native software-driver fixture
  checks pre-allocation admission, partial startup cleanup, repeated requests
  and close with active work. The fresh resident joins the actual sparse-header
  checkpoint plan, but its full weight upload is not exercised. The extended
  ABI fixture and ASan smoke pass; there is no real-CUDA/whole-model claim.
- [`integration/minicpm_v47_bootstrap`](../integration/minicpm_v47_bootstrap/README.mbt.md)
  joins actual config/tokenizer/index/shard headers to immutable text and weight
  plans. It checks index/header ownership without tensor-payload reads. Source
  export and resident startup now use the same declared attention/text builders;
  all seven v15 generated CUDA sources match v14. Requests tokenize before CUDA
  and explicitly select the bounded greedy text profile.
- [`cmd/minicpm_v47_generate`](../cmd/minicpm_v47_generate/README.md) now provides
  `inspect` and `run` entrypoints. Inspection and the release build pass locally.
  Run mode loads one resident session for repeated requests, then writes token
  IDs, decoded text, usage and finish reasons after each request. Truncated UTF-8
  uses replacement text and an explicit validity flag while preserving exact IDs.
  Its inspect/run modes write files and never publish robot commands. The inspection smoke uses
  actual config/tokenizer/headers with sparse zero payloads and no GPU access;
  actual run mode/whole-checkpoint upload remain untested.
- [`service/session_chat`](../service/session_chat/README.mbt.md) now reuses the
  existing bounded HTTP parser, bearer policy, OpenAI ingress, canonical events
  and buffered renderer around a single prepared-session callback. It handles
  repeated requests, capacity, timeout, disconnect and cancellation retirement.
  There is no new JSON/HTTP parser or model/robot dependency in the transport.
- [`integration/minicpm_v47_chat`](../integration/minicpm_v47_chat/README.mbt.md)
  tokenizes the canonical rendered prompt once, reserves output context and
  drives the existing runner. Incremental UTF-8 and token/string stops use the
  existing output owner. Native-driver HTTP tests use the actual tokenizer and
  scripted IDs through four successful 924-launch requests, plus stochastic
  rejection and enqueue-failure/recovery. Input usage agrees with the text
  template path. This is lifecycle/binding evidence, not model logits.
  `cmd/minicpm_v47_generate serve` exposes buffered JSON and streaming SSE;
  `serve-media` supports the same delivery modes with inline PNG/JPEG inputs.
  Streaming reuses the canonical renderer, delivers deltas before completion,
  and retains event/output views until bounded writes finish. Timeout or
  disconnect after headers cancels model work without a second HTTP response.
  All67 affected tests pass (`target/minicpm-v47-streaming-focused-test.log`),
  and the updated native fixture passes ASan
  (`target/minicpm-v47-streaming-native-asan-v2.log`, prebuilt runtime/mimalloc
  retained, leaks disabled and explicit native resources balanced). It covers
  text/image SSE, real tokenizer usage and cancellation after Accepted while
  vision is pending; sampled IDs remain scripted, with no GPU arithmetic.
  real checkpoint serving and model quality remain untested.

Deployment still needs actual executable checkpoint upload/generation,
continuous serving and numerical/token comparisons on the target.
Vision needs slicing/resize, patch/position embeddings, encoder, early window
merger and output merger; its encoder uses tanh GELU while the final merger's MLP
uses ordinary GELU. These are model semantics, not interchangeable approximations.
Do not use the existing direct-gamma RMSNorm or sigmoid-only gated norm unchanged.

## Preparation and evidence

From MoonRobo, download only selected metadata and the tensor index:

```sh
moon run scripts/prepare_minicpm_v47_assets.mbtx NEW_ASSET_DIRECTORY
moon run scripts/inspect_minicpm_v47_headers.mbtx NEW_ASSET_DIRECTORY/model.safetensors.index.json NEW_HEADER_DIRECTORY
```

The fetcher uses pinned `/resolve` URLs so Git LFS files are downloaded rather
than saved as pointer text. Header inspection uses bounded HTTP ranges; it does
not download the roughly 70 GB weight payload.

From LunaFlux:

```sh
LUNA_MINICPM_V47_ASSET_DIR=ABSOLUTE_ASSET_DIRECTORY moon test --target native --warn-list '-79-29-25-20' integration/minicpm_v47
moon run cmd/minicpm_v47_prepare --target native -- ASSET_DIRECTORY HEADER_DIRECTORY NEW_PREPARATION_DIRECTORY
```

The generated CUDA files and connected forward plans are not a deployable model
artifact. The report explicitly records that full-model execution, CUDA
compilation and device qualification are false. For one 4096-token sequence,
logical state budgets are 83,886,080 bytes of BF16 full-attention KV, 62,914,560
bytes of FP32 recurrent matrices, and 1,474,560 bytes of BF16 convolution history,
before workspace, pages and padding. Source artifacts and labels do not certify
downloaded weight contents.

The expanded model/protocol native suite passes **583/583**, including actual header
fixtures, actual resolved tokenizer assets, precision-IR, BF16 operator references
and source regressions, reused matrix/expert/dense kernels, paged-attention
compiler/source coverage, fragmented-page cache continuation, BF16 bit-preserving
packing, canvas coordinates/continuation, decoder/text/request/resident/bootstrap
composition, canonical HTTP/auth/OpenAI handling, cancellation/reuse,
bounded request admission, UTF-8 result handling and the
real sliced streaming transfer. MoonRobo's
planner/task/runner suite passes **8/8** with warnings
denied; the planner now sends explicit JSON Content-Length through per-request
headers rather than the convenience client's chunked default. LunaFlux validation uses the existing compiler-migration warning
exclusions `-79-29-25-20`; no module-wide warning settings were changed.
The successful preparation report is locally retained at
`target/minicpm-v47-prepared-v19/preparation.json`. Its recurrent packed QKV inputs have
8192 BF16 elements per row, with query/key/value pointer offsets 0/4096/8192
bytes; recurrent output stays packed independently.

The seven generated text runtime modules contain38 entrypoints covering recurrent delta,
causal convolution, centered/gated norms, routing/controls, expert branches,
residuals and text ingress/egress. Their v16 source bytes match v15. The additional
unbound vision module contains25 entrypoints, giving eight files/63 entrypoints
in the v17 preparation. v18 adds `image_scatter.cu` as the ninth module and64th
entrypoint. The existing text Resident still consumes its seven modules; a
combined image-capable resident, local media executable and buffered HTTP images are implemented. v19 records the resident and preserves all nine v18 CUDA sources.
Normalization metadata points into those shared modules; preparation no longer
emits standalone Q/K norm copies that the runtime does not load.
The MoE subset has nine ordered kernels: router projection, softmax/top K,
three routed-expert stages, two shared-MLP stages, shared-gate projection and
shared combination. Initial norm/router reductions are ordered FP32 programs,
with thread-zero baseline implementations. Expert projection lowering uses its
existing warp reduction; shared dense reference uses ordered dots. They preserve the
declared storage/stage boundaries, but are not bitwise-equivalence or performance
claims against upstream CUDA reduction trees. Routing ties use lower expert IDs;
upstream `topk` tie order is unspecified. Compare non-tied outputs and task quality
before optimizing or admitting a different floating program.

Each routed bank is 1,610,612,736 bytes, totaling 64,424,509,440 bytes across 40
layers. Startup slices copy directly into that destination layout using bounded
scratch. These bytes replace the routed source representation in memory; there
is no planned persistent second full routed bank. The complete checkpoint and
other workspaces still require whole-model capacity admission.

All nine MoE entrypoints now share `moe.cu`. Its non-aliased workspace is
1,756,224 bytes at 32 rows, reused serially across decoder layers. The native
frame borrows it; each layer does not require a separately persistent scratch
allocation. Source export, immutable command bindings and workspace arithmetic
are verified independently of actual CUDA execution.

`recurrent_branch.cu` contains nine entrypoints around five matrix projections,
including convolution, controls, grouped recurrence and gated normalization.
At 32 rows its non-aliased scratch occupies 1,845,248 bytes. Convolved K and V
are offset views into the same QKV bank, with row stride unchanged. Persistent
state at one slot is 49,152 bytes of BF16 history and 2,097,152 bytes of FP32
matrices per recurrent layer. The native frame borrows offset views into state
arenas and owns only nine function handles/launch descriptions. Both stateful
kernels consume the same caller-owned CSR, exclusive slots and reset flags.
The enclosing decoder assigns disjoint layer views; runtime allocation ownership
and actual execution qualification remain open.

Full attention adds `attention_common.cu` with 12 entrypoints, plus
`attention_prefill.cu` and `attention_decode.cu`. Compile these as separate
translation units/modules. Each phase uses 13 ordered launches and a
1,740,800-byte non-aliased scratch arena, reusable serially across layers.
Persistent K/V planes remain outside that arena; the native frame can borrow
offset views into the existing multi-layer cache. It owns only 14 function
handles and launch descriptions. Close enclosing executors before the frame,
then release borrowed modules/weights/metadata/buffers. Input is already
decoder-normalized; residual addition belongs to the enclosing decoder.
Preparation and runtime consume the same program, rather than maintaining two
stage lists. The report distinguishes frame implementation from physical
execution, which remains false.
Rotary coordinates use model position capacity 262144 and are distinct from
4096-token causal/cache capacity. The canvas sequence builder must supply correct
T/H/W positions; visual inputs must not silently fall back to 1D positions.
The reused attention path uses online FP32 softmax over BF16 Q/K/V, not the eager
fallback's materialized BF16 score/probability program. Logit and task-quality
comparisons on the target remain required.

`decoder_envelope.cu` adds common centered norm/residual entrypoints. One hidden,
normalized-input and branch-output bank (131,072 bytes each) serve the whole
stack. Separate shared branch/MoE scratch totals 3,601,472 bytes, preserving the
MoE expert map rather than overwriting it between layers. These are whole-stack
workspaces, not per-layer allocations. Persistent state slices appear in the
preparation report in model order.

`text_io.cu` contains four existing text-I/O stages. At one selected output row,
normalization uses 4,096 bytes and logits use 992,576 bytes. BF16-rounded logit
values are widened exactly into F32 storage; this does not authorize unrounded
F32 logits. The source/header binding retains two separate 1,016,397,824-byte
embedding/head tensors. Request state/position staging and queue completion now
exist in the request adapter. Autoregressive serving still needs resident
real executable upload/generation and device qualification. Buffered network
publication is implemented, with streaming/continuous admission still open.

Reproduce the focused LunaFlux checks:

```sh
moon test --target native --deny-warn --warn-list '-79-29-25-20' model/minicpm_v47_positions model/streaming_safetensors integration/minicpm_v47_weights integration/minicpm_v47_recurrent integration/minicpm_v47_decoder integration/minicpm_v47_text integration/minicpm_v47_request integration/minicpm_v47_runtime integration/minicpm_v47_bootstrap integration/minicpm_v47_chat cmd/minicpm_v47_generate service/session_chat service/openai_compat service/http1 service/incremental_output service/api_auth integration/text_io_frame kernels/text_io_execution_cuda_source compiler/hyper_connection_physical_ir kernels/decoder_hyper_connection_cuda_source
LUNA_MINICPM_V47_ASSET_DIR=ABSOLUTE_ASSET_DIRECTORY moon test --target native --deny-warn --warn-list '-79-29-25-20' model/hybrid_moe model/minicpm_v47 model/minicpm_v47_weights model/minicpm_v47_operators model/bf16_operator_reference model/gated_delta_reference model/qwen35_moe_spec tokenizer/json_reader integration/minicpm_v47 integration/minicpm_v47_moe integration/minicpm_v47_attention compiler/precision_ir compiler/grouped_matrix_ir kernels/recurrent_delta_execution_cuda_source kernels/recurrent_block_execution_cuda_source kernels/causal_convolution_execution_cuda_source kernels/rms_norm_execution_cuda_source kernels/softmax_topk_execution_cuda_source kernels/gated_delta_control_cuda_source kernels/shared_expert_execution_cuda_source kernels/selected_expert_mlp_cuda_source kernels/dense_mlp_execution_cuda_source kernels/tiled_matrix_cuda_source engine/precision_materialize kernels/luna_cuda_precision kernels/rotary_kv_cuda_source kernels/multi_axis_rotary_cuda_source kernels/pointwise_gate_cuda_source kernels/packed_qkv_cuda_source kernels/luna_cuda_pointwise_aot kernels/luna_attention_tile_compiler kernels/luna_cuda_attention_tile_aot kernels/luna_cuda_attention_tile_source
```

The native binding/lifetime fixture is separate from numerical tests:

```sh
moon run tests/minicpm_v47_decoder_abi --release --target native --deny-warn --warn-list '-79-29-25-20'
```

With `LUNA_MINICPM_V47_ASSET_DIR` set, this fixture also runs the actual-tokenizer
chat adapter across loopback HTTP. Its local successful log is
`target/minicpm-v47-chat-abi.log`, with the extended sanitizer smoke at
`target/minicpm-v47-chat-asan.log` (prebuilt runtime/support libraries and mimalloc
retained, leak detection disabled). The reused
driver's original allocation fixture also passes in its required release mode;
see `tests/minicpm_v47_decoder_abi/README.md` for commands and scope.

The executable inspection smoke is also separate and never opens a GPU:

```sh
moon run scripts/test-minicpm-v47-bootstrap.mbtx /absolute/path/lunaflux ACTUAL_ASSET_DIRECTORY NEW_FIXTURE_OUTPUT
```

The local successful plan is
`target/minicpm-v47-bootstrap-smoke-v1/inspection/plan.json`. Its shard files
have sparse zero payload holes; they must not be used for inference. Real
inspect/run commands and result schemas are in the executable guide above.

Generate an offline AOT command receipt, then compile on the supplied Spark
toolchain when available:

```sh
moon run scripts/build-minicpm-v47-aot.mbtx PREPARATION NEW_DRY_RUN_OUTPUT
moon run scripts/build-minicpm-v47-aot.mbtx PREPARATION NEW_COMPILE_OUTPUT --compile /absolute/path/to/nvcc
```

The helper targets the declared `sm_121` profile, keeps the translation units
separate and disables implicit multiply/add contraction. New preparations add
the eighth vision and ninth scatter modules; old seven/eight-module preparations
remain accepted. These are documented
[NVCC options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
Default mode invokes no compiler. Compile mode records per-module exit status
and nonempty output size, stops on failure and always leaves execution/numerical
qualification false. No CUBIN payload hashes are computed. Local dry-run receipt
`target/minicpm-v47-aot-multimodal-dry-v1/build.json` contains all nine commands; the historical intentional
failing-tool receipt at `target/minicpm-v47-aot-failure-v2/build.json` retains
`cuda_compiled: false`. No actual NVCC compilation has run here.

The initial BF16 weight payload is roughly 70 GB. Plan for the full expert set,
not only the 3B active parameters. Spark's 128 GB unified memory makes this a
plausible capacity target; measured OS, visual prefill, workspace and cache budgets
still determine admission. Begin with a bounded 4096-token task context and small
image batches. Do not silently reduce precision or expert participation.

Current software evidence covers the selected-model pieces above as well as the
actor, task HTTP, planner transport and shadow execution. MiniCPM5-2B's tests remain
[separate compatibility evidence](MINICPM5_SPARK.md). **The selected flagship does
not yet execute in LunaFlux.** Its hybrid MoE/vision path remains implementation
work, not merely hardware validation.

LunaFlux owns multimodal inference; LunaKine owns Ascend actor inference;
MoonRobo owns LibTorch/CUDA training and E1 control. See the
[two-host runbook](../../moonrobo/deploy/lunakine/README.md).
