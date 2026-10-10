# MiniMax H3 code integration — 2026-09-26

This is the current integration inventory. Earlier entries in
[the execution workstream](MINIMAX_H3_EXECUTION.md) are chronological records,
not a current list of missing modules. This round implements numerical code
and native orchestration; it does not claim a successful GPU generation.

## Implemented composition

| Boundary | Concrete implementation |
| --- | --- |
| Prompt presentation | `integration/minimax_h3_prompt`: verbatim text, keyframe and ordered reference presentations, token IDs, modality tags and visual placement; no chat template |
| Text conditioning | Config-derived Qwen3-VL text encoder, 50 layers, hidden-state selection, optional visual/deepstack injection, checkpoint upload and explicit request owner |
| Visual conditioning | RGB preprocessing, patch embedding, 27 vision blocks, position interpolation, rotary, main merger and three deepstack mergers |
| Reference media | Audio encoder and tiled VideoVAE encoder, immutable operation/liveness plans, CUDA source and explicit request owners |
| Denoising | Input projection, text refiner, reference/context row assembly, timestep embedding, all 50 DiT blocks, both output heads and paired RF updates |
| Output audio | Packed-latent bridge, complete AudioVAE frontend, upsampling/AMP stack and waveform tail |
| Output video | Latent bridge, full transformer decoder, overlapping temporal/spatial tiles, inverse normalization, crop and canonical F32 pixels |
| Request lifecycle | `integration/minimax_h3_request`: prepared producers → conditioning assembly → scheduled denoising → video decode → audio decode; outputs publish only after both decoders complete |
| Startup | Component-by-component checkpoint materialization/upload, explicit weight leases and bounded cumulative device-weight accounting |
| Offline source | Independent CUDA translation units with symbols, geometry and source hashes; schedule-derived timestep variants rather than a fixed variant list |

The request facade consumes prepared owners. It is not a one-command checkpoint
loader or an HTTP endpoint. A caller still provides admitted modules, device
context/stream, memory budgets, weights and input media. Source export does not
compile or admit those modules. These distinctions are integration boundaries,
not a substitute for executable kernels.

## Checkpoint-to-request bootstrap — 2026-10-10

The remaining full-request work connects actual checkpoint components to those
prepared owners. Startup retains each model-owned packing next to its allocation;
typed denoiser borrowing resolves input, timestep, output, context, refiners and
all DiT layers without reconstructing fixed allocation-index tables in a CLI.
This is preparation-time metadata selection, not payload validation or hashing.
All consumers retain an explicit weight lease until their queues are drained.

Real text encoding and standalone audio/video decoding have executed on Spark,
but they do not prove joint generation. Full execution must consume genuine
encoded conditioning, initialize explicit noise, run every scheduled prediction
and both RF updates, then feed the resulting latents to both real decoders.
Reference-media inputs must be encoded; fabricated hidden conditioning is not an
acceptable substitute. Component weights, modules and scratch have separate
lifetimes and must fit the cumulative per-host memory budget.

`PredictionSources` is the common immutable geometry used by both AOT export
and program bootstrap; it preserves the existing CUDA bytes and symbol names.
`PredictionPrograms` prepares all schedule-derived variants once and preflights
the sum of their exact block/timestep/output workspace layouts before touching
the device. A partial prepare retains each program for reverse cleanup. It
borrows modules and does not compile, hash or load weights during a step. Input,
metadata, rotary tables and other component allocations remain separately
budgeted. This bridge does not by itself establish full-model execution.

## Functional/effect boundary

Model/configuration parsing, sequence layout, schedules, row maps, operation
graphs, packing and workspace plans are preparation-time values. Source lowering
consumes those values. CUDA-specific instructions remain in source/backend
packages. Runtime effects are explicit owners: allocation, upload, queue
submission, completion and deterministic release.

`PreparedStage` erases a concrete frame's type once at preparation. It does not
build launch lists or materialize weights during `advance`. Generic execution
and joint-diffusion lifecycle packages do not import MiniMax or media codecs.

Every denoising evaluation has its own timestep values. Row-index tables are
shared only when their equality/order pattern is identical. The source exporter
enumerates the distinct timestep cardinalities from the complete schedule.
Future-destination borrowing permits startup binding, but is not a claim that
the producer has already executed.

## Correctness and lifecycle changes

- BF16 learned-position interpolation follows the pinned reference's rounding
  after coefficients, products and ordered additions, rather than one F32 sum.
- Video inverse normalization preserves reference subtraction/division order,
  and the output path performs the final clamp and crop.
- Audio output preserves the reference's rounded latent duration. At 124 video
  frames, 207 latent frames decode to 165,600 samples per channel; the requested
  duration's 165,333 samples must not be used to reject or silently trim this.
- Request cancellation is sticky; failed stages cannot be replayed. Reverse
  cleanup retains its cursor after a failed drain or release, including partial
  preparation. A release retry does not drain a partly released owner again.
- Every supplied decoder and conditioning owner is retained before fallible
  denoising queue construction; consumers close before producers.
- Plan equality alone is insufficient to connect device buffers. Startup joins
  must also check the actual latent allocation and region, including decoder
  phase, rather than accepting a same-shaped different request.

## Remaining execution qualification

Native tests cover planning, source structure, metadata, lifecycle failures,
packing and reference fixtures. They do not execute generated CUDA. Several new
encoders/decoders are correctness-oriented scalar CUDA lowerings; no performance
equivalence to an optimized H3 engine is asserted.

The next physical boundary is to compile the exported modules, bind a real
checkpoint, compare intermediate tensors and final outputs against the pinned
reference, then run sanitizer, cancellation/release and peak-memory tests. Only
after that should full-request performance be measured. No GPU work was run in
this implementation round.

Media inputs are decoded numeric arrays with explicit grids/timestamps. Container
decoding, audio resampling and file output muxing are outside these numeric
adapters. Request bootstrap/serving must not silently fabricate conditioning,
noise or unsupported media. Reference posterior sampling accepts explicit noise;
matching another runtime's RNG byte stream requires its corresponding input.

Reference encoder startup retains a lightweight weight manifest after upload,
not its host tensor payloads. Its current segmented VAE allocation is separate
from packed decoder weights; both copies count against the explicit device
budget. Program workspaces, AOT modules and normalized audio-weight caches have
their own owners and budgets and must also be included in deployment sizing.

The supported checkpoint is guidance-distilled. This inventory does not add
general multi-branch classifier-free guidance to the sequence engine.

## Local verification

The final affected-package native run passed **231/231 tests**, including
device-owner identity, partial cleanup retries, decoder binding state, the
124-frame audio-duration regression, reference encoder weight snapshots, prompt
presentation and deterministic complete source export. Formatting and generated
interfaces were refreshed for the affected packages.

Native checks/tests use `--deny-warn --warn-list '-79-20-92-29-25'`, the existing
toolchain-migration exclusions. This is not a claim that an unfiltered
warning-denied repository build is clean. CUDA compilation and GPU tests were
not run.

The full repository test attempt did **not** complete successfully. Parallel
tensor-parallel-worker edits failed compilation in
`engine/tensor_parallel_device_worker/remote_watchdog.mbt`: the build could not
resolve `ReusableMonotonicRead` or `MonotonicClock.prepare_poller`. Those files
were not changed by this H3 work. The 231-test affected-package result must not
be presented as a full-repository pass.
