# MiniCPM-V 4.7 vision program

`Program` constructs the selected model's complete ordered vision program and
one AOT CUDA module. It reuses generic biased-linear/GELU, LayerNorm and BF16
residual sources. Generic segmented attention, patch gather, absolute-position
lookup and row permutations add the missing semantics. No model-specific
branching enters the scheduler or device backend.

Native ownership is split into a bounded streaming `WeightPlan`/`Loaded`,
mutable `Storage`, borrowed-buffer `Frame`, and a single-flight `Execution`.
Weights upload once into final allocations through the existing shared transfer
session. Storage preallocates pixels, integer metadata, eight scratch regions
and final features; Frame resolves its 25 function handles/318 launches once.
Execution owns a module, stream and ordered executor, stages already-prepared
media outside the decoder tick, and retires/cancels work before reuse/close.
Feature buffers stay on device for the later text embedding bridge. None of
these owners imports camera/robot/HTTP logic or adds token-step file/hash work.

```mbt check
///|
async test "prepare the full vision stack without a device" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let program = @minicpm_v47_vision.Program::new(
    spec,
    maximum_patches=16384,
    maximum_crop_patches=4096,
  )
  assert_eq(program.stages().length(), 318)
  assert_eq(program.entries().length(), 25)
  assert_eq(program.weight_names().length(), 459)
  assert_eq(program.workspace_bytes(), 405274624L)
  assert_eq(program.storage_bytes(), 429178928L)
}
```

The 16x profile has 318 launches: patch unpack/projection/position, all 27 encoder
blocks, the early merger after block6, post norm, and the final merger/projector.
The 4x profile has 305 launches and 443 active weights, omitting 16 early-merger
tensors. Weight names join the existing actual-header vocabulary; all 459 vision
tensors contain 1,105,972,064 BF16 bytes in the default profile.

Eight nonoverlapping scratch regions hold seven1152-channel banks and one4304
bank, sized at uncompressed patch capacity. Early17216-channel MLP activations
at quarter rows fit the latter exactly. Patch projection temporarily uses the
norm bank; early and final permutations reuse the attention bank. Native frame
binding preserves stage order and these lifetimes.

At 16384 raw patches, scratch is 405,274,624 bytes. Storage also owns pixels,
features and integer metadata, totaling 429,178,928 bytes in 15 allocations.
Together with the vision weights, the known allocation payload is 1,535,150,992
bytes, excluding module/stream/queue overhead. The text session is separate.
Counted primitives consume the current row count and
do not read uninitialized capacity padding. The existing default primitive
source stays unchanged when `counted=false`.

The attention numeric profile follows upstream eager BF16 staging: BF16 dot
output, BF16 scaled scores, global F32 softmax, BF16 normalized probabilities,
F32 value accumulation, then BF16 output. The generated implementation uses a
128-thread query/head CTA and a bounded shared score array. Its reduction order
and transcendental implementations need physical/library comparison; matching
stage boundaries does not prove bitwise model equivalence or performance.

`cmd/minicpm_v47_prepare` now exports `vision_encoder.cu` and reports its25
entrypoints,318 stage ABIs and459 actual header bindings. The AOT helper includes
this eighth module when the vision section is present, and still accepts older
seven-module preparations. Native `Loaded`/`Storage`/`Frame`/`Execution` are
implemented and independently callable. **The text Resident does not yet bind
the vision module.** Native embedding insertion is implemented in the
[multimodal bridge](../minicpm_v47_multimodal/README.mbt.md). Combined residency,
media requests and actual checkpoint execution remain pending.

`Input::new` joins prepared pixels and layout, rejects capacity/crop/profile
mismatches and encodes immutable integer packets before native submission.
`Input::from_images` accepts several processed images, including all thumbnail/
detail crops, concatenating NaViT columns in prompt order. The one-image
constructor reuses its immutable pixel payload.
`Execution::submit` uploads those packets, then enqueues its prepared frame;
`poll`/`wait` make `features()` available without a host readback. Default 16x
uses 13 input copies; 4x skips three unused window packets and uses 10.
There is no concurrent request admission on one execution owner.

`FeatureView` borrows BF16 `[rows,2048]` storage until the next submit, cancel or
close. A consumer must finish using it before reusing this owner. Close
Execution before Loaded, then close their borrowed context. Active close and
cancellation retire the queue before releasing functions/buffers/modules. Failed
startup retains acquired resources for retryable close. Only count headers are
zeroed during startup; capacity padding is never read by the counted stages.

The extended native fixture uploads exactly 1,105,972,064 bytes from sparse zero
fixtures into all 459 final allocations, then binds/replays the real frame.
This is loader/ABI evidence, not checkpoint arithmetic. It covers both
compression profiles, budget rejection before resource creation, mutable/weight
alias rejection, partial startup, enqueue failure/reuse, cancellation and active
close. Queue replay allocates no heap/native resources; media preparation is
outside that measurement. Fake resource counters balance.

Run the full native ASan smoke with actual tokenizer coverage, from LunaFlux:

```sh
LUNA_MINICPM_V47_ASSET_DIR=RESOLVED_ASSETS moon run scripts/test-minicpm-v47-native-asan.mbtx ABSOLUTE_REPO MOON_ROOT NEW_OUTPUT
```

It instruments generated MoonBit C and the four fake-driver objects, retains
prebuilt runtime/support libraries and mimalloc, and disables leak detection.
It does not establish physical CUDA sanitizer, GPU leaks, numerics or latency.

For software-only source checking:

```sh
moon run scripts/test-minicpm-v47-vision-source.mbtx ABSOLUTE_REPO PREPARED_DIRECTORY NEW_OUTPUT
```

The probe compiles the actual generated vision source under a CPU test spelling
of CUDA attributes/intrinsics, with ASan/UBSan. It tests raw BF16 patch/gather
indices, learned-position addition, mean residual, head72 crop isolation and
normalized-probability rounding. Host threads model CTA barriers. It does not
use NVCC, emulate a GPU instruction set, execute the full encoder or qualify
CUDA synchronization/performance. The test shim is never a production backend.
