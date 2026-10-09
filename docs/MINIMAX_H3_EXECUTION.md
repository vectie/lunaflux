# MiniMax H3 execution workstream

Current code/module inventory: [2026-09-26 integration update](MINIMAX_H3_CODE_INTEGRATION_2026-09-26.md).
The entries below are chronological; old missing-module lists are superseded
by that inventory. GPU qualification remains a separate, uncompleted boundary.

## Architecture decision — 2026-09-22

One framework, shared compiler/device foundation, separate autoregressive and
diffusion engines. No second repository, universal scheduler, Python runtime,
or request-path JIT. Family builders describe semantics; pure compiler passes
choose layouts and reuse; prepared device backends implement effects.

## Implementation order

1. Bounded joint-denoising driver: precomputed paired timestep/sigma records,
   explicit progress/cancellation/failure, no per-iteration tensor allocation.
   Backend completion covers joint prediction followed by both Euler updates.
2. Bind H3 conditioning and latent initialization to prepared storage and the
   existing denoiser operations. Hoist invariant conditioning and normalized
   VAE weights out of repeated execution.
3. Complete VideoVAE and AudioVAE decoder bodies and test full output against
   an independent reference at one exact supported request geometry.
4. Compose deterministic device ownership, cancellation, bounded output and
   request admission. Keep codec/transport above numerical execution.
5. Measure conditioning, each denoiser block, updates and both VAE decoders;
   report peak memory, launch count, transfer volume, per-iteration and total
   latency. Optimize dominant costs through generic LunaTile passes.

## Initial implementation scope

`engine/joint_diffusion_loop` executes precomputed paired steps through a
synchronous prepared-backend interface. The backend is borrowed, owns its
buffers, and must complete effects before returning. The driver owns progress
and prevents replay after cancellation or failure; it does not load weights,
allocate device memory or claim H3 readiness. Its tests exercise H3's actual
video/audio grids with a deterministic numerical test backend.

Remaining: real denoiser/device binding, conditioning, complete VAE execution,
media output, numerical comparison and GPU performance measurement. Existing
source candidates are not performance results. Spark is currently assigned to
the restored GLM service; do not launch GPU work without renewed coordination.

## Validation — initial driver

Six native tests pass: joint old-state arithmetic, terminal non-replay, partial
failure, cancellation between steps, H3 five-step dual-grid dispatch,
preparation snapshot/invalid schedule checks, and cancellation with recursive
dispatch rejection inside a synchronous backend callback (some share tests).
`moon fmt --check` and scoped `moon info` pass. Test invocation temporarily
disables warnings 79 and 20 for existing test dependencies; no repository
warning policy is changed. Unmodified strict warning-denied checking is blocked
by those dependencies' toolchain migration warnings. No GPU test or speedup is
claimed for this driver.

## Request preparation — 2026-09-22

`PreparedPlan` now derives paired steps directly from the generic model plan
and resolved request. Production imports no reference or H3 family package.
Both H3 variants at 5/10/20/50 evaluations match the independent reference
grids; exact combined latent-element ceilings accept and one-element-short
ceilings reject before tensor allocation. Independent runs share immutable
steps but not progress, and execute through the numerical test backend.

These are element counts, not total memory admission. The boundary layout
plans described below now map video patches and audio rows to latent storage.

## Stereo latent row correction — 2026-09-22

The shape contract now distinguishes `audio_latent_frames` (time per stream)
from `audio_token_rows` (stream count times time). H3 explicitly declares two
latent streams; this is not inferred from output channels. At 124 video frames
the denoiser needs 414 audio rows of width 32, or 13,248 F32 values, not the
previous 207 rows / 6,624 values. SGLang's H3 `latent_preparation.py` constructs
`audio_rows_n = audio_t * 2`; `packed_tokens.py` restores these stream-major
rows to `[2,32,207]` for the AudioVAE.

Request preparation, flow/projection/output-head/final-normalization lowering,
packed modality selection, reference RNG limits and VAE memory assessment use
the full row count. VAE time-axis operations still use 207, not 414. The stream
count participates in plan identity so prior mono-sized artifacts cannot be
reused for the corrected H3 plan. Latent creation and device ownership remain
execution work, not completed functionality.

Stereo correction validation: 107 scoped native tests and 30 adjacent
requirements/materialization/conditioning/transformer/admission tests pass.
The regression covers one/two/three latent streams with output channels held
fixed, checked packed-row overflow, both official H3 variants, and preserved
per-stream VAE time. Existing migration-warning exclusions are unchanged.

## Boundary layout module — 2026-09-22

`kernels/dense_permutation_plan` represents a dense permutation as an immutable
output-index-to-input-index expression. It removes unit axes and merges adjacent
axes that are contiguous in both layouts. Rank-sized metadata replaces a
tensor-sized index table; inversion derives unpacking from the same plan.
`kernels/dense_permutation_cuda_source` lowers this expression to an out-of-place
32-bit-cell kernel, preserving F32 bits without floating-point arithmetic.
The initial lowering uses unrolled constant index expressions and no scratch or
barriers; it is not yet physically profiled or claimed to be an optimal transpose.

Request preparation binds both transformations from generic model geometry:

- Video raw `[1,C,T,H,W]` is split to `[1,C,t,pt,h,ph,w,pw]`, then permuted to
  `[1,t,h,w,C,pt,ph,pw]` and viewed as `[t*h*w,C*pt*ph*pw]`.
- Audio denoiser rows `[S*T,C]` are viewed as `[S,T,C]` and permuted to
  `[S,C,T]` for the AudioVAE, preserving stream order.

The execution policy is pack video once at initialization, retain packed rows
through every joint RF step, then unpack each modality once before VAE decode.
Conditioning rows remain separate from target slices. The arena now retains
three immutable boundary transfers, pairing each permutation with its exact
source/destination buffer. `LatentStorage::boundary_launch` binds those regions
to an admitted AOT function; both transfers use non-overlapping storage and
the same bit-preserving renderer. Video boundary storage is reused at output.
Audio initial noise is already token-major and needs no input transpose.
The full request runner still must schedule input packing before denoising and
both output unpacks after successful completion, before their VAE decoders.

Validation: 65 scoped native tests pass across the new layout packages,
joint loop, H3 AOT lowering and H3 integration. Layout regressions include
independently enumerated patch ordering, temporal patches with multiple batches,
stereo isolation, all rank-three permutations, inverse recovery, extent overflow
and bit-exact RF-update/permutation composition. The two new packages pass
warning-denied checking without warning exclusions; existing loop/AOT test
dependencies still require the previously documented migration exclusions.

## Prepared frame execution — 2026-09-22

FL2VA coordinate completion: `PositionedSequence::keyframes` now places
resolved first/last anchors on the target spatial grid without shifting target
audio/video origins. Duplicate anchors are rejected and caller order is
preserved. Last-anchor FP64 summation uses the distinct eight-lane, 128-element
pairwise reduction policy documented by [NumPy's implementation](https://raw.githubusercontent.com/numpy/numpy/v2.2.6/numpy/_core/src/umath/loops_utils.h.src),
not Ref2VA's sequential span. Eleven sequence tests pass, including reduction
boundaries and mixed-reference regression. These are host coordinate tests,
not an executed NumPy oracle or full H3 numerical comparison. Rotary coefficient
generation and GPU coordinate use remain incomplete.

### Explicit packed hidden assembly

`PositionedSequence::references` now prepares Ref2VA three-axis FP64 position
grids together with their structural sequence. It covers image, audio and
paired video/audio references in request order, advances origins by one for
images, by audio duration for audio, and by the longer span for paired blocks.
Video uses the H3 5/3-scaled `(1,4,4,4,4)` cadence; spatial axes preserve
linspace endpoint arithmetic with sqrt-area normalization and scale 32.
Stereo audio occupies the first/last spatial-width coordinates, and padding
coordinates stay zero. Encoded grid dimensions are explicit and bounded before
allocation. Nine sequence tests pass, including independent small coordinate
fixtures, rectangular grids and mixed-reference origins. This is not physical
RoPE execution. FL2VA first/last anchor reduction, rotary coefficients and GPU
coordinate upload remain incomplete; no approximate last-anchor rule was added.

H3 per-evaluation timestep preparation is now implemented on `Sequence`:
text and padding inherit target video time; visual conditions use
`max(video_time, visual_noise_time)` and audio references use
`max(audio_time, audio_noise_time)`. Candidates are converted to F32 before
ascending deduplication, preserving collisions introduced by F32 rounding.
The resulting immutable unique-value table and row indices are startup
metadata, not work to repeat inside the denoise loop. This follows local
SGLang `_expand_step_timesteps` and `prepare_timestep_plan` semantics.
Five sequence tests pass, including modality mapping, padding, noise-time
clamping, F32 collisions and invalid times.

`Sequence::schedule` now prepares all evaluations and interns equality/order
patterns before expanding row metadata. Evaluations with different time values
share row indices when their slot-to-unique mapping is identical. Each distinct
pattern also stores `3*timestep_index + modality` for AdaLN, including explicit
FL2VA vision-text overrides; padding uses clamped video modality zero. An
explicit budget limits resident Int32 index payloads (not all host overhead).
Seven sequence tests pass, covering repeated patterns, F32 collapse, modality
tags, overrides and storage limits. GPU upload and kernel binding remain
unconnected; no step-path host table generation should be introduced there.

`model/minimax_h3_sequence` now derives the actual H3 structural row order for
one CFG branch. Encoded conditioning blocks retain request order; paired
reference audio precedes its visual rows. Targets follow as audio then video,
with a 64-row padded extent. The resolved request supplies target dimensions
and hidden width; explicit post-encoder counts supply conditioning extents.
Its generic assembly map connects directly to `PackedHidden`, and retained
segments expose ranges for subsequent coordinate/timestep preparation.
This does not yet implement anchors, RoPE coordinates, per-row time values,
attention masks, CFG composition or conditioning encoders. In particular it
does not infer encoded dimensions from raw media or replace request admission.

Seven scoped native tests pass across this module and the packing plan/source
packages. Official FL2VA geometry with three text rows and two 1008-row image
blocks yields 37,296 target video rows, 414 audio rows, 39,729 live rows and
39,744 padded rows. Tests also cover ordered mixed references, padding budgets
and invalid conditioning counts. No GPU or complete model result is claimed.

Source review of SGLang's H3 `packed_sequence.py` shows FL2VA uses
text/conditioning, audio, then video, with 64-row sequence padding. The tiny
reference's conditioning/video/audio concatenation is a fixture convention,
not the production ordering contract. Do not derive H3 indices from it.

`kernels/packed_rows_plan` now snapshots explicit destination positions for
each projected source. It rejects overlapping/out-of-range destinations and
marks unassigned rows as padding. `packed_rows_cuda_source` performs a single
bit-preserving BF16 assembly with explicit zero padding. No FP conversion,
barrier, or per-step host concatenation is needed. `PackedHidden` allocates the
packed output and immutable map once and binds the existing projection outputs
plus preprojected conditioning. Mapping upload is startup-only. A model-specific
processor must still supply actual positions, rotary coordinates, timestep
selection and masks; this module does not claim those are complete.

Three strict warning-denied native tests cover interleaving, padding, BF16 bit
patterns, metadata encoding, snapshot ownership and rejected overlaps/budgets,
plus emitted CUDA source. Device binding passes scoped native checking. No
physical CUDA result or bandwidth claim is made.

### Latent input projections

`LatentStoragePlan::projections` derives both modality dense shapes and a
bounded BF16 output arena from the generic model and this exact request.
Video input remains patch-major F32 with width 96 for the official geometry;
audio input is F32 stream/token-major with 414 rows of width 32. Weights are
F32 row-major `[hidden,input]` with separate F32 bias. Output regions have
256-byte alignment and are reused across denoise steps after consumers finish.
The budget includes both outputs and padding, not weights or later activations.

`LatentProjections` owns this allocation and borrows the latent arena. Its
launch binder connects actual state/weight/bias/output regions to the existing
256-thread F32-to-BF16 dense AOT renderer. No model family or CUDA detail enters
the pure shape planner. Startup function/weight admission remains the caller's
responsibility. Close queues before projection outputs and latent storage.
The source renderer remains a scalar ordered-F32 correctness kernel, not a
qualified high-throughput GEMM. Conditioning projection/refinement, packed
transformer execution and physical model results remain incomplete.

The new projection plan and loop/device scope pass 11 native tests, including
both variants at all four step counts, video/audio extents, weight/bias byte
sizes, aligned non-overlap, exact/short budgets and a foreign-model rejection.
The device module passes native checking with the previously documented
migration-warning exclusions. Host tests do not verify actual CUDA execution.

### Native latent initialization

`LatentStoragePlan::noise` prepares a stateless, plan/seed-bound SplitMix64
and Box-Muller generator for video and audio. `NoisePlan::chunk` emits bounded
little-endian F32 payloads with separate modality domains; counters depend on
the element index, not chunk size, call order or mutable global RNG state.
Production does not import the correctness-reference package. Native
`LatentStorage::initialize_noise` stages chunks into the existing arena:
video raw channel-major noise into `VideoBoundary`, audio stream/token-major
noise directly into `AudioState`. Pack video once before model execution.
No full host latent tensor is allocated and no initialization occurs in the
denoise loop. Chunk size is explicit and bounds host staging memory, not GPU
model residency. A transfer failure leaves the arena partially initialized;
retry the entire initialization before submission or release the arena.

This is the native RNG v1 contract, not PyTorch CPU/CUDA RNG equivalence.
Independent full-model comparisons must supply identical initial latents,
not assume matching seeds imply matching tensors. The host Box-Muller path
is a functional startup implementation, not an optimized GPU initializer or
a measured latency improvement. Conditioning posterior noise is not covered.

The initialization/loop/device/reference scope passes 17 native tests. For
both H3 variants and 5/10/20/50 steps, tests compare F32 bit patterns against
the independent reference at initial, interior and final chunks, check
single-element versus multi-element chunk invariance with positive/negative
seeds, retain both stereo streams, and reject invalid ranges/chunk ceilings.
The device adapter passes scoped native checking; actual device transfers and
full initialized-model execution have not run.

Prepared execution now has an owning frame sequence in
`engine/joint_diffusion_execution`. Each startup-bound frame performs joint
prediction and both updates; the owner commits progress only after completion.
It stops submissions on cancellation/failure and drains/releases frame leases
with a retained retry cursor. `engine/joint_diffusion_device` adapts the existing
ordered CUDA executor: eager enqueue or captured-graph launch, followed by one
frame-level completion wait. It does not add per-kernel barriers or per-step
argument construction. The borrowed stream/function/allocation owners must
outlive those leases and close after the execution owner.

This connects the iteration protocol to the actual device submission API, but
does not yet construct the complete H3 launch list, allocate model/activation
buffers, or validate GPU numerical results. Startup
must pass exactly one exclusively owned, correctly bound frame for each step.

The request plan now caches a little-endian F32 coefficient table with 32 bytes
per step: aligned video and audio records each hold sigma_t, ratio, complement
and normalized timestep. `FlowBindings::upload` writes this table once into a
caller-owned allocation; frame construction binds fixed 12-byte coefficient
regions and optional 4-byte timestep regions. The adapter adds no per-step
uploads or coefficient arithmetic. The complete H3 launch list and hidden-state
activation allocation still remain; packed conditioning timestep tables are distinct.

The latent allocation module now separates a pure bounded `LatentStoragePlan`
from its explicit `LatentStorage` device owner. One 256-byte-aligned arena holds
video/audio states, velocities, reusable boundary permutation buffers and the
coefficient table. Stereo audio extents are retained throughout. Allocation and
upload are separate so upload failure cannot lose the resource owner. Queues
must release their leases before arena closure; cleanup errors remain retryable.
The arena is uninitialized until startup prepares its inputs. It excludes model
weights, conditioning, hidden-state and VAE workspace, so it must not be used
as a total-model memory estimate. Both variants and 5/10/20/50 steps now test
region sizes, nonoverlap, alignment, padding and exact/insufficient byte budgets.

`LatentStorage::flow_launch` now prepares the actual four-pointer RF launch:
state, velocity, this arena's step coefficient record, and exact in-place state
output. Grid geometry derives from the modality extent and admitted block size;
no temporary clean-state allocation or host transfer occurs in the update.
The caller must bind the matching AOT function and finish both predictions
before scheduling the two updates. This is a launch-binding implementation,
not a completed denoiser graph or a physical CUDA test.

`KernelFrame::denoise` now composes the supplied joint prediction list followed
by video and audio RF launches, preserving old-state prediction semantics.
Launch-list copying and argument/geometry preparation happen before native
queue creation; execution adds no per-step binding. Model-specific prediction
binding and real-model correctness remain incomplete. Boundary regressions
cover disjoint arena regions, modality extents, inverse mapping and stereo
order for both variants and all four supported step counts.

The boundary/frame binding module and its loop, execution, permutation and RF
renderer dependencies pass 27 scoped native tests. Linear launch geometry tests
cover partial blocks, official latent extents and the Int32 element ceiling.
`moon info` succeeds with existing dependency migration warnings; the scoped
warning-denied tests retain the documented exclusions. No CUDA execution,
sanitizer result or performance improvement is inferred from these host tests.

Coefficient-table regression checks both H3 variants and 5/10/20/50 steps,
including every scalar's F32 bit pattern, little-endian encoding, aligned
regions, terminal coefficients and index bounds. The loop/execution regression
set passes 15 native tests; the upload/binding adapter passes native checking.
No hardware upload or physical kernel execution is claimed by these host tests.

The owning execution module and adjacent loop/layout packages pass 24 native
tests, including retry after drain/release failures and rejection of close
during active execution. The device adapter passes scoped native checking with
the existing dependency migration-warning exclusions. No physical GPU test is
claimed; the tests use a synchronous numerical frame implementation.

## Corrected RF numerical contract — 2026-09-22

Inspection of both local supported framework implementations exposed a deeper
error than storage width. H3 predicts velocity with clean-state convention
`x0 = x + (1-t)*v`, using normalized increasing `t = 1-sigma`. Its eta-zero
update is `ratio*x + (1-ratio)*x0`, with ratio = next_sigma/sigma. State and
velocity remain F32. The older BF16 negative-delta candidate and the earlier
driver's training-scaled timestep assumption are not compatible with this
pipeline. Matching the existing reference sigma grid alone did not prove model
timestep or update correctness.

The driver now retains explicit current/next sigmas and normalized timesteps,
checks adjacent-step continuity and terminal zero, and prepares ordered F32
coefficients. Numerical tests cover velocity sign, sub-BF16 changes, and an
adversarial multiply/add where FMA would produce a different result.

A subsequent model-chain review found the old negative Euler delta still in
`joint_diffusion_transformer_reference`, including its expected-value fixture.
That independent Double oracle now reconstructs clean state and blends by the
sigma ratio. Both FL2VA-like and Ref2VA-like tiny fixtures check both modalities
at every evaluation using the separately simplified positive Euler delta,
including terminal zero sigma. The reference plus production loop pass 15
scoped native tests. This corrects the reference's velocity convention; it does
not make its simplified transformer a complete H3 numerical oracle.
`kernels/rectified_flow_update_cuda_source` lowers the matching F32 update to a
single elementwise kernel: intermediate values remain in registers, avoiding
the tensor-sized clean-state scratch and separate blend launches. Ratios are
prepared before dispatch, not divided per element. The H3 shifted-flow AOT
lowering now uses this renderer, with F32 state/velocity/output and a distinct
three-F32 coefficient operand. The v2 recipe prevents reuse of the obsolete
BF16 update identity. This remains an offline candidate: physical compilation,
device execution, and full model numerical qualification are not complete.

The scoped native regression set passes 65 tests across the joint loop,
generic update renderer, flow AOT package, artifact admission, and H3 AOT
integration. Integration checks include official-profile F32 buffer extents
and the changed candidate identities. These are host/source-contract tests,
not GPU numerical or performance results. The current toolchain requires
excluding existing migration warnings 20/25/29/79/92 for this scoped run;
repository-wide warning cleanup and full-suite validation are not claimed.

Source observations: local vLLM-Omni checkout HEAD
`0d339e25755f067d2f6981ffa7ac998127612163`,
`scheduling_minimax_h3_euler_ancestral.py` observed SHA-256
`e075d34a5415b0e72e91cbf25cf90cddeb037ec9397a2f560387661f2f3b192e`;
SGLang checkout HEAD `a2e88279c28c16945c7c7eacb27f1e066b670a41`,
`model_specific_stages/minimax_h3/denoise_loop.py`.
The [official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
lists both frameworks as supported local inference implementations. These
source observations are not independent real-model numerical qualification.

## H3 rotary coefficient preparation — 2026-09-22

`model/minimax_h3_sequence` now prepares immutable rotary cosine/sine tables
from checkpoint inverse frequencies and the previously prepared 3D positions.
Coordinates convert to F32 before F32 multiplication, axis frequency blocks
are concatenated and duplicated for the full-width attention operand ABI.
The two tables share an explicit element budget; padding is identity rotation.
Preparation happens outside the denoising step, without repeated trigonometric
work in dispatch. The sequence package passes 12 native tests covering layout,
timesteps, reference/keyframe positions, rotary layout and capacity bounds.

This is host preparation, not a complete H3 inference deployment. Conditioning
encoders, complete transformer weight/activation binding, VAE execution and
media output remain to be connected and validated. In particular, host
trigonometric rounding has not been compared against actual GPU RoPE output;
these tests establish neither physical numerical parity nor inference speed.

## Packed attention visibility — 2026-09-22

H3 sequence planning now exposes compact noncausal attention documents. The
live prefix (text, conditions, target audio and target video) is one document;
padding is a separate document, omitted when empty. The text refiner exposes
only its live text range. This follows the reference packed cumulative lengths
`[0, used, total]` and avoids allocating a quadratic attention mask.

The sequence regression set now passes 13 native tests, including padded and
exactly aligned layouts and all live modality segments. This also identifies
a remaining binding requirement: the existing unmasked attention renderer
cannot be launched over all padded rows as one document. Document-aware device
binding and GPU numerical tests remain outstanding; pure range tests are not
proof that the complete transformer already honors padding isolation.

The generic staged attention renderer now accepts cumulative document
boundaries and bounds both softmax passes to the query's own document.
`Sequence::attention_boundaries` supplies this representation without importing
CUDA into model planning. Empty/overlapping/out-of-range boundaries are rejected
before source generation. Four source-package tests pass, including both
reduction bounds; physical kernel execution and full transformer binding remain
outstanding. Existing callers which omit boundaries retain unmasked semantics.

The joint-transformer AOT entry now carries explicit cumulative boundaries
through to packed attention source generation and records segmented visibility
in its recipe. Single-document lowering retains its previous source and recipe;
segmented layouts receive distinct source/recipe identities. The H3 integration
test verifies propagation and identity separation, without claiming that the
fixture split is an official media layout or that device binding is complete.

## Shared rotary device operands — 2026-09-22

`engine/joint_diffusion_device.RotaryBindings` now accepts the prepared F32
cosine/sine tables and uploads them into a caller-owned contiguous allocation.
It exposes fixed device-region operands reusable across attention layers and
denoising frames. Geometry, finite values and allocation bounds are checked at
startup; upload uses caller-bounded chunks rather than a second full-size host
byte copy. Neither validation nor upload belongs in step dispatch.

Ownership stays with the caller on upload failure; retry the complete upload
before queuing work. Frame queues must release their allocation leases before
the caller closes storage. This generic adapter imports no H3 model package.
The device package passes two native host tests; the new regression checks F32
little-endian serialization, signed zero and chunk concatenation. These tests
do not exercise a physical allocation/upload or establish complete H3 binding.

The rotary binding now also prepares the staged QK normalization/RoPE launch,
with explicit borrowed Q/K input tensors, per-head weights, shared cosine/sine
tables and separate normalized outputs in the renderer's eight-operand order.
Geometry derives from the uploaded table rows/width and validates head shape
and element bounds before frame preparation. The adapter follows the current
single-thread-per-head correctness renderer; it is not a claim of efficient
GPU normalization. The device package passes three host tests, including
official-width extent arithmetic and invalid/overflow geometry. Full graph
assembly, physical numerical comparison and optimized lowering remain open.

The device adapter now prepares all four staged attention launches in ABI
order: QKV, QK normalization/RoPE, document-aware attention, output projection.
Six BF16 scratch tensors are reusable between serialized blocks. Packed
per-layer weights and scratch remain caller-owned; whole allocation spans are
checked before deriving offsets. The caller must provide nonoverlapping weight,
input/output and scratch regions and functions compiled for the prepared shape
and document boundaries. No work is submitted while this list is constructed.

The device package passes four host tests, including official unequal
hidden/inner-width byte extents and overflow rejection. This is a complete
attention launch-list adapter, not the complete transformer: packed checkpoint
weight materialization, AdaLN/residual/MLP composition, physical execution and
performance validation remain outstanding. The six-tensor correctness layout
also needs lifetime-based memory reuse before claiming efficient H3 execution.

The subsequent lifetime review reduces that adapter's scratch from six tensors
to four: QK normalization reads each whole head into private arrays before any
output write, so normalized Q/K replace their own inputs. Attention output stays
separate because attention still reads Q/K/V while producing it. This reuse is
valid for the current staged renderer, not an unrestricted aliasing contract
for future kernels. `staged_attention_scratch_bytes` exposes the actual extent
used by binding. For 39,744 rows and 56 heads of width 128, planned scratch drops
from 3,418,619,904 to 2,279,079,936 bytes (one third less). Four host tests pass;
this is verified allocation arithmetic and source lifetime analysis, not a
measured GPU peak-memory reduction. Physical race/numerical tests remain due.

## Staged MLP recomputation removal — 2026-09-22

The old scalar correctness renderer recomputes the full up/gate projection
inside each down-output channel. A new family-neutral two-stage packed BF16
SwiGLU renderer materializes one BF16 product per row/intermediate channel,
then performs the down projection. Ordered F32 reductions and BF16 rounding
after up, gate, activation and product are preserved in the generated source.
Projection work changes from approximately `2*rows*hidden^2*intermediate`
multiply-adds plus down to `3*rows*hidden*intermediate` total, at the cost of a
`2*rows*intermediate`-byte product buffer and a second launch. This arithmetic
comparison is not a measured speedup.

Six renderer host tests pass. This staged path still needs AOT launch/weight
binding and device numerical validation; the existing single-launch candidate
has not silently changed its ABI. Scalar source is an intermediate correctness
implementation, not a substitute for eventual tiled GEMM lowering.

`StagedMlpLayout` now computes input, intermediate-product and packed weight
extents and binds the two admitted functions in producer/consumer order.
The product region is shared between launches, with all storage caller-owned
and reusable across serialized layers. Whole weight spans are checked before
offset arithmetic; the total six-byte-per-matrix-element packed extent also
has an explicit Int64 overflow guard. Five device-package host tests pass.
This completes the staged device adapter but not checkpoint packing, AOT
artifact qualification, GPU execution or the full AdaLN/residual block chain.

## Indexed block modulation — 2026-09-22

The affine-modulation source package now emits indexed BF16 scale/shift and
gated-residual kernels. They read compact parameter rows using the prepared
combined timestep/modality indices, avoiding full per-token parameter expansion.
The affine sequence rounds `1+scale`, multiplication and addition separately
through BF16; gated residual rounds the product before adding the residual.
Indices must be validated before upload and remain immutable during execution.
The package passes 11 host/source tests. RMSNorm composition, index upload,
launch binding and physical numerical comparison remain outstanding; this
source module alone is not a completed transformer block.

The generic grouped-parameter projection renderer now accepts pre-activated
BF16 conditioning and BF16 weights/bias, producing field-major compact tables
directly: [fields, timestep_rows*modalities, hidden]. H3 block configuration is
three modalities and six fields. This matches the reference block's already
SiLU-activated input rather than incorrectly applying SiLU again (the final
AdaLN renderer has a different input contract). Direct output indexing avoids
a separate transpose and per-token expansion. Twelve affine-source host tests
pass; physical GEMM parity, parameter-weight upload and device launch binding
still remain to be completed. Ordered scalar dot products are not yet a
performance-qualified tiled lowering.

The block packing plan now additionally includes `adaln_proj.linear.weight`
and bias, with separate offsets and bounded byte extents. The grouped projection
device adapter binds pre-activated conditioning, these two weight regions and
the compact output. Its launch must precede the twelve block consumers in the
same ordered frame; parameter storage can then be reused for the next layer.
The device/weight host regression set passes 26 tests, including agreement on
the official 96,768-by-2,688 BF16 weight and six compact output table sizes.
This is host construction and upload-plan coverage, not executed GPU AdaLN.

`bind_conditioned_block` now returns parameter projection followed by all
twelve block consumers in one ordered list. Distinct timestep row count is
derived from compact parameter rows and the explicit modality count; incomplete
groups are rejected rather than truncated. Parameter production cannot be
accidentally omitted by callers using this entry point. Ten device host tests
pass, but real pre-activated conditioning generation, compiled artifact loading
and GPU execution still need integration/validation.

The missing F32-SiLU-to-BF16 conditioning renderer and device launch adapter are
now implemented. They consume timestep-MLP F32 output and cast only after SiLU,
matching the inspected H3 preactivation order. Prepare this shared table once
per evaluation before all blocks, not once per block. The combined source and
device host regression set passes 23 tests. Full evaluation assembly, actual
CUDA execution and comparison against the reference transcendental/numerical
behavior remain open; source inspection is not device parity proof.

`ModulationBindings` now validates every Int32 row index against the compact
parameter count, uploads the immutable map once into borrowed storage, and
binds either affine or gated-residual operands. It distinguishes the compact
scale table from the full residual-branch tensor when checking extents. The
map is reusable across serialized layers for the same prepared timestep
pattern, with no upload in step dispatch. Six device-package host tests pass;
new tests cover byte encoding and invalid index rejection. Physical execution,
complete block assembly and RMSNorm binding still remain to be verified.

The fixed-row RMSNorm device adapter and twelve-stage block composition are
now present: norm/affine, four attention launches, gated residual, norm/affine,
two MLP launches, gated residual. Composition produces one ordered launch list
to append to the existing denoising frame, rather than adding per-layer host
waits. Seven device-package host tests pass, including exact stage ordering and
missing-stage rejection. The composition helper does not establish matching
weights, tensor regions or numerical parity: complete checkpoint-backed block
construction and physical validation remain required. RMSNorm uses the
existing 256-thread fixed-row reduction contract, whose match to the actual
H3 device reference still needs numerical testing.

`BlockWorkspaceLayout` now plans two disjoint aligned hidden temporaries
(normalized input and branch output), followed by one shared scratch interval
sized to max(attention scratch, MLP product). Residual state stays external.
The layout checks the complete request-local byte budget before allocation;
`BlockWorkspace` owns exactly one allocation with explicit close after frame
leases. Reuse is across serialized stages/layers, never concurrent requests.
Eight device-package host tests pass, including both scratch-dominance cases,
alignment and exact/one-byte-short budgets. End-to-end block binding and GPU
lifetime/race tests are still required before this becomes a validated runtime.

`BlockWorkspace::bind_block` now performs the actual twelve-launch assembly,
not just list concatenation: it wires norm/affine into attention and MLP,
their outputs into indexed gated residual, and both residual updates back to
the external state. The six compact modulation tables have explicit
attention-shift/scale/gate then MLP-shift/scale/gate order. Uploaded rotary and
modulation geometry must match the workspace; complete external parameter/norm
spans are checked before offsets are formed. Functions and already-uploaded
weights/parameters remain caller-supplied and exact-shape-specific.

The adapter compiles and the eight existing device host regressions pass.
These do not instantiate a GPU-backed block: physical argument/alias checks,
numerical block comparison, checkpoint packing and actual parameter production
are still required. No full-model execution or speed claim follows from the
successful host compilation.

## Checkpoint block packing — 2026-09-22

The H3 weight package now emits bounded per-block copy plans for Q/K/V,
Q/K normalization, output projection, MLP and both block RMSNorm vectors.
Crucially, checkpoint `ff.net.0.proj.weight` is [gate,up], whereas the staged
runtime ABI is [up,gate]: two source slices explicitly swap the halves without
numeric conversion. Destination offsets match the device block binder's
attention/MLP/norm spans. AdaLN projection remains a separate weight component.
This pure plan is not itself a checkpoint upload; actual materialized source
resolution and chunked copying still need integration.

The materialized host component can now resolve each block copy to its actual
arena index and byte offset, including the two swapped MLP slices. Resolution
checks arena availability, expected component identity, denoiser component kind,
tensor presence and overflow-safe slice bounds without allocating payloads.
Returned offsets are borrowed and expire when the host arena is released.
This closes metadata-to-materialized-storage resolution; it does not yet copy
those ranges to the packed device arena or verify a complete model execution.

`integration/minimax_h3_block_upload` now connects those resolved source ranges
to actual device-copy calls through the existing synchronous host-arena borrow.
Each transfer creates only a caller-bounded chunk, applies the planned
destination offset and revokes the borrow on success or failure. The device
allocation remains caller-owned; any failed upload requires complete retry
before execution. A host regression checks reordered slices, final partial
chunks, invalid source ranges and immediate failure stop. No physical GPU
upload was executed by this regression; complete model/device validation remains
open. The family-specific bridge lives in integration, not the generic engine.

## Complete block source assembly — 2026-09-22

`integration/minimax_h3_block_source` now assembles eleven entry points from
one model/sequence geometry: conditioning activation, AdaLN parameter projection,
RMSNorm, indexed affine/residual, staged QKV/QK normalization/attention/output,
and staged gate/up/down. Padding is a separate attention document. Block norm
epsilon is explicitly supplied from the transformer specification, rather than
implicitly borrowing the final-output norm epsilon. This is offline source
assembly, not request-path JIT.

The native source-bundle regression passes, checking all eleven emitted symbols,
padding isolation and rejection of an empty timestep table. CUDA compilation,
physical numerical validation and full-model execution remain outstanding.
These scalar correctness kernels are not yet efficient tiled H3 inference;
conditioning encoders, complete denoising graph and VAE execution still require
integration before an end-to-end benchmark is meaningful.

## Prepared block program — 2026-09-22

`integration/minimax_h3_block_program.Program` connects the assembled source's
entry list to a caller-admitted AOT module, allocates the shared block workspace
from that same source geometry, and binds the thirteen-launch conditioned block
using the checkpoint packing offsets. Conditioning activation has a separate
launch so it can be shared across blocks. Compact modulation parameter-row count
must match the source's distinct-timestep count at binding, not at every step.

The program owns function handles and workspace, but not the module, model
weights or external tensors. The caller retains the program after a failed
prepare and calls close; partial acquisitions stay tracked. Close removes a
handle only after successful release, supports retry, and must follow frame
lease release. Caller admission must still establish that the loaded module
is the AOT result of the exact source bundle; symbol lookup alone does not
establish that relationship.

Two native lifecycle regressions verify partial acquisition, reverse release,
idempotent empty cleanup and retaining a failed-release handle for retry. The
source-bundle regression and ten generic device host regressions also pass.
These host tests do not load CUDA modules or execute a block. Physical ABI,
numerical and leak validation remain required before device execution is claimed.

## Final media heads — 2026-09-22

The source assembler now also emits the six final-head entry points: compact
two-field AdaLN parameter projection, final RMSNorm, selected audio/video
modulation and separate F32 audio/video output projections. Media row offsets
come from the sequence segments; text, reference and padding rows do not pass
through the output projections. BF16 rounding of `1+scale`, multiply and add is
retained before widening to F32. Final parameter indices refer to distinct
timesteps, not the block's three-modality indices.

Generic device bindings select the corresponding normalized and index slices,
consume compact shift/scale tables, and bind the existing F32 output projection
directly into the latent velocity region. The request/model join is checked
at binding. The supported H3 checkpoint is CFG-distilled and uses one positive
branch, as already represented by `plan.guidance_distilled()`. No second
prediction or runtime CFG combination belongs in this checkpoint's flow path.

The affected native host suites pass (12 device, 14 modulation source, one
combined block/output source regression). They cover selected-range bounds,
official video/audio byte extents, emitted entry points and source rounding
structure. They do not prove CUDA numerical equivalence. Full final-head
resource ownership, admitted AOT loading, end-to-end conditioning/VAE
integration and physical tests remain open; these scalar kernels are not a
claim of efficient end-to-end H3 inference.

## Final-head checkpoint upload — 2026-09-22

Final-head packing now covers all seven checkpoint tensors, preserving BF16
normalization/AdaLN and F32 video/audio heads without conversion. It uses the
checkpoint's `norm_out`, `proj_out` and `audio_proj_out` names, not the reference
backend's `final_layer` module aliases. Bounded host resolution and chunked
device copying share the existing block upload implementation, including borrow
revocation and caller ownership on transfer failure.

The native packing regression verifies official byte extents, contiguous spans,
F32 alignment and exact/short budgets. The weights suite passes 18 tests and
the chunk-copy test passes; these do not execute a physical output-head upload.
Earlier references to missing CFG composition were incorrect for this supported
distilled checkpoint. The local SGLang H3 adapter also explicitly rejects CFG
scales and negative prompts for its single-positive-branch checkpoint.

## Prepared final-head execution — 2026-09-22

`OutputProgram` now owns final-head function handles and one request-local
workspace. It binds parameter production, final normalization, video modulation
and projection, then audio modulation and projection as six ordered launches.
The F32 selected-hidden region is reused only after the video projection has
consumed it; neither latent state is updated by this list. Both resulting
velocity buffers are then available to the existing joint flow-update frame.

Preparation derives aligned normalized/parameter/selected spans from the same
source geometry and checks the workspace budget before loading functions.
Binding checks the modulation geometry and both prediction row counts against
the actual latent request. Partial load/allocation failures retain acquired
handles for explicit close, using the same retryable cleanup as block programs.

The native suites pass (three program/lifecycle/layout tests, one source-bundle
test and twelve device host tests). The layout regression covers official media
extents, selected-scratch reuse and exact/short budgets. No CUDA module was
loaded; physical final-head numerical/ABI/leak tests and complete request
orchestration remain outstanding. This closes the previously missing final-head
owner and launch assembly, not end-to-end H3 inference.

## Staged timestep MLP — 2026-09-22

The timestep source package now offers a staged dense+SiLU / dense F32 path.
Unlike the earlier workspace-free correctness kernel, it computes each first
projection element once and reuses the materialized hidden row across all output
columns. It preserves that renderer's bias-first, ordered F32 contract; matching
that contract is not yet proof of equivalence to a reference backend GEMM.

Generic device bindings cover sinusoidal frequency generation and both MLP
stages. The existing F32-SiLU-to-BF16 activation can consume the resulting F32
embedding for block/final AdaLN. At three distinct timesteps the MLP needs
64,512 bytes of hidden scratch and 32,256 bytes of output, excluding frequency
rows and the BF16 activation. These buffers are prepared and reused, not
allocated inside the denoising loop.

Five timestep-source tests and thirteen device host tests pass, including the
staged entry layout and bounded official workspace geometry. Startup ownership,
timestep-weight upload and full request orchestration still need integration;
no GPU timing or numerical success is claimed for these source/binding tests.

## Prepared timestep conditioning — 2026-09-22

The offline source assembly now exports frequency, staged MLP hidden/output and
SiLU/BF16 activation entries together. `TimestepProgram` loads those four entries
from the caller-admitted module, owns reusable F32 scratch, and prepares the
four-launch chain into caller-owned BF16 conditioning storage. That output is
shared by the block and final-head programs; timestep embedding is not repeated
per transformer block. Three official-width timestep rows require 99,840 bytes
of internal scratch, excluding the external input times and BF16 conditioning.

Checkpoint packing, host resolution and bounded upload now cover the four F32
`time_embedder.linear_1/linear_2` tensors through the shared copy path. Program
preparation failures retain acquired resources for explicit retryable close.
Native source/program/weights/host/upload suites pass 37 tests total. These
verify host composition, byte extents and lifecycle helpers, not device results.
This closes timestep module ownership and upload wiring; full request setup,
conditioning encoders, VAE integration and GPU validation remain unfinished.

## Whole transformer stack binding — 2026-09-22

The block program now binds every model layer in order using one shared block
workspace and compact parameter arena, then appends both final output heads.
For the official fifty-layer shape this produces 650 block launches plus six
final-head launches, without fifty separate block workspaces. Weight arenas
remain resident and caller-owned; this is not a weight-offloading implementation.
Packing carries its layer ordinal so missing, repeated and reordered layers are
rejected before binding. Block/output sources must agree on model plan, packed
row mapping, row count and distinct timestep count.

The stack is a prepared launch list, not an independently complete inference
request. Timestep production and latent projection/packing must precede it;
both flow updates follow it. Conditioning encoders and VAE stages are still not
joined into a full request. Native tests pass (five program tests, nineteen
weight tests and one source test), including exact layer-order rejection. These
tests do not execute the fifty-layer GPU stack or establish performance.

## Packed-state and iteration join — 2026-09-22

Stack binding now consumes the actual `PackedHidden` owner rather than requiring
an unrelated raw allocation. Its startup callback verifies width and the exact
immutable row mapping before exposing the state region to prepared consumers;
the trailing mapping remains read-only and the owner outlives queue leases.

`KernelFrame::denoise_staged` orders timestep production, input projection/packing,
the full transformer/output list, then both RF updates under the existing single
completion boundary. For the current fifty-layer composition that is 663
prediction launches and two updates. The native ordered executor's existing
65,536-launch ceiling accommodates it; no capacity change was needed.

Fourteen device host tests and five program tests pass. A stage-order regression
covers the full 663-entry prediction sequence and rejects omitted stages. The
frame still requires correctly prepared input/conditioning functions and AOT
artifacts; this API does not manufacture them or prove an executed H3 request.
No GPU execution was performed during these host checks.

## Prepared latent input module — 2026-09-22

`InputProgram` now joins the two F32-to-BF16 latent projections and the packed
row gather. It owns projected buffers and packed state, uploads the immutable
row map once, and exposes the packed state for transformer-stack binding.
Its three-launch list reads current audio/video latents each iteration; encoded
text/reference conditioning is a distinct input from timestep conditioning.
Preparation checks the actual request row counts against its compiled source.
Partial failures retain resources for explicit close after queue lease release.

The source bundle contains all three entries. Input-weight packing/resolution/
upload covers four F32 checkpoint tensors, with hidden-width biases (not the
latent-width biases of output heads). The shared upload implementation is reused.
The affected native host suites pass 39 tests total. These establish source and
packing composition, not a CUDA run; real condition encoding, VAE execution,
artifact compilation/admission and complete physical request validation remain.

## Token-refiner block primitives — 2026-09-22

The generic block workspace can now bind a standard ten-launch pre-norm
attention/MLP refiner block without AdaLN or rotary. An ungated BF16 residual
addition renderer replaces the main denoiser's modulation/gating stages.
Attention binding shares the existing QKV/attention/output path while selecting
the correct six-operand unrotated QK-normalization ABI; the rotated path retains
eight operands. It does not allocate dummy rotary tables.

Sixteen device host tests and seven foundation-source tests pass. New tests
cover refiner order, six/eight-operand normalization layouts and ungated residual
source semantics. H3's two refiner layers still need checkpoint/source-program
assembly and the final refiner norm before integration with the real encoder
output. No physical refiner execution or encoder completion is claimed.

## Prepared two-layer token refiner — 2026-09-22

The H3 refiner now has a nine-entry offline source bundle and `RefinerProgram`
owner. Its two ordered ten-launch blocks share one workspace and are followed
by the final RMSNorm, operating only on live projected text rows. The input text
slice is updated in place before it enters multimodal packed assembly.

Refiner weight packing shares the main block's attention/MLP/norm copy planner
without AdaLN, including the explicit checkpoint gate/up half swap. The final
normalization vector is stored once, after the last refiner layer. Host resolution
and chunked upload use the existing shared implementation. Function/workspace
preparation retains partial resources on failure for explicit close.

The affected native suites pass forty tests (21 weights, five program, twelve
host, one upload and one source). Source entry and packing regressions are not
GPU numerical tests. The actual upstream encoder/context projection, full request
setup, VAE path and physical execution remain incomplete; the refiner program
does not synthesize missing text conditioning.

## Context projection connection — 2026-09-22

The refiner AOT source bundle now includes the biased BF16 context projection
from the model's text-conditioning width to its hidden width. It reuses the
generic grouped projection with one group and one field, without SiLU or AdaLN.
`RefinerProgram.bind_context` prepends this launch to the existing 21-launch
refiner. This 22-launch preparation is request-static, not part of each denoising
iteration. Encoder input and projected state must use separate allocations.

The API borrows admitted context weights. Source-entry and byte-extent
tests do not establish GPU numerical correctness. The actual encoder, VAE,
complete request orchestration and physical performance validation remain open.

## Context checkpoint materialization — 2026-09-22

Dedicated context packing now maps `context_embedder.weight` and
`context_embedder.bias` into a contiguous BF16 allocation, without host numeric
conversion. Official geometry uses 55,060,992 bytes. Exact-budget, short-budget,
zero and negative budget regressions accompany the copy layout.

Host identity/tensor resolution and bounded chunk upload reuse the common
materialization implementation. `bind_context` derives the offsets from its own
model instead of accepting arbitrary offsets; upload this dedicated allocation
at destination offset zero. The encoder input and projected-state buffers remain
caller-owned and must not alias. Forty affected host tests pass; these are not
GPU execution or numerical comparison results. This completes context weight
plumbing, not text-encoder execution or end-to-end H3 inference.

## Executable text-conditioning owner — 2026-09-22

`TextConditioningProgram` now owns the projected text allocation, refiner and
request-static eager executor. Preparation accepts actual BF16 encoder rows and
admitted context/refiner weights; execution submits all 22 launches and waits
for completion before exposing them to `InputProgram` latent assembly. The fresh
output allocation eliminates input/output aliasing for this composed path.

The state machine rejects repeated execution and use before completion. A
partial preparation or execution remains failed until explicit close; close
drains and releases the queue before functions and storage, retaining owners on
release failure for retry. Downstream denoising queues must close first.

This connection currently handles text-only conditioning. It checks model,
live text count and total conditioning count, so reference-media requests cannot
silently consume the text allocation as if it contained reference rows. Separate
reference assembly remains required. Seven program tests cover host lifecycle
composition and checked output sizing; this is not a physical CUDA validation.
The upstream encoder and VAE are still required for a complete H3 request.

## Reference-conditioning assembly — 2026-09-22

`ConditioningProgram` now composes completed text refinement with already
projected reference-media BF16 rows into one invariant device allocation. The
AOT source bundle derives copy extents and destinations from the same `Sequence`
used by latent assembly. Each video reference contributes its audio segment
before its video segment; empty audio contributes no input. Target audio/video
and padding are excluded. Generic two-byte copies preserve all BF16 bit patterns
without numeric conversion or host round trips.

Preparation binds caller-owned reference allocation regions in sequence order.
The fresh destination avoids source/destination overlap. An eager queue executes
once and must complete before binding to `InputProgram`; binding checks model,
segment semantics and packed mapping. Partial failures require close, which
drains/releases queue leases before functions and destination storage.

Twenty-eight affected host tests pass, including mixed image/video reference
ordering, zero-audio omission and exact packed byte extents. This provides the
assembly stage, not reference encoders/projection execution. Actual encoder and
VAE integration, complete physical request correctness and performance remain
unfinished. No GPU execution result is claimed by these host tests.

## Reference latent projection — 2026-09-22

Conditioning preparation now additionally accepts packed F32 reference latents
when `latent_weights` is supplied. Visual references use the same patch-width
and F32 input-projection weight/bias as target video; audio uses the same audio
projection. The checkpoint-backed input packing determines offsets. Text remains
the completed BF16 refiner result. This follows the reference transformer's
F32 latent embedders followed by BF16 conversion, without claiming bitwise GPU
equivalence to its library GEMM.

The AOT conditioning bundle includes reference projection entries alongside its
copy entries. In latent mode each projection writes directly into the final
conditioning range: no intermediate hidden allocation or subsequent hidden-row
copy is needed. Like other invariant conditioning preparation, it executes once
per request. Without latent weights the existing projected-BF16 input contract
remains available. Input reference ordering is unchanged in both modes.

Eight affected source/program tests pass, checking visual width 96, audio width
32, entry coverage, ordering and existing lifecycle rules. These are host tests.
Raw media encoding, visual latent patching, actual encoder/VAE orchestration and
physical numerical/performance verification remain outstanding.

## Reference VAE layout conversion — 2026-09-22

Pure reference-layout planners now describe raw video `[C,T,H,W]` to patch rows
and stereo audio `[streams,channels,frames]` to stream-major latent rows. Model
patch factors/channel counts determine the mapping; caller-supplied expected
reference rows must agree. Misaligned visual dimensions and byte/element budget
violations are rejected before allocation.

The shared device `PermutationBuffer` owns a fresh F32 output and an eager
one-shot queue for an admitted `dense_permutation_cuda_source` kernel. No numeric
conversion, index upload or host data round trip occurs. It exposes output only
after completion; failed execution cannot replay, and close drains/releases the
queue before output storage. Consumers borrow the output for reference projection
and must release those leases before closing the permutation buffer.

Thirty-three affected host tests pass. Reference-layout regressions check exact
2x2 patch ordering, exhaustive small inverse composition, stereo channel order,
row-count mismatch and insufficient budget. These establish host mapping/source
contracts, not CUDA execution. Raw media VAE encoding, full encoder/VAE program
assembly and end-to-end physical validation remain unfinished.

## Complete AudioVAE stage indexing — 2026-09-22

The seven AudioVAE upsampling transitions now have one config-driven binder,
covering rates `[5,5,2,2,2,2,2]`, halving channels from 1024 to 8, and validating
each checkpoint transposed-convolution shape. Output time uses checked cumulative
growth and padding `(kernel-stride)/2`, matching the reference decoder.
This describes the upsampling spine; AMP bodies must still execute between its
transitions, followed by the final activation/convolution/tanh.

AMP parameter packing now supports all seven stages through one public indexed
entry point. Existing first/second-stage helpers delegate to it. The three
parameter-layout tests pass, including all 126 regions at every stage, tensor
index ranges, nonoverlap and invalid-stage rejection. The new complete transition
binder compiles but has not yet received a config-backed execution test. No
complete AudioVAE or GPU result is claimed.

## Audio AMP activation device binding — 2026-09-22

`AliasFreeActivationLayout` binds the existing F32 ratio-two/12-tap source into
three actual ordered device launches: upsample filter, log-parameter SnakeBeta,
then downsample filter. It validates all device regions and allocates no storage
itself; callers provide reusable scratch and checkpoint parameter regions.
Both doubled-time intermediate arrays coexist, with an aligned second offset.
Input/output may alias because the first launch consumes input before final
output writes; scratch must be distinct from live inputs/outputs/weights.

Twenty related device/source host tests pass, including official first-stage
extents, exact scratch budget, one-byte-short rejection and extreme dimensions.
This is the device-binding constituent needed by every AMP activation, not yet
a complete AMP residual body, complete AudioVAE or physical execution test.

## Audio AMP convolution device binding — 2026-09-22

`AudioConvolutionLayout` now binds the existing equal-channel time-preserving
F32 dilated-convolution source ABI. Weight normalization and convolution are
separate ordered-launch builders, allowing immutable normalized weights to be
prepared once and retained rather than recomputed for every request. An upper
layer still needs to choose that cache policy and own the allocation.

The normalization launch uses one thread per output-channel CTA as required by
the current scalar source; convolution uses the source's 256-thread linear
launch. This is functional device integration, not an optimized convolution
claim. Input/output convolution regions must not alias. Twenty-eight affected
host tests pass, including all seven AMP channel widths, all three kernel widths,
exact weight-budget limits and extreme geometry rejection. Complete residual
body composition, VAE output execution and physical comparison remain open.

## Complete AMP residual-body composition — 2026-09-22

`AudioAmpLayout` now binds all three branches, each containing three residual
pairs, into 100 ordered launches. Each pair performs alias-free activation,
normalized dilated convolution, another activation/convolution, and residual
addition. The first convolution uses dilation 1/3/5 by pair; the second uses 1.
Branches use kernels 3/7/11 and finish with ordered F32 three-way averaging.

The matching AOT source bundle exposes 41 functions, sharing activation kernels
across all 18 activation sites. `AudioAmpProgram` owns those functions and one
reusable scratch allocation. Branch intermediates reuse scratch sequentially;
all three branch results coexist until averaging. The bounded-memory path
normalizes each convolution into shared scratch, so it does not yet exploit a
persistent normalized-weight cache. It adds no host data round trips.

Twenty-two device/source tests and seven program tests pass. New tests verify
scratch nonoverlap, exact budgets, all source entries and dilation/average source
semantics. No GPU execution or full reference numerical match has been run.
Seven-stage VAE orchestration, weight upload, final waveform output and physical
validation remain required before claiming a complete AudioVAE.

## AMP checkpoint upload connection — 2026-09-22

The admitted AudioVAE host arena can now resolve all AMP parameter regions into
whole-tensor device copies, using the same layout consumed by `AudioAmpProgram`.
Identity/component validation occurs once during preparation. Unlike denoiser
gate/up slicing, AMP requires exact tensor byte lengths. Missing or oversized
tensors cannot silently become partial parameter copies.

`upload_audio_amp` reuses bounded chunk upload and explicit borrow revocation;
failed uploads leave caller-owned partial destination storage unusable until a
full successful retry. Use destination offset zero for the current AMP binder.
Fourteen affected host/upload tests pass, including 126-region offset matching,
missing-tensor rejection and whole-tensor length enforcement. No real checkpoint
upload or GPU execution is established by these tests. Full decoder orchestration
and physical validation remain incomplete.

## Audio upsampling device connection — 2026-09-22

`AudioUpsampleLayout` binds the existing normalized transposed convolution into
two device launches with checked F32 input/output/weight extents. Its normalization
grid follows input channels, not the output-channel axis used by AMP's ordinary
convolutions. The matching offline source bundle exposes normalization and
convolution symbols for each geometry.

Thirty device/convolution tests plus two integration-source tests pass. The
seven-stage layout regression grows time from 207 to 165600, checks channel
halving and distinct g/bias extents, and rejects empty output geometry. These
remain host tests, not a GPU campaign. Full upsample/AMP orchestration, final
activation/convolution/tanh and physical checkpoint validation remain open.

## Composite audio decode stage — 2026-09-22

`AudioDecodeStage` now owns a prepared upsampler and complete AMP body, composing
their 2+100 launches. Construction joins batch/channel/time geometry before
loading functions. One budget covers both the upsampler output/normalized-weight
storage and the AMP scratch allocation, rather than applying the same budget
independently to each component.

Binding consumes external input/output plus separate upsampler and AMP weights.
Upsampler weights use contiguous F32 g/v/bias at zero; a checkpoint upload helper
for that packing is still needed. Partial preparation retains explicit ownership,
and close requires consuming queues to have released their leases first.
Eight program tests pass, including exact combined-budget and mismatched-time
regressions. Seven-stage chain orchestration, waveform tail and physical execution
remain incomplete; this is not a complete AudioVAE claim.

## Seven-stage audio decode chain — 2026-09-22

`AudioDecodeChain` now composes seven prepared upsample/AMP stages into 714
ordered launches. It checks each channel/time/batch edge, owns two aligned
ping-pong intermediate buffers and writes the final stage to caller-owned output.
Inter-stage storage is twice the largest intermediate, not the sum of all stage
outputs. Stage-local scratch remains owned by each prepared stage and is not
included in this inter-stage budget; further sequential scratch sharing is an
optimization opportunity, not currently implemented.

Nine program tests pass, including complete stage count, adjacent byte-extent
continuity, aligned ping-pong sizing and exact budget rejection. The chain
borrows stages and weight allocations; consumers close queues before chain and
stage owners. It begins at conv_pre output and ends before the final alias-free
activation/conv_post/tanh. Those boundaries and physical validation still prevent
a complete AudioVAE inference claim.

## Waveform-tail source and device binding — 2026-09-22

**Superseded numerical assumption:** the initial six-launch tail below used
BigVGAN defaults. Official H3 explicitly disables conv_post bias and final tanh;
the correction described after this entry is the current implementation.

The AudioVAE tail now has a six-entry source bundle and `AudioTailLayout` device
binder: alias-free activation (three launches), normalized 7-tap channel-to-one
convolution (two), then in-place F32 tanh. The ordinary convolution binder now
supports different input/output channel counts while preserving its existing
equal-channel AMP behavior. Stereo streams remain separate batch entries.

Tail scratch accounts for both activation intermediates, activated input and
normalized weights. Tanh reuses waveform output. The current weight ABI is
contiguous F32 alpha/beta/up-filter/down-filter/g/v/bias; official 8-channel
geometry requires 392 bytes. Dedicated checkpoint packing and prepared tail
ownership still need connecting before complete decoder execution.

Twenty-four device/source host tests pass, including exact tail extents and
source entry coverage. No numerical GPU equivalence or performance is claimed;
front-end projection/conv_pre, tail ownership/weights and physical validation
remain unfinished.

## H3 waveform-tail correction and ownership — 2026-09-22

Inspection of the actual H3 AudioVAE constructor (not BigVGAN defaults) showed
`use_bias_at_final=false` and `use_tanh_at_final=false`. The tail is corrected to
five launches and six checkpoint tensors totaling 388 bytes at eight channels.
The generic convolution renderer has an explicit bias-free numerical contract;
its unused bias pointer operand is never read. H3 no longer clips the waveform
through an unsupported tanh. The generic tanh renderer remains independent and
is not selected by this model.

`AudioTailProgram` now owns admitted functions and scratch with explicit
preparation, binding and retryable release. Thirty-three device/source tests
pass, including absence of tanh and bias reads in generated H3 source. Dedicated
tail checkpoint packing/upload and complete AudioVAE execution remain open.

## Audio decoder checkpoint packing and upload — 2026-09-22

The seven upsamplers and corrected waveform tail now have official-schema
weight packing, whole-tensor host resolution and bounded device upload.
Upsamplers pack per-input-channel normalization magnitudes, convolution weights
and output bias. The tail packs only alpha/beta, both filters and conv_post g/v;
its exact extent is 388 bytes, with no fabricated bias slot.

Packing is a pure model-level plan. Host resolution checks the component and
complete tensor extents before upload; device programs consume the packed
allocation without checkpoint-name lookups in execution. AMP packing shares
the same whole-tensor resolver. The affected weights, materialization and upload
packages pass 37 native tests, including all seven upsampler shapes, exact tail
offsets, budget rejection and invalid stage indices.

This closes the checkpoint-packing gap recorded above, not the complete
AudioVAE or H3 request. Full decoder frontend composition, conditioning encoders,
VideoVAE execution and numerical/performance validation on GPU remain required.

## AudioVAE decoder composition — 2026-09-22

The official AudioVAE `decode` calls dec_in_proj then BigVGAN directly; its
attention pre_block belongs to encoding, not decoding. The new frontend source
and prepared owner implement F32 32→2048 pointwise projection followed by
normalized 7-tap 2048→1024 convolution. A generic device layout plans the three
launches and disjoint projected/normalized-weight workspace; model integration
supplies H3 dimensions. Frontend checkpoint packing now includes all five real
tensors, totaling 58,998,784 bytes, through the existing bounded upload path.

`AudioDecoder` composes frontend, seven-stage body and corrected tail into 722
ordered launches. It checks adjacent batch/channel/time geometry and owns
bounded bridge storage, while borrowing prepared stages and immutable weights.
The caller owns the consuming executor and must drain and close it before
decoder/chain/stage owners. Its input is already destandardized, channel-major
F32; latent unpacking and destandardization are upstream operations, not silently
folded into this interface.

This establishes complete decoder-body launch composition, not a physically
validated decoder. Weight normalization is still included in the request launch
list and should be moved to retained startup caches for efficient serving.
Current host tests cover frontend extents, packing, source entries and budget
boundaries; CUDA numerical equivalence, lifecycle fault injection and throughput
remain unverified. Full H3 conditioning, VideoVAE and request orchestration are
still incomplete.

## Audio decode request lifecycle — 2026-09-23

`AudioDecodeRequest` now owns bounded waveform output and an eager ordered
executor over the complete prepared decoder. Preparation binds once; submit
enqueues without per-kernel host waits, and poll observes device completion.
Output borrowing is rejected until completion. Partial submission or polling
failure makes the request non-replayable. Cancellation drains before returning;
close releases executor leases before output and retains failed-release owners
for retry. Decoder, stage and weight owners remain borrowed and must outlive the
request. The lifecycle does not grant concurrent reuse of shared decoder scratch.

Host regression tests cover one-shot submission, pending/completed polling,
submission/poll/drain failures, cancellation idempotence, invalid phases and the
public unprepared/closed API. These are lifecycle tests, not physical CUDA fault
injection or waveform numerical validation. GPU correctness/performance and
complete H3 inference remain pending. Work is paused after this module at the
user's request; no following module is started.

## Parallel implementation resumed — 2026-09-26

The user resumed implementation with parallel module ownership: VideoVAE decode,
Qwen3-VL text conditioning and AudioVAE normalized-weight caching. This supersedes
the paused-work note above, not the physical-validation limitations.

Audio startup now prepares retained normalized weights for frontend, all seven
upsamplers, all 126 AMP convolutions and the waveform tail. Aggregate cache
budgeting happens before allocations; partial failures retain owners for cleanup.
Request preparation requires completed caches and binds the original immutable
weight allocations/AMP layouts. This removes 135 normalization launches from
the decoder request list (722 to 587); existing normalization scratch is still
allocated, and retained cache memory is additional. No measured speedup is claimed.

`AudioInputProgram` connects packed `[2,time,32]` denoiser rows through permutation
and channel-major destandardization into the decoder, with two disjoint bounded
regions. Those two launches precede decode in the same executor, without a host
transfer or intermediate host wait. With this bridge the cached request has
589 launches. Statistics and AOT functions remain admitted, borrowed inputs.

Requests now retain an exclusive decoder lease until close, including failed
preparation and cancellation. A second request cannot reuse the same decoder's
scratch, and the decoder cannot close while leased. Preparation/control operations
remain single-host-thread confined. Composition leases also reject two decoder
wrappers borrowing the same frontend/chain/tail or two chains borrowing a stage;
claims are acquired only after successful allocation and released only after
successful storage close. Latent-input bridge storage also retains an
exclusive request lease until close. These controls are lifecycle work, not a new
model check inside individual GPU operations.

### Text encoder and VideoVAE coverage

The text-only Qwen3-VL extractor now has a pure 50-layer plan, exact per-layer
weight packing and bounded upload, causal 64Q/8KV attention, QKNorm/RoPE,
projection/MLP/residual source and a 651-launch binding. Output is unnormalized
`hidden_states[50]` at `[rows,5120]`, not a language-model logits result. It omits
unused upper layers, final norm and head in execution. Whole-checkpoint host
materialization still loads unused tensors, however. Config/rotary preparation,
resource-owning text request integration and vision/deepstack remain missing.
The new attention implementation is correctness-grade scalar CUDA, not an
efficient tiled kernel or a demonstrated throughput improvement.

VideoVAE now implements its actual ViT frontend: post-quant projection, fused
channel-major packing/embedding, four learned registers and zero CLS. Five
checkpoint tensors have exact packing, host resolution and bounded upload.
The next attention-stem slice implements affine F32 RMSNorm, biased autocast
BF16 Q/K/V, three-axis normalized-coordinate rotary tables and affine-free
QK RMSNorm with 48-of-64 NeoX rotation. A seven-tensor packing can select any of
the 36 official blocks. Softmax attention, output projection/residual, MLP, full
36-block chaining, output head/unpacking and temporal/spatial tiling are still
absent. These are executable source/layout/binding modules, not proof of a
complete VideoVAE forward pass.

All three parallel workstreams require actual CUDA compilation, independent
numerical comparisons, native/GPU lifecycle checks and performance measurements
before claims of complete or efficient H3 inference. No GPU workload was run in
this implementation round; the documented Spark service reservation is unchanged.

The public VAE input binder now rejects overlapping live read/write regions,
including statistics overlapping scratch or output. The normal request path
already owns fresh disjoint storage; the explicit check also protects direct
binder callers. No interval end-offset addition is needed, avoiding overflow.

Final combined validation: 100/100 affected native tests pass, scoped native
checking passes with the existing migration-warning exclusions
`-79-20-92-29-25`, formatting and `git diff --check` pass. `moon info` regenerates
interfaces with zero errors and 4,162 repository migration warnings; this is
not a warning-clean aggregate release or physical CUDA validation.
