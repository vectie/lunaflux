# Joint diffusion device frames

Adapts existing device ordered executors to prepared diffusion frames. Eager
queues enqueue every prebound launch then wait once at frame completion;
captured queues launch their existing graph. No per-kernel synchronization,
queue reconstruction, coefficient upload or host readback is added by this
adapter. Queues are single-use per request and own leases, not the underlying
model/latent allocations. Close the execution owner before those allocations.

`KernelFrame::new` accepts already admitted functions, geometry and argument
regions. It is not a replacement for startup model/kernel binding. Partial
startup construction must release earlier frames through the prepared-frame
drain/release methods if a later constructor fails. Full H3 graph binding and
physical CUDA validation remain incomplete.

`FlowBindings::upload` uploads the prepared immutable coefficient table into
a caller-owned allocation once, then exposes fixed startup argument regions:
12-byte RF coefficients and 4-byte normalized timesteps. All steps share that
allocation through queue leases. It must not be overwritten or closed while
frames exist. This does not bind packed conditioning timestep tables or allocate
the denoiser's activations. Upload failures retain caller ownership for cleanup.

`LatentStoragePlan` calculates a bounded, 256-byte-aligned arena before device
execution. `LatentStorage` owns one allocation for video/audio states, velocity
outputs, boundary permutation buffers and the immutable coefficient table.
Stereo streams are included in every audio region. No allocation or host copy
is required to advance a denoise step. Contents initially remain uninitialized:
the caller must prepare inputs and upload coefficients before submission.
Host transfer methods are for startup/terminal boundaries only. Close frame
queues first, then close the arena; a busy/error result retains ownership for
retry. This allocation excludes weights, conditioning, hidden states and VAE
workspace and is not a total-model memory estimate.

Boundary maps now belong to that same storage plan. `boundary_launch` binds
the exact input/output regions for video input packing, video output unpacking
and audio output unpacking. These are out-of-place, bit-preserving permutations;
input and output cannot alias. The caller supplies an admitted function compiled
from the matching boundary permutation and block size. Audio initial noise is
already token-major and goes directly into `AudioState`; no input transpose is
inserted for it. Output regions are VAE latent inputs, not decoded media.

`KernelFrame::denoise` snapshots the supplied joint prediction launches, then
appends video and audio RF updates. Both predictions therefore precede either
update on the ordered stream. The prediction list must already bind the correct
timestep and write both velocity regions without modifying the states. This
constructor deliberately does not replace the missing complete model binder.
Preparing boundary and denoise launch descriptors performs no device work;
execution and deterministic release remain explicit through prepared frames.

`initialize_noise` fills the preallocated video boundary and audio state using
bounded native F32 chunks before execution. Its explicit chunk limit avoids a
full host latent temporary. Video must then be packed before denoising. It is
not CUDA/PyTorch RNG-compatible: use identical externally supplied initial
latents for model comparisons. Copy failure retains the storage owner and
requires a full retry or close; never submit partially initialized storage.
This host startup path is not a measured GPU initialization optimization.

`LatentProjections` allocates both BF16 latent-input projection outputs once.
It borrows the latent storage and externally owned F32 weights/bias. `launch`
binds their exact regions to the existing 256-thread dense projection ABI;
the admitted function must match the modality shape. Both outputs may be
reused between steps only after consumers complete. Release queues, then this
output allocation, then the latent arena. This does not implement conditioning
projection, transformer blocks, or a tuned high-throughput GEMM.

`PackedHidden` owns packed BF16 output and a row-map region in one allocation.
Upload the immutable map once before any frame, then reuse its assembly launch
each step after projection. Its explicit positions, including padding, must
come from the family request builder; no fixed modality order is assumed.
Projection extents/width are checked during binding. Conditioning input is
already projected BF16, not text tokens or raw media. Release queued leases
before closing this owner. The allocation/upload split preserves cleanup
ownership when upload fails. Physical execution is not yet validated.
