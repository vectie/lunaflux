# MiniMax H3 reference-media encoding

This package owns executable reference AudioVAE and VideoVAE request bindings,
not a missing-capability assessment. CUDA artifacts are compiled offline from
`AudioSource.kernels.source` or `VideoSource.kernels.source` and admitted by the
caller before `prepare` loads the complete symbol table.

Audio accepts stereo channel-major F32 PCM at 32 kHz. It pads to the 800-sample
hop, runs all five DAC convolution/residual stages, the causal attention
projection, its head mean and adaptive average pool, the GeGLU branch, and
`mean_proj`. It returns normalized F32 `[2*T,32]` rows. It does not sample the
audio posterior or evaluate `logs_proj`.

Video accepts resized/cropped F32 ImageNet-normalized `[3,T,H,W]` pixels;
`upload_rgb` converts a shared RGB24 payload in bounded chunks without a second
complete host F32 copy. The immutable graph runs every encoder CNN block over
complete spatial tiles, blends moments vertically before horizontally, pads
17-frame clips by repeating the last frame, drops the final three latent
positions for video, and samples the diagonal Gaussian. Images use one frame
without temporal token drop. Sampling consumes an **explicit** F32 standard
normal array `[24,latentT,H/16,W/16]`; the official reference uses a scoped
PyTorch CPU seed-42 draw. This package does not claim that another generator
with the same seed is byte-equivalent. Sampled latents round through FP16 before
normalization and `[1,2,2]` patch packing into F32 width-96 rows.

Both request owners borrow a completed `SegmentedDeviceWeights` upload and
match its tensor offsets against the exact host component manifest. They own
the execution queue, function handles, statistics and liveness-planned scratch.
Model startup captures `WeightManifest::new(host)` before releasing host
staging. `prepare_resolved` accepts that immutable metadata snapshot and the
resident upload; it does not retain the host weight arena. The `prepare` entry
is a convenience wrapper for callers that still hold live staging.
They implement `PreparedFrame`; `execute` submits once and waits once. `submit`
and `poll` provide the asynchronous alternative. `with_destination` is solely
for binding an ordered downstream consumer during preparation; `with_rows`
requires observed completion. Close downstream queues first, then requests,
then borrowed model/module/input/noise/stream/context owners. Failed preparation
or partial enqueue retains ownership for retryable `close`, never replay.

These are complete scalar correctness-oriented CUDA sources and MoonBit
bindings. Native plan/source/lifecycle tests do not establish CUDA compilation,
GPU numerical tolerance, sanitizer/leak safety, throughput, image quality or
end-to-end H3 validation. Tile/GEMM optimization and physical qualification
remain required before describing the path as efficient production inference.
