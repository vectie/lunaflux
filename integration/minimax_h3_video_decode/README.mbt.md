# MiniMax H3 VideoVAE decoder

This executable two-launch frontend consumes **destandardized** F32
`[B,24,T,H,W]` for one selected spatial/temporal tile. It performs the learned
post-quantization 1x1x1 convolution, then projects into F32 decoder rows, appends
four learned register rows and one all-zero class row. The pack is fused with
the embedding rather than materializing another full latent transpose.

Official checkpoint names are mapped in `video_frontend_weight_packing`;
`decoder.proj_in` is the serialized equivalent of the upstream `x_embedder`.
`resolve_video_frontend_weights` resolves whole tensors only from the admitted
VideoVAE component; `upload_video_frontend` transfers bounded chunks to the
239,968-byte execution layout.
The arithmetic retains an intermediate F32 result between projections: folding
the two weight matrices would change rounding and is not performed.

Compile the returned source offline and admit the matching module before
`FrontendProgram.prepare`. Bind returns two ordered launches; the downstream
decoder can borrow `with_output` solely for queue construction. The owner does
not claim GPU completion. Close consuming queues first, then close this owner,
then its module. Partial acquisitions survive prepare/release failure for retry.

`AttentionStem` now binds four real launches for a decoder block's attention
prefix: F32 affine RMSNorm, biased autocast-BF16 Q/K/V, a 3D normalized-coordinate
rotary table, and affine-free head RMSNorm plus partial NeoX rotation. It rotates
48 of 64 channels and leaves 16 unchanged; register/CLS positions are zero.
Input hidden states and checkpoint weights remain F32. Q/K normalization, rotary
coefficient, multiply and output rounding retain explicit BF16 boundaries.
Its planar output is `[3,rows,32,64]`, not the upstream fused-QKV physical order.
The caller provides separate scratch/output and admitted functions for queue
binding. Weight packing covers one layer selected from the official 36.

The subsequent `Block` binds noncausal online-softmax attention (linear scratch,
not an allocated score matrix), biased BF16 output projection, F32 scaled
residual, second F32 RMSNorm, biased BF16 SwiGLU and down projection, and the
second scaled residual. `TileDecoderProgram` composes the frontend, all 36
checkpoint-indexed blocks, final **LayerNorm**, F32 output projection and direct
patch unpacking into 400 ordered launches. It reuses one block scratch allocation
and two F32 hidden buffers across the entire layer stack. Block weights are
packed/uploaded separately, preserving all 16 official per-layer tensors.

`TiledDecoderProgram` adds actual latent gather, spatial assemble and temporal
publish kernels around that reusable tile decoder. Spatial overlap follows the
released 16-pixel overlap redistribution, vertical-before-horizontal blending
against raw neighbor tiles, then non-overlapping cropped publication. Temporal
decode uses 7-token clips at stride 5, repeats the last latent for padding,
removes three initial frames per subclip, blends five overlap frames and removes
padding from the tail. Requested shorter results retain the causal **last**
frames, rather than the first frames. The 37-latent-frame geometry produces
124 frames. Only two decoded temporal clips and one raw spatial tile collection
are live; latent gathers and the tile decoder workspace are reused sequentially.

The complete plan reports input/output byte extents, output dimensions, temporal
crop and total working-set budget. `bind` consumes a destandardized input region
(optional byte offset), three groups of admitted weights, and returns one ordered
launch list for the outer request pipeline. `with_pixels` borrows normalized F32
`[3,frames,height,width]` output; callers must observe their queue's completion
before readback and close consuming queues before any owner. ImageNet inverse
normalization/media encoding is a separate output boundary, not applied twice
inside the VAE. Public planners reject excessive job/working-set bounds before
allocating working tensors.

Native tests cover planning, schema layouts and generated source, not CUDA
numerical equivalence, leaks or performance. These scalar correctness kernels
are not claimed to be efficient tiled lowering. The VideoVAE **encoder** for
image/video reference conditioning remains a separate causal 3D CNN and is not
implemented by this decoder. Full physical decoder comparison remains required.
# Request output

`VideoDecodeBundle` combines the final packed-latent bridge, tiled decoder and
canonical video postprocess into one AOT source. `VideoDecodeRequest` owns their
scratch, function handles, output and ordered executor, exposes one-shot
`submit`/`poll`/`wait` plus deterministic `cancel`/`close`, and implements
`PreparedFrame`. The caller retains immutable checkpoint weights, means/stds,
input, module, stream and context until the request closes.

Output becomes available only after completion as contiguous F32
`[1,3,T,H,W]`. Postprocessing follows the reference ImageNet inverse Normalize
operation order (F32 subtract, then divide), NaN-preserving clamp to `[0,1]`,
and top-left target-canvas crop. It does not introduce a new interleaved layout.
Failures never replay a partially submitted queue; close drains before release
and retains any unsuccessfully released resource for retry.
