# Checkpoint-backed quantized query/KV prefix

This adapter supplies official DeepSeek tensor names and model dimensions to
the model-neutral `QuantizedQueryKvPrecision` plan. CUDA details remain in
`kernels/quantized_query_kv_cuda_source`; prepared effects and deterministic
release remain in `integration/quantized_query_kv_frame`.

The six stages are Query-A, learned query RMSNorm, Query-B, unweighted headwise
normalization, shared-KV projection, and learned KV RMSNorm. Projection matrices
retain E4M3 payload and block-128 UE8M0 scales; norm vectors retain BF16. The
unweighted head norm preserves the reference's BF16 square/mean/epsilon/inverse/
product boundaries, unlike learned RMSNorm's F32 arithmetic and final BF16 round.

Preparation accounts for all five compact checkpoint banks plus six scratch
buffers before upload. The caller borrows the six launches into its decoder
queue; there is no extra per-layer submission or completion. Release the queue
first, then this owner, then its external operands/context. Partial preparation
can be closed deterministically. No file reads, authentication, source generation
or scratch allocation occur in the token-step path.

`DeepSeekAttentionInputs` extends this prefix with a model-neutral
`RotaryKvPrecision` plan and prepared rotary/KV frame. It borrows all eight
launches into the same executor. The adapter selects base theta without YaRN
for compression-zero layers, compressed theta with YaRN otherwise. Both Q
and KV use full-head storage; only the rotary suffix changes under rotation.
KV's non-rotary prefix undergoes block-64 E4M3/power-of-two simulation and
returns to BF16, preserving the positional suffix. The combined constructor
and preparation account for all eight scratch buffers and compact weights.

`DeepSeekWindowAttention` additionally binds the checkpoint's F32 `attn_sink`
and request-owned shared-KV ring. Twelve launches share the caller's executor:
the eight input operations, then reserve, causal sink attention, bit-copy and
history publication. Chunked prefill can exceed the window; old slots are not
overwritten until all queries consume them. Its append descriptor joins the
containing decoder's error boundary. The adapter explicitly rejects compressed
layers, rather than substituting a window-only answer.

`DeepSeekWindowSublayer` extends this to fifteen launches with inverse suffix
RoPE, grouped Output-A and Output-B. It binds compact checkpoint payload/scales
for both learned projections. Output-A uses BF16-rounded parameters with F32
accumulation; Output-B uses dynamic block-128 FP8 activation scaling. These
semantics live in immutable precision plans rather than a family branch in
the CUDA renderer. Its hidden-width output and append descriptor join the
containing decoder; aggregate weights/scratch are checked before upload.

These remain single-rank components, **not a DeepSeek/DSpark model runner**.
Compressed cache/compressor/indexer, mHC/decoder composition and distributed
head ownership remain subsequent integration work. No whole-model accuracy
or performance claim follows from the small numerical fixtures.
