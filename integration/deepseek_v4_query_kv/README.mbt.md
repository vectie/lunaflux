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
# Learned compressor pooling

The next learned-index execution edge uses a separate model-neutral query plan:
packed E4M3/UE8M0 projection of the already-normalized query low rank, BF16
head-weight projection from hidden states, BF16 head scaling, then the existing
suffix rotary/normalized Hadamard/block-32 E2M1 simulation. The adapter owns
checkpoint names only. All six effects belong to the decoder's existing queue;
head reduction, causal selection and complete compressed attention remain
separate consumers, not implicitly replaced by the window path.

`DeepSeekLearnedIndexer` now composes learned queries, the distinct indexer
compressor/rotary/Hadamard/FP4 retained cache, weighted score reduction and
causal top-k. Nineteen prepared effects share the existing decoder queue.
Its outputs are compressed row IDs/counts with the caller's per-query offset;
they are not raw token positions. Every component's scratch, retained state
and compact checkpoint banks counts against the aggregate preparation budget.
This is the full-head single-rank indexer, not tensor-parallel head reduction,
compressed attention or a complete DeepSeek/DSpark model runner.

`DeepSeekCompressorPool` binds attention/indexer-specific BF16 projection and
normalization banks and F32 position biases to the shared learned pooling plan.
Its five prepared effects join the caller's queue, expose compact output counts,
positions and the sticky append error, and retain no filesystem authority after
startup. Close the borrowing queue, this owner, then caller input ports.
Compression-zero layers and indexer requests outside ratio four are rejected.
These normalized outputs are before rotary, cache conversion/publication and
index selection; this owner alone is not complete compressed attention.

`DeepSeekAttentionCompressor` composes that learned pool with standalone
group-start rotary and non-rotary-prefix E4M3/power-of-two simulation. Seven
effects share the containing queue, and the transformed BF16 values/counts/
positions remain borrowed by the downstream persistent cache owner. The plan
uses compact row capacity, preserves the rotated suffix and accounts for one
additional output allocation. It does not substitute this operation for the
indexer's distinct Hadamard/FP4 transformation. Persistent compressed-cache
publication, selection and whole-model generation remain open.
