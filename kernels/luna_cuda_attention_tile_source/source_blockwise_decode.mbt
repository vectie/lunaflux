///|
/// One subgroup per query head, shared immutable K/V per GQA group. Unlike
/// per-key decode, probabilities and PV are formed as a block and the running
/// state is rescaled once per key tile. The alternative law is explicit in
/// the physical program, symbol and recipe; this is not a silent fallback.
fn emit_cuda_blockwise_decode_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  requested_symbol : String,
  maximum_source_bytes : Int,
  fold : @attention_physical_ir.BlockwiseFold,
  effects : @luna_attention_tile_schedule.AttentionGroupedEffectPlan,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  let (query_tile, keys, dimension) = lowering.tile()
  let (_, grid_y, threads, shared_bytes) = lowering.launch()
  let (heads, kv_heads, abi_dimension, page_tokens) = abi.shape()
  let (input_width, _, _) = abi.layout()
  let (max_rows, max_tokens, max_pages) = abi.limits()
  let group = heads / kv_heads
  let required_shared = (keys * dimension * 4 + group * keys * 4 + 16).to_int64()
  guard !abi.query_metadata &&
    lowering.phase() == Decode &&
    query_tile == 1 &&
    lowering.compute() == CudaGroupedQueryBlockwiseShuffleDot &&
    !effects.is_pipelined() &&
    (keys == 32 || keys == 64) &&
    dimension == abi_dimension &&
    fold.dimensions() == (dimension, keys) &&
    heads % kv_heads == 0 &&
    threads == group * 32 &&
    grid_y == kv_heads &&
    shared_bytes >= required_shared else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let symbol = requested_symbol + "_blockwise_f32_v1"
  guard valid_cuda_symbol(symbol) else {
    raise InvalidCudaAttentionTileSource(InvalidSymbol)
  }
  let transfer = emit_decode_key_value_transfer(
    abi,
    8,
    hoist_page_lookup=lowering.hoists_paged_key_value_page_lookup(),
  )
  let acquire = cuda_grouped_effects(
    effects,
    AcquireKeyOperands,
    transfer,
    "    ",
  )
  let release = cuda_grouped_effects(effects, ReleaseTileReaders, "", "    ")
  let body = cuda_blockwise_fold_body(fold)
  let items = (dimension + 31) / 32
  let source =
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|// numeric_law=\{fold.numeric_law()}
    $|#define LF_BLOCK_THREADS \{threads}
    $|#define LF_KEY_VALUE_TILE \{keys}
    $|#define LF_HEAD_DIMENSION \{dimension}
    $|#define LF_ACCUMULATOR_ITEMS \{items}
    $|extern "C" __global__ void \{symbol}(
    $|    const int *counts, const int *query_positions,
    $|    const int *query_row_offsets, const int *sequence_lengths,
    $|    const int *page_table_offsets, const int *page_table_page_indices,
    $|    const __nv_bfloat16 *rotated_qkv_rows, __nv_bfloat16 *output,
    $|    const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache) {
    $|  const int thread = (int)threadIdx.x;
    $|  const int lane = thread & 31;
    $|  const int warp = thread >> 5;
    $|  const int key_value_head = (int)blockIdx.y;
    $|  const int query_head = key_value_head * \{group} + warp;
    $|  const int local_row = (int)blockIdx.x;
    $|  if (blockDim.x != LF_BLOCK_THREADS || key_value_head >= \{kv_heads} || warp >= \{group}) return;
    $|  const int prefill_count = counts[0], decode_count = counts[1];
    $|  const int row_count = counts[2], token_count = counts[3], page_entry_count = counts[4];
    $|  if (prefill_count < 0 || decode_count <= 0 || row_count != prefill_count + decode_count ||
    $|      row_count > \{max_rows} || token_count <= 0 || token_count > \{max_tokens} ||
    $|      page_entry_count <= 0 || page_entry_count > \{max_pages} || local_row >= decode_count) return;
    $|  const int row = prefill_count + local_row;
    $|  const int token = query_row_offsets[row], row_end = query_row_offsets[row + 1];
    $|  const int sequence_length = sequence_lengths[row];
    $|  const int query_position = token >= 0 && token < token_count ? query_positions[token] : -1;
    $|  const int page_start = page_table_offsets[row], page_end = page_table_offsets[row + 1];
    $|  if (token < 0 || row_end != token + 1 || row_end > token_count || query_position < 0 ||
    $|      query_position >= sequence_length || page_start < 0 || page_start >= page_end ||
    $|      page_end > page_entry_count || page_end - page_start !=
    $|          (sequence_length + \{page_tokens - 1}) / \{page_tokens}) return;
    $|  extern __shared__ __align__(16) unsigned char shared_arena[];
    $|  __nv_bfloat16 *key_tile = reinterpret_cast<__nv_bfloat16 *>(shared_arena);
    $|  __nv_bfloat16 *value_tile = key_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|  float *scores = reinterpret_cast<float *>(value_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION) + warp * LF_KEY_VALUE_TILE;
    $|  int *tile_valid = reinterpret_cast<int *>(value_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION) + \{group * keys};
    $|  const unsigned long long query_offset = (unsigned long long)token * \{input_width}ULL + query_head * LF_HEAD_DIMENSION;
    $|  float query_fragment[LF_ACCUMULATOR_ITEMS], accumulator[LF_ACCUMULATOR_ITEMS] = {};
    $|  #pragma unroll
    $|  for (int item = 0; item < LF_ACCUMULATOR_ITEMS; ++item) {
    $|    const int component = lane + item * 32;
    $|    query_fragment[item] = component < LF_HEAD_DIMENSION ? __bfloat162float(rotated_qkv_rows[query_offset + component]) : 0.0f;
    $|  }
    $|  const float scale = rsqrtf((float)LF_HEAD_DIMENSION);
    $|  float maximum = -CUDART_INF_F, denominator = 0.0f;
    $|  const int key_limit = query_position + 1;
    $|  for (int key_base = 0; key_base < key_limit; key_base += LF_KEY_VALUE_TILE) {
    $|    const int active_keys = min(LF_KEY_VALUE_TILE, key_limit - key_base);
    $|\{acquire}
    $|\{body}
    $|\{release}
    $|  }
    $|  const unsigned long long output_offset = (unsigned long long)token * \{heads * dimension}ULL + query_head * LF_HEAD_DIMENSION;
    $|  #pragma unroll
    $|  for (int item = 0; item < LF_ACCUMULATOR_ITEMS; ++item) {
    $|    const int component = lane + item * 32;
    $|    if (component < LF_HEAD_DIMENSION)
    $|      output[output_offset + component] = __float2bfloat16_rn(accumulator[item] / denominator);
    $|  }
    $|}
    $|#undef LF_BLOCK_THREADS
    $|#undef LF_KEY_VALUE_TILE
    $|#undef LF_HEAD_DIMENSION
    $|#undef LF_ACCUMULATOR_ITEMS
  let bytes = @utf8.encode(source)
  guard bytes.length() <= maximum_source_bytes else {
    raise InvalidCudaAttentionTileSource(SourceLimit)
  }
  {
    lowering_digest: lowering.digest(),
    abi,
    symbol,
    bytes,
    digest: CudaAttentionTileSourceDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    ),
  }
}
