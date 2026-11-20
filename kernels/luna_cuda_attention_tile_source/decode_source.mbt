///|
/// Emits the compiler-selected grouped-query decode schedule. Each subgroup
/// owns one query head while all subgroups reuse the K/V tile for their shared
/// key/value head. The online-softmax fold remains subgroup-local.
fn emit_cuda_decode_attention_tile_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  symbol : String,
  maximum_source_bytes : Int32,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  let compute = lowering.compute()
  let effect_plan = cuda_grouped_effect_plan(lowering)
  let copy = lowering.copy()
  let (query_tile, key_value_tile, head_dimension) = lowering.tile()
  let (_, grid_y, block_threads, _) = lowering.launch()
  let (pipeline_stages, vector_bytes) = lowering.pipeline()
  let (query_heads, key_value_heads, abi_head_dimension, tokens_per_page) = abi.shape()
  let (input_row_width, page_stride_values, physical_page_capacity) = abi.layout()
  let (maximum_query_rows, maximum_query_tokens, maximum_page_entries) = abi.limits()
  let group_size = query_heads / key_value_heads
  if compute != CudaGroupedQuerySubgroupShuffleDot ||
    copy != CudaCooperativeSharedLoad ||
    query_tile != 1 ||
    key_value_tile != 64 ||
    head_dimension != abi_head_dimension ||
    query_heads % key_value_heads != 0 ||
    group_size <= 1 ||
    block_threads != group_size * 32 ||
    grid_y != key_value_heads ||
    pipeline_stages != 1 ||
    vector_bytes != 8 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let query_width = query_heads * head_dimension
  let key_fold = cuda_key_fold(lowering)
  let accumulator_items = key_fold.values_per_owner(32).unwrap()
  let source =
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|#define LF_BLOCK_THREADS \{block_threads}
    $|#define LF_QUERY_HEADS_PER_BLOCK \{group_size}
    $|#define LF_KEY_VALUE_TILE 64
    $|#define LF_HEAD_DIMENSION \{head_dimension}
    $|#define LF_PIPELINE_STAGES 1
    $|
    $|static __device__ __forceinline__ float lf_decode_warp_sum(float value) {
    $|  for (int offset = 16; offset > 0; offset >>= 1)
    $|    value += __shfl_down_sync(0xffffffffU, value, offset);
    $|  return __shfl_sync(0xffffffffU, value, 0);
    $|}
    $|
    $|extern "C" __global__ void \{symbol}(
    $|    const int *counts, const int *query_positions,
    $|    const int *query_row_offsets, const int *sequence_lengths,
    $|    const int *page_table_offsets, const int *page_table_page_indices,
    $|    const __nv_bfloat16 *rotated_qkv_rows, __nv_bfloat16 *output,
    $|    const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache) {
    $|  const int thread = (int)threadIdx.x;
    $|  const int lane = thread & 31;
    $|  const int warp = thread >> 5;
    $|  const int local_row = (int)blockIdx.x;
    $|  const int key_value_head = (int)blockIdx.y;
    $|  const int query_head =
    $|      key_value_head * LF_QUERY_HEADS_PER_BLOCK + warp;
    $|  if (blockDim.x != LF_BLOCK_THREADS ||
    $|      key_value_head >= \{key_value_heads} ||
    $|      warp >= LF_QUERY_HEADS_PER_BLOCK || query_head >= \{query_heads}) return;
    $|
    $|  const int prefill_count = counts[0];
    $|  const int decode_count = counts[1];
    $|  const int row_count = counts[2];
    $|  const int token_count = counts[3];
    $|  const int page_entry_count = counts[4];
    $|  if (prefill_count < 0 || decode_count <= 0 ||
    $|      row_count != prefill_count + decode_count ||
    $|      row_count > \{maximum_query_rows} || token_count <= 0 ||
    $|      token_count > \{maximum_query_tokens} || page_entry_count <= 0 ||
    $|      page_entry_count > \{maximum_page_entries} || local_row >= decode_count)
    $|    return;
    $|
    $|  const int row = prefill_count + local_row;
    $|  const int token = query_row_offsets[row];
    $|  const int row_end = query_row_offsets[row + 1];
    $|  const int sequence_length = sequence_lengths[row];
    $|  const int query_position = token >= 0 && token < token_count
    $|      ? query_positions[token] : -1;
    $|  const int page_start = page_table_offsets[row];
    $|  const int page_end = page_table_offsets[row + 1];
    $|  const int output_offset = token * \{query_width} +
    $|      query_head * LF_HEAD_DIMENSION;
    $|  if (token < 0 || row_end != token + 1 || row_end > token_count ||
    $|      query_position < 0 || query_position >= sequence_length ||
    $|      page_start < 0 || page_start >= page_end ||
    $|      page_end > page_entry_count ||
    $|      page_end - page_start !=
    $|          (sequence_length + \{tokens_per_page - 1}) / \{tokens_per_page}) {
    $|    if (token >= 0 && token < token_count && lane == 0)
    $|      output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);
    $|    return;
    $|  }
    $|
    $|  extern __shared__ __align__(16) unsigned char shared_arena[];
    $|  __nv_bfloat16 *key_tiles =
    $|      reinterpret_cast<__nv_bfloat16 *>(shared_arena);
    $|  __nv_bfloat16 *value_tiles = key_tiles +
    $|      LF_PIPELINE_STAGES * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|  int *tile_valid = reinterpret_cast<int *>(value_tiles +
    $|      LF_PIPELINE_STAGES * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION);
    $|  const unsigned long long query_offset =
    $|      (unsigned long long)token * \{input_row_width}ULL +
    $|      (unsigned long long)query_head * LF_HEAD_DIMENSION;
    $|  const float scale = rsqrtf((float)LF_HEAD_DIMENSION);
    $|  float maximum = -CUDART_INF_F;
    $|  float denominator = 0.0f;
    $|  float accumulator[\{accumulator_items}];
    $|  #pragma unroll
    $|  for (int item = 0; item < \{accumulator_items}; ++item)
    $|    accumulator[item] = 0.0f;
    $|
    $|  for (int key_base = 0, tile_index = 0; key_base <= query_position;
    $|       key_base += LF_KEY_VALUE_TILE, ++tile_index) {
    $|    const int stage = tile_index & (LF_PIPELINE_STAGES - 1);
    $|    const int active_keys = min(LF_KEY_VALUE_TILE, query_position + 1 - key_base);
    $|\{cuda_grouped_effects(effect_plan, AcquireKeyOperands, cuda_cooperative_decode_transfer(tokens_per_page, physical_page_capacity, page_stride_values, key_value_heads), "    ")}
    $|    const __nv_bfloat16 *stage_keys = key_tiles +
    $|        stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    const __nv_bfloat16 *stage_values = value_tiles +
    $|        stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    for (int key = 0; key < active_keys; ++key) {
    $|      float partial = 0.0f;
    $|      for (int component = lane; component < LF_HEAD_DIMENSION;
    $|           component += 32)
    $|        partial += __bfloat162float(rotated_qkv_rows[query_offset + component]) *
    $|            __bfloat162float(stage_keys[key * LF_HEAD_DIMENSION + component]);
    $|      const float score = lf_decode_warp_sum(partial) * scale;
    $|\{cuda_key_scale_source(key_fold)}
    $|      #pragma unroll
    $|      for (int item = 0; item < \{accumulator_items}; ++item) {
    $|        const int component = lane + item * 32;
    $|        if (component < LF_HEAD_DIMENSION)
    $|          accumulator[item] = accumulator[item] * old_scale + new_scale *
    $|              __bfloat162float(stage_values[
    $|                  key * LF_HEAD_DIMENSION + component]);
    $|      }
    $|    }
    $|\{cuda_grouped_effects(effect_plan, ReleaseTileReaders, "", "    ")}
    $|  }
    $|
    $|  denominator = __shfl_sync(0xffffffffU, denominator, 0);
    $|  #pragma unroll
    $|  for (int item = 0; item < \{accumulator_items}; ++item) {
    $|    const int component = lane + item * 32;
    $|    if (component < LF_HEAD_DIMENSION)
    $|      output[output_offset + component] = __float2bfloat16_rn(
    $|          denominator > 0.0f ? accumulator[item] / denominator
    $|                             : CUDART_NAN_F);
    $|  }
    $|}
  let bytes = @utf8.encode(source)
  if bytes.length() > maximum_source_bytes {
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
