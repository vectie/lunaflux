///|
/// Emits the first production-shaped matrix-tiled prefill kernel from the
/// functional schedule. K/V tiles are loaded once and reused by a complete
/// query tile; online softmax and PV accumulation remain fused.
pub fn emit_cuda_attention_tile_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  symbol : String,
  maximum_source_bytes : Int,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  if !valid_cuda_symbol(symbol) {
    raise InvalidCudaAttentionTileSource(InvalidSymbol)
  }
  if maximum_source_bytes <= 0 || maximum_source_bytes > 16777216 {
    raise InvalidCudaAttentionTileSource(SourceLimit)
  }
  if lowering.compute() != CudaMatrixMultiplyAccumulate ||
    lowering.copy() != CudaCooperativeSharedLoad {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let (query_tile, key_value_tile, head_dimension) = lowering.tile()
  let (_, _, block_threads, shared_memory_bytes) = lowering.launch()
  let (pipeline_stages, vector_bytes) = lowering.pipeline()
  let (query_heads, key_value_heads, abi_head_dimension, tokens_per_page) = abi.shape()
  let (input_row_width, page_stride_values, physical_page_capacity) = abi.layout()
  let (maximum_query_rows, maximum_query_tokens, maximum_page_entries) = abi.limits()
  if query_tile != 16 ||
    key_value_tile < 16 ||
    key_value_tile > 128 ||
    key_value_tile % 16 != 0 ||
    head_dimension != abi_head_dimension ||
    head_dimension % 16 != 0 ||
    block_threads < key_value_tile / 16 * 32 ||
    block_threads > 1024 ||
    pipeline_stages < 1 ||
    pipeline_stages > 4 ||
    vector_bytes != 16 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let query_width = query_heads * head_dimension
  let group_size = query_heads / key_value_heads
  let values_per_vector = vector_bytes / 2
  let source =
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|#include <mma.h>
    $|#define LF_QUERY_TILE \{query_tile}
    $|#define LF_KEY_VALUE_TILE \{key_value_tile}
    $|#define LF_HEAD_DIMENSION \{head_dimension}
    $|#define LF_BLOCK_THREADS \{block_threads}
    $|#define LF_SHARED_MEMORY_BYTES \{shared_memory_bytes}
    $|#define LF_PIPELINE_STAGES \{pipeline_stages}
    $|#define LF_VALUES_PER_VECTOR \{values_per_vector}
    $|
    $|static __device__ __forceinline__ float lf_warp_sum(float value) {
    $|  for (int offset = 16; offset > 0; offset >>= 1)
    $|    value += __shfl_down_sync(0xffffffffU, value, offset);
    $|  return __shfl_sync(0xffffffffU, value, 0);
    $|}
    $|
    $|static __device__ __forceinline__ float lf_warp_max(float value) {
    $|  for (int offset = 16; offset > 0; offset >>= 1)
    $|    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
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
    $|  const int warp_count = LF_BLOCK_THREADS >> 5;
    $|  const int query_head = (int)blockIdx.y;
    $|  if (blockDim.x != LF_BLOCK_THREADS || query_head >= \{query_heads}) return;
    $|
    $|  const int prefill_count = counts[0];
    $|  const int decode_count = counts[1];
    $|  const int row_count = counts[2];
    $|  const int token_count = counts[3];
    $|  const int page_entry_count = counts[4];
    $|  if (prefill_count <= 0 || decode_count != 0 || row_count != prefill_count ||
    $|      row_count > \{maximum_query_rows} || token_count <= 0 ||
    $|      token_count > \{maximum_query_tokens} || page_entry_count <= 0 ||
    $|      page_entry_count > \{maximum_page_entries}) return;
    $|
    $|  const int tile_ordinal = (int)blockIdx.x;
    $|  int row = -1, token_base = 0, tile_prefix = 0;
    $|  for (int candidate_row = 0; candidate_row < row_count; ++candidate_row) {
    $|    const int begin = query_row_offsets[candidate_row];
    $|    const int end = query_row_offsets[candidate_row + 1];
    $|    const int row_tiles = end > begin ? (end - begin + LF_QUERY_TILE - 1) / LF_QUERY_TILE : 0;
    $|    if (tile_ordinal < tile_prefix + row_tiles) {
    $|      row = candidate_row;
    $|      token_base = begin + (tile_ordinal - tile_prefix) * LF_QUERY_TILE;
    $|      break;
    $|    }
    $|    tile_prefix += row_tiles;
    $|  }
    $|  if (row < 0) return;
    $|
    $|  const int row_start = query_row_offsets[row];
    $|  const int row_end = query_row_offsets[row + 1];
    $|  const int active_queries = min(LF_QUERY_TILE, row_end - token_base);
    $|  const int page_start = page_table_offsets[row];
    $|  const int page_end = page_table_offsets[row + 1];
    $|  const int sequence_length = sequence_lengths[row];
    $|  if (row_start < 0 || row_start >= row_end || row_end > token_count ||
    $|      active_queries <= 0 || page_start < 0 || page_start >= page_end ||
    $|      page_end > page_entry_count || sequence_length <= 0 ||
    $|      page_end - page_start != (sequence_length + \{tokens_per_page - 1}) / \{tokens_per_page})
    $|    return;
    $|
    $|  extern __shared__ __align__(16) unsigned char shared_arena[];
    $|  unsigned char *shared_cursor = shared_arena;
    $|  __nv_bfloat16 *query_tile = reinterpret_cast<__nv_bfloat16 *>(shared_cursor);
    $|  shared_cursor += LF_QUERY_TILE * LF_HEAD_DIMENSION * sizeof(__nv_bfloat16);
    $|  __nv_bfloat16 *key_tiles = reinterpret_cast<__nv_bfloat16 *>(shared_cursor);
    $|  shared_cursor += LF_PIPELINE_STAGES * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION * sizeof(__nv_bfloat16);
    $|  __nv_bfloat16 *value_tiles = reinterpret_cast<__nv_bfloat16 *>(shared_cursor);
    $|  shared_cursor += LF_PIPELINE_STAGES * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION * sizeof(__nv_bfloat16);
    $|  float *scores = reinterpret_cast<float *>(shared_cursor);
    $|  shared_cursor += LF_QUERY_TILE * LF_KEY_VALUE_TILE * sizeof(float);
    $|  float *weights = reinterpret_cast<float *>(shared_cursor);
    $|  shared_cursor += LF_QUERY_TILE * LF_KEY_VALUE_TILE * sizeof(float);
    $|  float *accumulator = reinterpret_cast<float *>(shared_cursor);
    $|  shared_cursor += LF_QUERY_TILE * LF_HEAD_DIMENSION * sizeof(float);
    $|  float *running_maximum = reinterpret_cast<float *>(shared_cursor);
    $|  shared_cursor += LF_QUERY_TILE * sizeof(float);
    $|  float *running_denominator = reinterpret_cast<float *>(shared_cursor);
    $|  shared_cursor += LF_QUERY_TILE * sizeof(float);
    $|  int &tile_valid = *reinterpret_cast<int *>(shared_cursor);
    $|
    $|  const int query_vector_count = LF_QUERY_TILE * LF_HEAD_DIMENSION / LF_VALUES_PER_VECTOR;
    $|  for (int vector = thread; vector < query_vector_count; vector += LF_BLOCK_THREADS) {
    $|    const int first_value = vector * LF_VALUES_PER_VECTOR;
    $|    const int query = first_value / LF_HEAD_DIMENSION;
    $|    const int component = first_value % LF_HEAD_DIMENSION;
    $|    uint4 packed = make_uint4(0U, 0U, 0U, 0U);
    $|    if (query < active_queries) {
    $|      const unsigned long long source_index =
    $|          (unsigned long long)(token_base + query) * \{input_row_width}ULL +
    $|          (unsigned long long)query_head * LF_HEAD_DIMENSION + component;
    $|      packed = *reinterpret_cast<const uint4 *>(rotated_qkv_rows + source_index);
    $|    }
    $|    *reinterpret_cast<uint4 *>(query_tile + first_value) = packed;
    $|  }
    $|  for (int linear = thread; linear < LF_QUERY_TILE * LF_HEAD_DIMENSION; linear += LF_BLOCK_THREADS)
    $|    accumulator[linear] = 0.0f;
    $|  if (thread < LF_QUERY_TILE) {
    $|    running_maximum[thread] = -CUDART_INF_F;
    $|    running_denominator[thread] = 0.0f;
    $|  }
    $|  __syncthreads();
    $|
    $|  int maximum_query_position = query_positions[token_base];
    $|  if (thread == 0) {
    $|    tile_valid = 1;
    $|    for (int query = 0; query < active_queries; ++query) {
    $|      const int position = query_positions[token_base + query];
    $|      if (position < 0 || position >= sequence_length) tile_valid = 0;
    $|      maximum_query_position = max(maximum_query_position, position);
    $|    }
    $|    scores[0] = __int_as_float(maximum_query_position);
    $|  }
    $|  __syncthreads();
    $|  maximum_query_position = __float_as_int(scores[0]);
    $|  if (!tile_valid) return;
    $|
    $|  const int key_value_vector_count = LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION / LF_VALUES_PER_VECTOR;
    $|  const int key_value_head = query_head / \{group_size};
    $|  const float scale = rsqrtf((float)LF_HEAD_DIMENSION);
    $|  for (int key_base = 0, tile_index = 0; key_base <= maximum_query_position;
    $|       key_base += LF_KEY_VALUE_TILE, ++tile_index) {
    $|    const int stage = tile_index % LF_PIPELINE_STAGES;
    $|    if (thread == 0) tile_valid = 1;
    $|    __syncthreads();
    $|    for (int vector = thread; vector < key_value_vector_count; vector += LF_BLOCK_THREADS) {
    $|      const int first_value = vector * LF_VALUES_PER_VECTOR;
    $|      const int key = first_value / LF_HEAD_DIMENSION;
    $|      const int component = first_value % LF_HEAD_DIMENSION;
    $|      const int key_position = key_base + key;
    $|      uint4 packed_key = make_uint4(0U, 0U, 0U, 0U);
    $|      uint4 packed_value = make_uint4(0U, 0U, 0U, 0U);
    $|      if (key_position < sequence_length && key_position <= maximum_query_position) {
    $|        const int logical_page = key_position / \{tokens_per_page};
    $|        const int page_slot = page_start + logical_page;
    $|        const int physical_page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;
    $|        if (physical_page < 0 || (unsigned long long)physical_page >= \{physical_page_capacity}ULL) {
    $|          atomicExch(&tile_valid, 0);
    $|        } else {
    $|          const int token_in_page = key_position % \{tokens_per_page};
    $|          const unsigned long long cache_index =
    $|              (unsigned long long)physical_page * \{page_stride_values}ULL +
    $|              ((unsigned long long)token_in_page * \{key_value_heads}ULL + key_value_head) *
    $|              LF_HEAD_DIMENSION + component;
    $|          packed_key = *reinterpret_cast<const uint4 *>(key_cache + cache_index);
    $|          packed_value = *reinterpret_cast<const uint4 *>(value_cache + cache_index);
    $|        }
    $|      }
    $|      __nv_bfloat16 *stage_keys = key_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|      __nv_bfloat16 *stage_values = value_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|      *reinterpret_cast<uint4 *>(stage_keys + first_value) = packed_key;
    $|      *reinterpret_cast<uint4 *>(stage_values + first_value) = packed_value;
    $|    }
    $|    __syncthreads();
    $|    if (!tile_valid) return;
    $|
    $|    if (warp < LF_KEY_VALUE_TILE / 16) {
    $|      using namespace nvcuda;
    $|      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> query_fragment;
    $|      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> key_fragment;
    $|      wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_fragment;
    $|      wmma::fill_fragment(score_fragment, 0.0f);
    $|      const int key_column = warp * 16;
    $|      #pragma unroll
    $|      for (int reduction = 0; reduction < LF_HEAD_DIMENSION; reduction += 16) {
    $|        wmma::load_matrix_sync(query_fragment, query_tile + reduction, LF_HEAD_DIMENSION);
    $|        wmma::load_matrix_sync(key_fragment,
    $|            key_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +
    $|                key_column * LF_HEAD_DIMENSION + reduction,
    $|            LF_HEAD_DIMENSION);
    $|        wmma::mma_sync(score_fragment, query_fragment, key_fragment, score_fragment);
    $|      }
    $|      wmma::store_matrix_sync(scores + key_column, score_fragment,
    $|          LF_KEY_VALUE_TILE, wmma::mem_row_major);
    $|    }
    $|    __syncthreads();
    $|
    $|    for (int linear = thread; linear < LF_QUERY_TILE * LF_KEY_VALUE_TILE;
    $|         linear += LF_BLOCK_THREADS) {
    $|      const int query = linear / LF_KEY_VALUE_TILE;
    $|      const int key = linear % LF_KEY_VALUE_TILE;
    $|      const int key_position = key_base + key;
    $|      const int query_position = query < active_queries ? query_positions[token_base + query] : -1;
    $|      scores[linear] = query < active_queries && key_position <= query_position
    $|          ? scores[linear] * scale : -CUDART_INF_F;
    $|    }
    $|    __syncthreads();
    $|
    $|    for (int query = warp; query < active_queries; query += warp_count) {
    $|      float local_maximum = -CUDART_INF_F;
    $|      for (int key = lane; key < LF_KEY_VALUE_TILE; key += 32)
    $|        local_maximum = fmaxf(local_maximum, scores[query * LF_KEY_VALUE_TILE + key]);
    $|      const float tile_maximum = lf_warp_max(local_maximum);
    $|      const float previous_maximum = running_maximum[query];
    $|      const float next_maximum = fmaxf(previous_maximum, tile_maximum);
    $|      const float previous_scale = isinf(previous_maximum)
    $|          ? 0.0f : expf(previous_maximum - next_maximum);
    $|      if (lane == 0) {
    $|        running_maximum[query] = next_maximum;
    $|        weights[query * LF_KEY_VALUE_TILE] = previous_scale;
    $|      }
    $|    }
    $|    __syncthreads();
    $|    for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;
    $|         linear += LF_BLOCK_THREADS) {
    $|      const int query = linear / LF_HEAD_DIMENSION;
    $|      accumulator[linear] *= weights[query * LF_KEY_VALUE_TILE];
    $|    }
    $|    __syncthreads();
    $|
    $|    for (int query = warp; query < active_queries; query += warp_count) {
    $|      const float previous_scale = weights[query * LF_KEY_VALUE_TILE];
    $|      const float maximum = running_maximum[query];
    $|      float local_sum = 0.0f;
    $|      for (int key = lane; key < LF_KEY_VALUE_TILE; key += 32) {
    $|        const float weight = expf(scores[query * LF_KEY_VALUE_TILE + key] - maximum);
    $|        weights[query * LF_KEY_VALUE_TILE + key] = weight;
    $|        local_sum += weight;
    $|      }
    $|      const float tile_sum = lf_warp_sum(local_sum);
    $|      if (lane == 0)
    $|        running_denominator[query] =
    $|            running_denominator[query] * previous_scale + tile_sum;
    $|    }
    $|    __syncthreads();
    $|
    $|    for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;
    $|         linear += LF_BLOCK_THREADS) {
    $|      const int query = linear / LF_HEAD_DIMENSION;
    $|      const int component = linear % LF_HEAD_DIMENSION;
    $|      float partial = 0.0f;
    $|      #pragma unroll 4
    $|      for (int key = 0; key < LF_KEY_VALUE_TILE; ++key)
    $|        partial += weights[query * LF_KEY_VALUE_TILE + key] *
    $|            __bfloat162float(value_tiles[
    $|                stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +
    $|                key * LF_HEAD_DIMENSION + component]);
    $|      accumulator[linear] += partial;
    $|    }
    $|    __syncthreads();
    $|  }
    $|
    $|  for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;
    $|       linear += LF_BLOCK_THREADS) {
    $|    const int query = linear / LF_HEAD_DIMENSION;
    $|    const int component = linear % LF_HEAD_DIMENSION;
    $|    const float denominator = running_denominator[query];
    $|    const int output_index = (token_base + query) * \{query_width} +
    $|        query_head * LF_HEAD_DIMENSION + component;
    $|    output[output_index] = __float2bfloat16_rn(
    $|        denominator > 0.0f ? accumulator[linear] / denominator : CUDART_NAN_F);
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
