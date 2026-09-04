///|
/// Emits the first production-shaped matrix-tiled prefill kernel from the
/// functional schedule. K/V tiles are loaded once and reused by a complete
/// query tile; online softmax and PV accumulation remain fused.
fn emit_cuda_prefill_attention_tile_source(
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
  let compute = lowering.compute()
  let key_value_partitions = lowering.key_value_partitions()
  if !(compute == CudaMatrixMultiplyAccumulate ||
    compute == CudaMatrixQkPvMultiplyAccumulate) ||
    lowering.copy() != CudaCooperativeSharedLoad {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let (query_tile, key_value_tile, head_dimension) = lowering.tile()
  let (_, _, block_threads, shared_memory_bytes) = lowering.launch()
  let (pipeline_stages, vector_bytes) = lowering.pipeline()
  let (query_heads, key_value_heads, abi_head_dimension, tokens_per_page) = abi.shape()
  let (input_row_width, page_stride_values, physical_page_capacity) = abi.layout()
  let (maximum_query_rows, maximum_query_tokens, maximum_page_entries) = abi.limits()
  if query_tile < 16 ||
    query_tile > 64 ||
    query_tile % 16 != 0 ||
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
  let vectors_per_key = head_dimension / values_per_vector
  if vectors_per_key <= 0 || vectors_per_key > 32 || 32 % vectors_per_key != 0 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let paged_key_value_load = if lowering.hoists_paged_key_value_page_lookup() {
    "      int physical_page = -1;\n" +
    "      if (key_position < partition_end && key_position < sequence_length &&\n" +
    "          key_position <= maximum_query_position && component == 0) {\n" +
    "        const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "        const int page_slot = page_start + logical_page;\n" +
    "        physical_page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;\n" +
    "        if (physical_page < 0 || (unsigned long long)physical_page >= " +
    physical_page_capacity.to_string() +
    "ULL) {\n" +
    "          atomicExch(&tile_valid, 0);\n" +
    "          physical_page = -1;\n" +
    "        }\n" +
    "      }\n" +
    "      const int page_source_lane =\n" +
    "          (lane / " +
    vectors_per_key.to_string() +
    ") * " +
    vectors_per_key.to_string() +
    ";\n" +
    "      physical_page = __shfl_sync(0xffffffffU, physical_page, page_source_lane);\n" +
    "      if (physical_page >= 0) {\n" +
    "        const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "        const unsigned long long cache_index =\n" +
    "            (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL +\n" +
    "            ((unsigned long long)token_in_page * " +
    key_value_heads.to_string() +
    "ULL + key_value_head) *\n" +
    "            LF_HEAD_DIMENSION + component;\n" +
    "        packed_key = *reinterpret_cast<const uint4 *>(key_cache + cache_index);\n" +
    "        packed_value = *reinterpret_cast<const uint4 *>(value_cache + cache_index);\n" +
    "      }\n"
  } else {
    "      if (key_position < partition_end && key_position < sequence_length &&\n" +
    "          key_position <= maximum_query_position) {\n" +
    "        const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "        const int page_slot = page_start + logical_page;\n" +
    "        const int physical_page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;\n" +
    "        if (physical_page < 0 || (unsigned long long)physical_page >= " +
    physical_page_capacity.to_string() +
    "ULL) {\n" +
    "          atomicExch(&tile_valid, 0);\n" +
    "        } else {\n" +
    "          const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "          const unsigned long long cache_index =\n" +
    "              (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL +\n" +
    "              ((unsigned long long)token_in_page * " +
    key_value_heads.to_string() +
    "ULL + key_value_head) *\n" +
    "              LF_HEAD_DIMENSION + component;\n" +
    "          packed_key = *reinterpret_cast<const uint4 *>(key_cache + cache_index);\n" +
    "          packed_value = *reinterpret_cast<const uint4 *>(value_cache + cache_index);\n" +
    "        }\n" +
    "      }\n"
  }
  let probability_type = if compute == CudaMatrixQkPvMultiplyAccumulate {
    "__nv_bfloat16"
  } else {
    "float"
  }
  let probability_store = if compute == CudaMatrixQkPvMultiplyAccumulate {
    "        weights[query * LF_KEY_VALUE_TILE + key] = __float2bfloat16_rn(weight);\n"
  } else {
    "        weights[query * LF_KEY_VALUE_TILE + key] = weight;\n"
  }
  let reuse_tile_storage = compute == CudaMatrixQkPvMultiplyAccumulate &&
    lowering.reuses_online_softmax_tile_storage()
  let shared_layout = if reuse_tile_storage {
    "  float *scores = reinterpret_cast<float *>(shared_cursor);\n" +
    "  shared_cursor += LF_QUERY_TILE * LF_KEY_VALUE_TILE * sizeof(float);\n" +
    "  float *accumulator = reinterpret_cast<float *>(shared_cursor);\n" +
    "  __nv_bfloat16 *weights = key_tiles;\n" +
    "  float *running_maximum = reinterpret_cast<float *>(key_tiles);\n" +
    "  float *running_denominator = running_maximum + LF_QUERY_TILE;\n" +
    "  float *previous_scales = reinterpret_cast<float *>(\n" +
    "      key_tiles + LF_QUERY_TILE * LF_KEY_VALUE_TILE);\n" +
    "  int &tile_valid = *reinterpret_cast<int *>(\n" +
    "      scores + LF_QUERY_TILE * LF_KEY_VALUE_TILE - 1);\n"
  } else {
    "  float *scores = reinterpret_cast<float *>(shared_cursor);\n" +
    "  shared_cursor += LF_QUERY_TILE * LF_KEY_VALUE_TILE * sizeof(float);\n" +
    "  " +
    probability_type +
    " *weights = reinterpret_cast<" +
    probability_type +
    " *>(shared_cursor);\n" +
    "  shared_cursor += LF_QUERY_TILE * LF_KEY_VALUE_TILE * sizeof(" +
    probability_type +
    ");\n" +
    "  float *accumulator = reinterpret_cast<float *>(shared_cursor);\n" +
    "  shared_cursor += LF_QUERY_TILE * LF_HEAD_DIMENSION * sizeof(float);\n" +
    "  float *running_maximum = reinterpret_cast<float *>(shared_cursor);\n" +
    "  shared_cursor += LF_QUERY_TILE * sizeof(float);\n" +
    "  float *running_denominator = reinterpret_cast<float *>(shared_cursor);\n" +
    "  shared_cursor += LF_QUERY_TILE * sizeof(float);\n" +
    "  float *previous_scales = reinterpret_cast<float *>(shared_cursor);\n" +
    "  shared_cursor += LF_QUERY_TILE * sizeof(float);\n" +
    "  int &tile_valid = *reinterpret_cast<int *>(shared_cursor);\n"
  }
  let fold_state_declaration = if reuse_tile_storage {
    "  float fold_maximum[LF_QUERY_STATE_SLOTS];\n" +
    "  float fold_denominator[LF_QUERY_STATE_SLOTS];\n" +
    "  #pragma unroll\n" +
    "  for (int slot = 0; slot < LF_QUERY_STATE_SLOTS; ++slot) {\n" +
    "    fold_maximum[slot] = -CUDART_INF_F;\n" +
    "    fold_denominator[slot] = 0.0f;\n" +
    "  }\n"
  } else {
    ""
  }
  let shared_state_initialization = if reuse_tile_storage {
    ""
  } else {
    "  if (thread < LF_QUERY_TILE) {\n" +
    "    running_maximum[thread] = -CUDART_INF_F;\n" +
    "    running_denominator[thread] = 0.0f;\n" +
    "  }\n"
  }
  let output_and_cache_parameters = if key_value_partitions == 1 {
    "    __nv_bfloat16 *output, const __nv_bfloat16 *key_cache,\n" +
    "    const __nv_bfloat16 *value_cache)"
  } else {
    "    const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache,\n" +
    "    float *workspace)"
  }
  let partition_guard = if key_value_partitions == 1 {
    ""
  } else {
    "  const int split = (int)blockIdx.z;\n" +
    "  if (split >= LF_KEY_VALUE_PARTITIONS) return;\n"
  }
  let partition_bounds = if key_value_partitions == 1 {
    "  const int partition_begin = 0;\n" +
    "  const int partition_end = maximum_query_position + 1;\n"
  } else {
    "  const int live_key_count = maximum_query_position + 1;\n" +
    "  const int base_keys_per_partition = live_key_count / LF_KEY_VALUE_PARTITIONS;\n" +
    "  const int larger_partition_count = live_key_count % LF_KEY_VALUE_PARTITIONS;\n" +
    "  const int partition_begin = split * base_keys_per_partition +\n" +
    "      (split < larger_partition_count ? split : larger_partition_count);\n" +
    "  const int partition_key_count = base_keys_per_partition +\n" +
    "      (split < larger_partition_count ? 1 : 0);\n" +
    "  const int partition_end = partition_begin + partition_key_count;\n"
  }
  let terminal_store = if key_value_partitions == 1 {
    "  for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;\n" +
    "       linear += LF_BLOCK_THREADS) {\n" +
    "    const int query = linear / LF_HEAD_DIMENSION;\n" +
    "    const int component = linear % LF_HEAD_DIMENSION;\n" +
    "    const float denominator = running_denominator[query];\n" +
    "    const int output_index = (token_base + query) * " +
    query_width.to_string() +
    " +\n" +
    "        query_head * LF_HEAD_DIMENSION + component;\n" +
    "    output[output_index] = __float2bfloat16_rn(\n" +
    "        denominator > 0.0f ? accumulator[linear] / denominator : CUDART_NAN_F);\n" +
    "  }\n"
  } else {
    "  for (int query = thread; query < active_queries; query += LF_BLOCK_THREADS) {\n" +
    "    const unsigned long long state =\n" +
    "        (((unsigned long long)(token_base + query) * " +
    query_heads.to_string() +
    "ULL + query_head) * LF_KEY_VALUE_PARTITIONS + split);\n" +
    "    workspace[state] = running_maximum[query];\n" +
    "    workspace[LF_PARTIAL_STATE_COUNT + state] = running_denominator[query];\n" +
    "  }\n" +
    "  for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;\n" +
    "       linear += LF_BLOCK_THREADS) {\n" +
    "    const int query = linear / LF_HEAD_DIMENSION;\n" +
    "    const int component = linear % LF_HEAD_DIMENSION;\n" +
    "    const unsigned long long state =\n" +
    "        (((unsigned long long)(token_base + query) * " +
    query_heads.to_string() +
    "ULL + query_head) * LF_KEY_VALUE_PARTITIONS + split);\n" +
    "    workspace[2ULL * LF_PARTIAL_STATE_COUNT +\n" +
    "        state * LF_HEAD_DIMENSION + component] = accumulator[linear];\n" +
    "  }\n"
  }
  let value_accumulation = if compute == CudaMatrixQkPvMultiplyAccumulate {
    "    if (warp < LF_HEAD_DIMENSION / 16) {\n" +
    "      using namespace nvcuda;\n" +
    "      const int output_column = warp * 16;\n" +
    "      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> probability_fragment;\n" +
    "      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> value_fragment;\n" +
    "      wmma::fragment<wmma::accumulator, 16, 16, 16, float> output_fragment;\n" +
    "      #pragma unroll\n" +
    "      for (int query_base = 0; query_base < LF_QUERY_TILE; query_base += 16) {\n" +
    "        wmma::load_matrix_sync(output_fragment,\n" +
    "            accumulator + query_base * LF_HEAD_DIMENSION + output_column,\n" +
    "            LF_HEAD_DIMENSION, wmma::mem_row_major);\n" +
    "        #pragma unroll\n" +
    "        for (int reduction = 0; reduction < LF_KEY_VALUE_TILE; reduction += 16) {\n" +
    "          wmma::load_matrix_sync(probability_fragment,\n" +
    "              weights + query_base * LF_KEY_VALUE_TILE + reduction,\n" +
    "              LF_KEY_VALUE_TILE);\n" +
    "          wmma::load_matrix_sync(value_fragment,\n" +
    "              value_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +\n" +
    "                  reduction * LF_HEAD_DIMENSION + output_column,\n" +
    "              LF_HEAD_DIMENSION);\n" +
    "          wmma::mma_sync(output_fragment, probability_fragment, value_fragment, output_fragment);\n" +
    "        }\n" +
    "        wmma::store_matrix_sync(\n" +
    "            accumulator + query_base * LF_HEAD_DIMENSION + output_column,\n" +
    "            output_fragment, LF_HEAD_DIMENSION, wmma::mem_row_major);\n" +
    "      }\n" +
    "    }\n"
  } else {
    "    for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;\n" +
    "         linear += LF_BLOCK_THREADS) {\n" +
    "      const int query = linear / LF_HEAD_DIMENSION;\n" +
    "      const int component = linear % LF_HEAD_DIMENSION;\n" +
    "      float partial = 0.0f;\n" +
    "      #pragma unroll 4\n" +
    "      for (int key = 0; key < LF_KEY_VALUE_TILE; ++key)\n" +
    "        partial += weights[query * LF_KEY_VALUE_TILE + key] *\n" +
    "            __bfloat162float(value_tiles[\n" +
    "                stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +\n" +
    "                key * LF_HEAD_DIMENSION + component]);\n" +
    "      accumulator[linear] += partial;\n" +
    "    }\n"
  }
  let online_softmax_update = if reuse_tile_storage {
    "    for (int query = warp; query < active_queries; query += warp_count) {\n" +
    "      const int state_slot = query / warp_count;\n" +
    "      float local_maximum = -CUDART_INF_F;\n" +
    "      for (int key = lane; key < LF_KEY_VALUE_TILE; key += 32)\n" +
    "        local_maximum = fmaxf(local_maximum, scores[query * LF_KEY_VALUE_TILE + key]);\n" +
    "      const float tile_maximum = lf_warp_max(local_maximum);\n" +
    "      const float previous_maximum = fold_maximum[state_slot];\n" +
    "      const float next_maximum = fmaxf(previous_maximum, tile_maximum);\n" +
    "      const float previous_scale = isinf(previous_maximum)\n" +
    "          ? 0.0f : expf(previous_maximum - next_maximum);\n" +
    "      float local_sum = 0.0f;\n" +
    "      for (int key = lane; key < LF_KEY_VALUE_TILE; key += 32) {\n" +
    "        const float weight = isinf(next_maximum) ? 0.0f :\n" +
    "            expf(scores[query * LF_KEY_VALUE_TILE + key] - next_maximum);\n" +
    probability_store +
    "        local_sum += weight;\n" +
    "      }\n" +
    "      const float tile_sum = lf_warp_sum(local_sum);\n" +
    "      fold_maximum[state_slot] = next_maximum;\n" +
    "      fold_denominator[state_slot] =\n" +
    "          fold_denominator[state_slot] * previous_scale + tile_sum;\n" +
    "      if (lane == 0) previous_scales[query] = previous_scale;\n" +
    "    }\n" +
    "    __syncthreads();\n" +
    "    for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;\n" +
    "         linear += LF_BLOCK_THREADS) {\n" +
    "      const int query = linear / LF_HEAD_DIMENSION;\n" +
    "      accumulator[linear] *= previous_scales[query];\n" +
    "    }\n" +
    "    __syncthreads();\n"
  } else {
    "    for (int query = warp; query < active_queries; query += warp_count) {\n" +
    "      float local_maximum = -CUDART_INF_F;\n" +
    "      for (int key = lane; key < LF_KEY_VALUE_TILE; key += 32)\n" +
    "        local_maximum = fmaxf(local_maximum, scores[query * LF_KEY_VALUE_TILE + key]);\n" +
    "      const float tile_maximum = lf_warp_max(local_maximum);\n" +
    "      const float previous_maximum = running_maximum[query];\n" +
    "      const float next_maximum = fmaxf(previous_maximum, tile_maximum);\n" +
    "      const float previous_scale = isinf(previous_maximum)\n" +
    "          ? 0.0f : expf(previous_maximum - next_maximum);\n" +
    "      if (lane == 0) {\n" +
    "        running_maximum[query] = next_maximum;\n" +
    "        previous_scales[query] = previous_scale;\n" +
    "      }\n" +
    "    }\n" +
    "    __syncthreads();\n" +
    "    for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;\n" +
    "         linear += LF_BLOCK_THREADS) {\n" +
    "      const int query = linear / LF_HEAD_DIMENSION;\n" +
    "      accumulator[linear] *= previous_scales[query];\n" +
    "    }\n" +
    "    __syncthreads();\n" +
    "    for (int query = warp; query < active_queries; query += warp_count) {\n" +
    "      const float previous_scale = previous_scales[query];\n" +
    "      const float maximum = running_maximum[query];\n" +
    "      float local_sum = 0.0f;\n" +
    "      for (int key = lane; key < LF_KEY_VALUE_TILE; key += 32) {\n" +
    "        const float weight = isinf(maximum) ? 0.0f :\n" +
    "            expf(scores[query * LF_KEY_VALUE_TILE + key] - maximum);\n" +
    probability_store +
    "        local_sum += weight;\n" +
    "      }\n" +
    "      const float tile_sum = lf_warp_sum(local_sum);\n" +
    "      if (lane == 0)\n" +
    "        running_denominator[query] =\n" +
    "            running_denominator[query] * previous_scale + tile_sum;\n" +
    "    }\n" +
    "    __syncthreads();\n"
  }
  let terminal_fold_state = if reuse_tile_storage {
    "  for (int query = warp; query < active_queries; query += warp_count) {\n" +
    "    const int state_slot = query / warp_count;\n" +
    "    if (lane == 0) {\n" +
    "      running_maximum[query] = fold_maximum[state_slot];\n" +
    "      running_denominator[query] = fold_denominator[state_slot];\n" +
    "    }\n" +
    "  }\n" +
    "  __syncthreads();\n"
  } else {
    ""
  }
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
    $|#define LF_KEY_VALUE_PARTITIONS \{key_value_partitions}
    $|#define LF_QUERY_STATE_SLOTS ((LF_QUERY_TILE + (LF_BLOCK_THREADS / 32) - 1) / (LF_BLOCK_THREADS / 32))
    $|#define LF_PARTIAL_STATE_COUNT \{maximum_query_tokens.to_int64() * query_heads.to_int64() * key_value_partitions.to_int64()}ULL
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
    $|    const __nv_bfloat16 *rotated_qkv_rows,
    $|\{output_and_cache_parameters} {
    $|  const int thread = (int)threadIdx.x;
    $|  const int lane = thread & 31;
    $|  const int warp = thread >> 5;
    $|  const int warp_count = LF_BLOCK_THREADS >> 5;
    $|  const int query_head = (int)blockIdx.y;
    $|  if (blockDim.x != LF_BLOCK_THREADS || query_head >= \{query_heads}) return;
    $|\{partition_guard}
    $|
    $|  const int prefill_count = counts[0];
    $|  const int decode_count = counts[1];
    $|  const int row_count = counts[2];
    $|  const int token_count = counts[3];
    $|  const int page_entry_count = counts[4];
    $|  if (prefill_count < 0 || decode_count < 0 ||
    $|      prefill_count + decode_count <= 0 ||
    $|      row_count != prefill_count + decode_count ||
    $|      row_count > \{maximum_query_rows} || token_count <= 0 ||
    $|      token_count > \{maximum_query_tokens} || page_entry_count <= 0 ||
    $|      page_entry_count > \{maximum_page_entries}) return;
    $|
    $|  for (int tile_ordinal = (int)blockIdx.x; ;
    $|       tile_ordinal += (int)gridDim.x) {
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
    $|\{shared_layout}
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
    $|\{shared_state_initialization}
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
    $|\{partition_bounds}
    $|
    $|  const int key_value_vector_count = LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION / LF_VALUES_PER_VECTOR;
    $|  const int key_value_head = query_head / \{group_size};
    $|  const float scale = rsqrtf((float)LF_HEAD_DIMENSION);
    $|\{fold_state_declaration}
    $|  for (int key_base = partition_begin, tile_index = 0; key_base < partition_end;
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
    $|\{paged_key_value_load}
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
    $|      const int key_column = warp * 16;
    $|      #pragma unroll
    $|      for (int query_base = 0; query_base < LF_QUERY_TILE; query_base += 16) {
    $|        wmma::fill_fragment(score_fragment, 0.0f);
    $|        #pragma unroll
    $|        for (int reduction = 0; reduction < LF_HEAD_DIMENSION; reduction += 16) {
    $|          wmma::load_matrix_sync(query_fragment,
    $|              query_tile + query_base * LF_HEAD_DIMENSION + reduction,
    $|              LF_HEAD_DIMENSION);
    $|          wmma::load_matrix_sync(key_fragment,
    $|              key_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +
    $|                  key_column * LF_HEAD_DIMENSION + reduction,
    $|              LF_HEAD_DIMENSION);
    $|          wmma::mma_sync(score_fragment, query_fragment, key_fragment, score_fragment);
    $|        }
    $|        wmma::store_matrix_sync(
    $|            scores + query_base * LF_KEY_VALUE_TILE + key_column,
    $|            score_fragment, LF_KEY_VALUE_TILE, wmma::mem_row_major);
    $|      }
    $|    }
    $|    __syncthreads();
    $|
    $|    for (int linear = thread; linear < LF_QUERY_TILE * LF_KEY_VALUE_TILE;
    $|         linear += LF_BLOCK_THREADS) {
    $|      const int query = linear / LF_KEY_VALUE_TILE;
    $|      const int key = linear % LF_KEY_VALUE_TILE;
    $|      const int key_position = key_base + key;
    $|      const int query_position = query < active_queries ? query_positions[token_base + query] : -1;
    $|      scores[linear] = query < active_queries &&
    $|          key_position < partition_end && key_position <= query_position
    $|          ? scores[linear] * scale : -CUDART_INF_F;
    $|    }
    $|    __syncthreads();
    $|
    $|\{online_softmax_update}
    $|
    $|\{value_accumulation}
    $|    __syncthreads();
    $|  }
    $|
    $|\{terminal_fold_state}
    $|\{terminal_store}
    $|  __syncthreads();
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

///|
/// Emits the compiler-selected grouped-query decode schedule. Each subgroup
/// owns one query head while all subgroups reuse the K/V tile for their shared
/// key/value head. The online-softmax fold remains subgroup-local.
fn emit_cuda_decode_attention_tile_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  symbol : String,
  maximum_source_bytes : Int,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  let compute = lowering.compute()
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
  let accumulator_items = (head_dimension + 31) / 32
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
    $|    if (thread == 0) *tile_valid = 1;
    $|    __syncthreads();
    $|    for (int linear = thread;
    $|         linear < LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|         linear += LF_BLOCK_THREADS) {
    $|      const int key = linear / LF_HEAD_DIMENSION;
    $|      const int component = linear % LF_HEAD_DIMENSION;
    $|      __nv_bfloat16 staged_key = __float2bfloat16_rn(0.0f);
    $|      __nv_bfloat16 staged_value = __float2bfloat16_rn(0.0f);
    $|      if (key < active_keys) {
    $|        const int key_position = key_base + key;
    $|        const int page_slot = page_start + key_position / \{tokens_per_page};
    $|        const int physical_page =
    $|            page_slot < page_end ? page_table_page_indices[page_slot] : -1;
    $|        if (physical_page < 0 ||
    $|            (unsigned long long)physical_page >= \{physical_page_capacity}ULL) {
    $|          atomicExch(tile_valid, 0);
    $|        } else {
    $|          const int token_in_page = key_position % \{tokens_per_page};
    $|          const unsigned long long cache_index =
    $|              (unsigned long long)physical_page * \{page_stride_values}ULL +
    $|              ((unsigned long long)token_in_page * \{key_value_heads}ULL +
    $|                  key_value_head) * LF_HEAD_DIMENSION + component;
    $|          staged_key = key_cache[cache_index];
    $|          staged_value = value_cache[cache_index];
    $|        }
    $|      }
    $|      const unsigned long long stage_offset =
    $|          (unsigned long long)stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +
    $|          linear;
    $|      key_tiles[stage_offset] = staged_key;
    $|      value_tiles[stage_offset] = staged_value;
    $|    }
    $|    __syncthreads();
    $|    if (!*tile_valid) return;
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
    $|      float old_scale = 0.0f;
    $|      float new_scale = 0.0f;
    $|      if (lane == 0) {
    $|        const float next_maximum = fmaxf(maximum, score);
    $|        old_scale = denominator == 0.0f
    $|            ? 0.0f : expf(maximum - next_maximum);
    $|        new_scale = expf(score - next_maximum);
    $|        maximum = next_maximum;
    $|        denominator = denominator * old_scale + new_scale;
    $|      }
    $|      old_scale = __shfl_sync(0xffffffffU, old_scale, 0);
    $|      new_scale = __shfl_sync(0xffffffffU, new_scale, 0);
    $|      #pragma unroll
    $|      for (int item = 0; item < \{accumulator_items}; ++item) {
    $|        const int component = lane + item * 32;
    $|        if (component < LF_HEAD_DIMENSION)
    $|          accumulator[item] = accumulator[item] * old_scale + new_scale *
    $|              __bfloat162float(stage_values[
    $|                  key * LF_HEAD_DIMENSION + component]);
    $|      }
    $|    }
    $|    __syncthreads();
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

///|
/// Emits CUDA only after the generic strategy, LunaTile IR, functional
/// rewrites, and immutable schedule have selected a phase-specific lowering.
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
  if lowering.key_value_partitions() != 1 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  match lowering.phase() {
    Prefill =>
      if lowering.copy() == CudaDirectGlobalLoad {
        emit_cuda_direct_attention_tile_source(
          lowering, abi, symbol, maximum_source_bytes,
        )
      } else {
        emit_cuda_prefill_attention_tile_source(
          lowering, abi, symbol, maximum_source_bytes,
        )
      }
    Decode =>
      if lowering.copy() == CudaDirectGlobalLoad {
        emit_cuda_direct_attention_tile_source(
          lowering, abi, symbol, maximum_source_bytes,
        )
      } else if lowering.compute() == CudaGroupedQuerySplitSubgroupShuffleDot {
        emit_cuda_grouped_split_decode_attention_tile_source(
          lowering, abi, symbol, maximum_source_bytes,
        )
      } else {
        emit_cuda_decode_attention_tile_source(
          lowering, abi, symbol, maximum_source_bytes,
        )
      }
    MixedCompatibility =>
      raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
}
