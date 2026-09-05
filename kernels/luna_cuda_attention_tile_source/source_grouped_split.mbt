///|
/// Emits a grouped-query decode tile with several subgroups cooperating on
/// each query head. The block loads one K/V tile for the whole GQA group,
/// folds disjoint key positions in registers, then merges the partial online
/// softmax states once at the end.
fn emit_cuda_grouped_split_decode_attention_tile_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  symbol : String,
  maximum_source_bytes : Int,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  let (query_tile, key_value_tile, head_dimension) = lowering.tile()
  let (_, grid_y, block_threads, shared_memory_bytes) = lowering.launch()
  let (pipeline_stages, vector_bytes) = lowering.pipeline()
  let (query_heads, key_value_heads, abi_head_dimension, tokens_per_page) = abi.shape()
  let (input_row_width, page_stride_values, physical_page_capacity) = abi.layout()
  let (maximum_query_rows, maximum_query_tokens, maximum_page_entries) = abi.limits()
  let group_size = query_heads / key_value_heads
  let subgroup_count = block_threads / 32
  let splits_per_head = subgroup_count / group_size
  if lowering.phase() != Decode ||
    lowering.compute() != CudaGroupedQuerySplitSubgroupShuffleDot ||
    lowering.copy() != CudaCooperativeSharedLoad ||
    query_tile != 1 ||
    key_value_tile != 64 ||
    head_dimension != abi_head_dimension ||
    query_heads % key_value_heads != 0 ||
    group_size <= 1 ||
    block_threads <= group_size * 32 ||
    block_threads > 256 ||
    block_threads % (group_size * 32) != 0 ||
    grid_y != key_value_heads ||
    shared_memory_bytes <= 0L ||
    pipeline_stages != 1 ||
    vector_bytes != 8 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let query_width = query_heads * head_dimension
  let accumulator_items = (head_dimension + 31) / 32
  let partitions = lowering.key_value_partitions()
  let partitioned = partitions > 1
  let state_count = maximum_query_rows.to_int64() *
    query_heads.to_int64() *
    partitions.to_int64()
  let output_arguments = if partitioned {
    "const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache, float *workspace"
  } else {
    "__nv_bfloat16 *output, const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache"
  }
  let initialize_state = if partitioned {
    (
      $|  const int partition = (int)blockIdx.z;
      $|  if (partition >= \{partitions}) return;
      $|  const unsigned long long state =
      $|      ((unsigned long long)row * \{query_heads} + query_head) * \{partitions} + partition;
      $|  if (lane == 0 && split == 0) {
      $|    workspace[state] = CUDART_NAN_F;
      $|    workspace[\{state_count}ULL + state] = CUDART_NAN_F;
      $|  }
    )
  } else {
    ""
  }
  let invalid_output = if partitioned {
    ""
  } else {
    (
      $|    if (token >= 0 && token < token_count && lane == 0 && split == 0)
      $|      output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);
    )
  }
  let partition_bounds = if partitioned {
    (
      $|  const int tile_count = (query_position + LF_KEY_VALUE_TILE) / LF_KEY_VALUE_TILE;
      $|  const int first_tile = (int)((long long)tile_count * partition / \{partitions});
      $|  const int last_tile = (int)((long long)tile_count * (partition + 1) / \{partitions});
      $|  const int key_begin = first_tile * LF_KEY_VALUE_TILE;
      $|  const int key_limit = min(query_position + 1, last_tile * LF_KEY_VALUE_TILE);
    )
  } else {
    (
      $|  const int key_begin = 0;
      $|  const int key_limit = query_position + 1;
    )
  }
  let write_result = if partitioned {
    (
      $|        workspace[2ULL * \{state_count}ULL + state * LF_HEAD_DIMENSION + component] = merged;
    )
  } else {
    (
      $|        output[output_offset + component] = __float2bfloat16_rn(
      $|            merged_denominator > 0.0f ? merged / merged_denominator : CUDART_NAN_F);
    )
  }
  let write_state = if partitioned {
    (
      $|    if (lane == 0) {
      $|      workspace[state] = merged_maximum;
      $|      workspace[\{state_count}ULL + state] = merged_denominator;
      $|    }
    )
  } else {
    ""
  }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|#define LF_BLOCK_THREADS \{block_threads}
    $|#define LF_SUBGROUP_COUNT \{subgroup_count}
    $|#define LF_QUERY_HEADS_PER_BLOCK \{group_size}
    $|#define LF_SPLITS_PER_HEAD \{splits_per_head}
    $|#define LF_KEY_VALUE_TILE 64
    $|#define LF_HEAD_DIMENSION \{head_dimension}
    $|
    $|static __device__ __forceinline__ float lf_split_warp_sum(float value) {
    $|  for (int offset = 16; offset > 0; offset >>= 1)
    $|    value += __shfl_down_sync(0xffffffffU, value, offset);
    $|  return __shfl_sync(0xffffffffU, value, 0);
    $|}
    $|
    $|extern "C" __global__ void \{symbol}(
    $|    const int *counts, const int *query_positions,
    $|    const int *query_row_offsets, const int *sequence_lengths,
    $|    const int *page_table_offsets, const int *page_table_page_indices,
    $|    const __nv_bfloat16 *rotated_qkv_rows, \{output_arguments}) {
    $|  const int thread = (int)threadIdx.x;
    $|  const int lane = thread & 31;
    $|  const int warp = thread >> 5;
    $|  const int query_in_group = warp / LF_SPLITS_PER_HEAD;
    $|  const int split = warp % LF_SPLITS_PER_HEAD;
    $|  const int local_row = (int)blockIdx.x;
    $|  const int key_value_head = (int)blockIdx.y;
    $|  const int query_head =
    $|      key_value_head * LF_QUERY_HEADS_PER_BLOCK + query_in_group;
    $|  if (blockDim.x != LF_BLOCK_THREADS || warp >= LF_SUBGROUP_COUNT ||
    $|      key_value_head >= \{key_value_heads} ||
    $|      query_in_group >= LF_QUERY_HEADS_PER_BLOCK ||
    $|      query_head >= \{query_heads}) return;
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
    $|\{initialize_state}
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
    $|\{invalid_output}
    $|    return;
    $|  }
    $|
    $|  extern __shared__ __align__(16) unsigned char shared_arena[];
    $|  __nv_bfloat16 *key_tile =
    $|      reinterpret_cast<__nv_bfloat16 *>(shared_arena);
    $|  __nv_bfloat16 *value_tile =
    $|      key_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|  int *tile_valid = reinterpret_cast<int *>(
    $|      value_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION);
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
    $|\{partition_bounds}
    $|  for (int key_base = key_begin; key_base < key_limit;
    $|       key_base += LF_KEY_VALUE_TILE) {
    $|    const int active_keys =
    $|        min(LF_KEY_VALUE_TILE, key_limit - key_base);
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
    $|      key_tile[linear] = staged_key;
    $|      value_tile[linear] = staged_value;
    $|    }
    $|    __syncthreads();
    $|    if (!*tile_valid) return;
    $|    for (int key = split; key < active_keys; key += LF_SPLITS_PER_HEAD) {
    $|      float partial = 0.0f;
    $|      for (int component = lane; component < LF_HEAD_DIMENSION;
    $|           component += 32)
    $|        partial +=
    $|            __bfloat162float(rotated_qkv_rows[query_offset + component]) *
    $|            __bfloat162float(key_tile[key * LF_HEAD_DIMENSION + component]);
    $|      const float score = lf_split_warp_sum(partial) * scale;
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
    $|              __bfloat162float(
    $|                  value_tile[key * LF_HEAD_DIMENSION + component]);
    $|      }
    $|    }
    $|    __syncthreads();
    $|  }
    $|
    $|  float *partial_numerators = reinterpret_cast<float *>(shared_arena);
    $|  float *partial_maxima = partial_numerators +
    $|      LF_SUBGROUP_COUNT * LF_HEAD_DIMENSION;
    $|  float *partial_denominators = partial_maxima + LF_SUBGROUP_COUNT;
    $|  if (lane == 0) {
    $|    partial_maxima[warp] = maximum;
    $|    partial_denominators[warp] = denominator;
    $|  }
    $|  #pragma unroll
    $|  for (int item = 0; item < \{accumulator_items}; ++item) {
    $|    const int component = lane + item * 32;
    $|    if (component < LF_HEAD_DIMENSION)
    $|      partial_numerators[warp * LF_HEAD_DIMENSION + component] =
    $|          accumulator[item];
    $|  }
    $|  __syncthreads();
    $|  if (split == 0) {
    $|    const int first_warp = query_in_group * LF_SPLITS_PER_HEAD;
    $|    float merged_maximum = -CUDART_INF_F;
    $|    float merged_denominator = 0.0f;
    $|    if (lane == 0) {
    $|      #pragma unroll
    $|      for (int part = 0; part < LF_SPLITS_PER_HEAD; ++part)
    $|        merged_maximum = fmaxf(
    $|            merged_maximum, partial_maxima[first_warp + part]);
    $|      #pragma unroll
    $|      for (int part = 0; part < LF_SPLITS_PER_HEAD; ++part)
    $|        if (partial_denominators[first_warp + part] > 0.0f)
    $|          merged_denominator += partial_denominators[first_warp + part] *
    $|            expf(partial_maxima[first_warp + part] - merged_maximum);
    $|    }
    $|    merged_maximum =
    $|        __shfl_sync(0xffffffffU, merged_maximum, 0);
    $|    merged_denominator =
    $|        __shfl_sync(0xffffffffU, merged_denominator, 0);
    $|\{write_state}
    $|    #pragma unroll
    $|    for (int item = 0; item < \{accumulator_items}; ++item) {
    $|      const int component = lane + item * 32;
    $|      if (component < LF_HEAD_DIMENSION) {
    $|        float merged = 0.0f;
    $|        #pragma unroll
    $|        for (int part = 0; part < LF_SPLITS_PER_HEAD; ++part)
    $|          if (partial_denominators[first_warp + part] > 0.0f)
    $|            merged += partial_numerators[
    $|              (first_warp + part) * LF_HEAD_DIMENSION + component] *
    $|              expf(partial_maxima[first_warp + part] - merged_maximum);
    $|\{write_result}
    $|      }
    $|    }
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
