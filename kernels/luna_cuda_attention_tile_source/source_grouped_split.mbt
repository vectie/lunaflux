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
  let (input_row_width, _, _) = abi.layout()
  let (maximum_query_rows, maximum_query_tokens, maximum_page_entries) = abi.limits()
  let group_size = query_heads / key_value_heads
  let subgroup_count = block_threads / 32
  let splits_per_head = subgroup_count / group_size
  let asynchronous = lowering.transfer_readiness() ==
    KeyScoreBeforeValueReadiness &&
    lowering.copy() == CudaAsynchronousSharedLoad &&
    pipeline_stages == 2
  if lowering.phase() != Decode ||
    lowering.compute() != CudaGroupedQuerySplitSubgroupShuffleDot ||
    (
      !(lowering.copy() == CudaCooperativeSharedLoad && pipeline_stages == 1) &&
      !asynchronous
    ) ||
    query_tile != 1 ||
    (key_value_tile != 64 && !(asynchronous && key_value_tile == 32)) ||
    head_dimension != abi_head_dimension ||
    query_heads % key_value_heads != 0 ||
    group_size <= 1 ||
    block_threads <= group_size * 32 ||
    block_threads > 256 ||
    block_threads % (group_size * 32) != 0 ||
    grid_y != key_value_heads ||
    shared_memory_bytes <= 0L ||
    vector_bytes != 8 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let query_width = query_heads * head_dimension
  let transfer = emit_decode_key_value_transfer(abi, vector_bytes)
  let (_, page_stride, _) = abi.layout()
  if asynchronous && (head_dimension % 4 != 0 || page_stride % 4L != 0L) {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let storage = if asynchronous {
    (
      $|  __nv_bfloat16 *stage_arena = key_tile;
      $|  int *stage_valid = reinterpret_cast<int *>(stage_arena +
      $|      4 * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION);
    )
  } else {
    (
      $|  int *tile_valid = reinterpret_cast<int *>(
      $|      value_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION);
    )
  }
  let prologue = if asynchronous { emit_decode_async_stage(abi) } else { "" }
  let iteration = if asynchronous {
    emit_decode_pipeline_iteration()
  } else {
    (
      $|    if (thread == 0) *tile_valid = 1;
      $|    __syncthreads();
      $|\{transfer}
      $|    __syncthreads();
      $|    if (!*tile_valid) return;
    )
  }
  let accumulator_items = (head_dimension + 31) / 32
  let partitions = lowering.key_value_partitions()
  let partition_grain = lowering.key_value_partition_grain()
  let partitioned = partitions > 1
  let output_offset_binding = if partitioned {
    ""
  } else {
    (
      $|  const int output_offset = token * \{query_width} +
      $|      query_head * LF_HEAD_DIMENSION;
    )
  }
  let warp_sum_symbol = symbol + "_warp_sum"
  let dot =
    $|      float partial = 0.0f;
    $|      #pragma unroll
    $|      for (int item = 0; item < \{accumulator_items}; ++item) {
    $|        const int component = lane + item * 32;
    $|        if (component < LF_HEAD_DIMENSION)
    $|          partial += query_fragment[item] *
    $|              __bfloat162float(key_tile[key * LF_HEAD_DIMENSION + component]);
    $|      }
    $|      const float score = \{warp_sum_symbol}(partial) * scale;
  let score_preparation = if asynchronous {
    (
      $|    float scores[(LF_KEY_VALUE_TILE + LF_SPLITS_PER_HEAD - 1) / LF_SPLITS_PER_HEAD];
      $|    for (int key = split; key < active_keys; key += LF_SPLITS_PER_HEAD) {
      $|\{dot}
      $|      scores[key / LF_SPLITS_PER_HEAD] = score;
      $|    }
      $|    asm volatile("cp.async.wait_group 0;" ::: "memory");
      $|    __syncthreads();
      $|    if (!stage_valid[slot]) return;
      $|    // The alternate slot is dead; overlap its next K/V with this V fold.
      $|    if (key_base + LF_KEY_VALUE_TILE < key_limit)
      $|      stage_tile(key_base + LF_KEY_VALUE_TILE, slot ^ 1);
    )
  } else {
    ""
  }
  let score_consumption = if asynchronous {
    "      const float score = scores[key / LF_SPLITS_PER_HEAD];"
  } else {
    dot
  }
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
      $|  const int tile_count = (query_position + \{partition_grain}) / \{partition_grain};
      $|  const int first_tile = (int)((long long)tile_count * partition / \{partitions});
      $|  const int last_tile = (int)((long long)tile_count * (partition + 1) / \{partitions});
      $|  const int key_begin = first_tile * \{partition_grain};
      $|  const int key_limit = min(query_position + 1, last_tile * \{partition_grain});
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
    $|#define LF_KEY_VALUE_TILE \{key_value_tile}
    $|#define LF_HEAD_DIMENSION \{head_dimension}
    $|
    $|static __device__ __forceinline__ float \{warp_sum_symbol}(float value) {
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
    $|\{output_offset_binding}
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
    $|\{storage}
    $|  const unsigned long long query_offset =
    $|      (unsigned long long)token * \{input_row_width}ULL +
    $|      (unsigned long long)query_head * LF_HEAD_DIMENSION;
    $|  const float scale = rsqrtf((float)LF_HEAD_DIMENSION);
    $|  float maximum = -CUDART_INF_F;
    $|  float denominator = 0.0f;
    $|  float accumulator[\{accumulator_items}];
    $|  // The query belongs to the outer row/head map, not the KV fold.
    $|  float query_fragment[\{accumulator_items}];
    $|  #pragma unroll
    $|  for (int item = 0; item < \{accumulator_items}; ++item) {
    $|    accumulator[item] = 0.0f;
    $|    const int component = lane + item * 32;
    $|    query_fragment[item] = component < LF_HEAD_DIMENSION
    $|        ? __bfloat162float(rotated_qkv_rows[query_offset + component]) : 0.0f;
    $|  }
    $|
    $|\{partition_bounds}
    $|\{prologue}
    $|  for (int key_base = key_begin; key_base < key_limit;
    $|       key_base += LF_KEY_VALUE_TILE) {
    $|    const int active_keys =
    $|        min(LF_KEY_VALUE_TILE, key_limit - key_base);
    $|\{iteration}
    $|\{score_preparation}
    $|    for (int key = split; key < active_keys; key += LF_SPLITS_PER_HEAD) {
    $|\{score_consumption}
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
    $|#undef LF_BLOCK_THREADS
    $|#undef LF_SUBGROUP_COUNT
    $|#undef LF_QUERY_HEADS_PER_BLOCK
    $|#undef LF_SPLITS_PER_HEAD
    $|#undef LF_KEY_VALUE_TILE
    $|#undef LF_HEAD_DIMENSION
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
