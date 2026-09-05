///|
/// Emits the compiler-selected direct-global attention schedule. One subgroup
/// owns one query head and folds the paged K/V stream in registers. This is
/// the low-launch-overhead schedule for decode and short/ragged prefill; it is
/// selected by the backend-neutral compiler rather than by model code.
fn emit_cuda_direct_attention_tile_source(
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
  if lowering.compute() != CudaScalarFma ||
    lowering.copy() != CudaDirectGlobalLoad ||
    query_tile != 1 ||
    key_value_tile != 16 ||
    head_dimension != abi_head_dimension ||
    block_threads != 32 ||
    grid_y != query_heads ||
    shared_memory_bytes != 0L ||
    pipeline_stages != 1 ||
    vector_bytes != 4 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let query_width = query_heads * head_dimension
  let group_size = query_heads / key_value_heads
  let accumulator_items = (head_dimension + 31) / 32
  let partitions = lowering.key_value_partitions()
  let partitioned = partitions > 1
  let output_offset_binding = if partitioned {
    ""
  } else {
    (
      $|    const int output_offset = token * \{query_width} +
      $|        query_head * LF_HEAD_DIMENSION;
    )
  }
  if partitioned && lowering.phase() != Decode {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
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
      $|    const int partition = (int)blockIdx.z;
      $|    if (partition >= \{partitions}) return;
      $|    const unsigned long long state =
      $|        ((unsigned long long)row * \{query_heads} + query_head) * \{partitions} + partition;
      $|    if (lane == 0) {
      $|      workspace[state] = CUDART_NAN_F;
      $|      workspace[\{state_count}ULL + state] = CUDART_NAN_F;
      $|    }
    )
  } else {
    ""
  }
  let invalid_output = if partitioned {
    ""
  } else {
    (
      $|      if (token >= 0 && token < token_count && lane == 0)
      $|        output[output_offset] = __float2bfloat16_rn(CUDART_NAN_F);
    )
  }
  let partition_bounds = if partitioned {
    (
      $|    const int tile_count = (query_position + \{key_value_tile}) / \{key_value_tile};
      $|    const int first_tile = (int)((long long)tile_count * partition / \{partitions});
      $|    const int last_tile = (int)((long long)tile_count * (partition + 1) / \{partitions});
      $|    const int key_begin = first_tile * \{key_value_tile};
      $|    const int key_limit = min(query_position + 1, last_tile * \{key_value_tile});
    )
  } else {
    (
      $|    const int key_begin = 0;
      $|    const int key_limit = query_position + 1;
    )
  }
  let write_result = if partitioned {
    (
      $|        workspace[2ULL * \{state_count}ULL + state * LF_HEAD_DIMENSION + component] = accumulator[item];
    )
  } else {
    (
      $|        output[output_offset + component] = __float2bfloat16_rn(
      $|            denominator > 0.0f ? accumulator[item] / denominator : CUDART_NAN_F);
    )
  }
  let write_state = if partitioned {
    (
      $|    if (lane == 0) {
      $|      workspace[state] = maximum;
      $|      workspace[\{state_count}ULL + state] = denominator;
      $|    }
    )
  } else {
    ""
  }
  let row_open = match lowering.phase() {
    Prefill =>
      (
        $|  for (int token = (int)blockIdx.x; token < token_count;
        $|       token += (int)gridDim.x) {
        $|    int row = 0;
        $|    while (row + 1 < row_count && token >= query_row_offsets[row + 1])
        $|      ++row;
        $|    if (row >= prefill_count) continue;
        $|    const int row_start = query_row_offsets[row];
        $|    const int row_end = query_row_offsets[row + 1];
      )
    Decode =>
      (
        $|  const int local_row = (int)blockIdx.x;
        $|  if (local_row >= decode_count) return;
        $|  const int row = prefill_count + local_row;
        $|  const int token = query_row_offsets[row];
        $|  const int row_start = token;
        $|  const int row_end = query_row_offsets[row + 1];
      )
    MixedCompatibility =>
      raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let row_close = match lowering.phase() {
    Prefill => "  }\n"
    Decode => ""
    MixedCompatibility =>
      raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let phase_counts = match lowering.phase() {
    Prefill => "prefill_count <= 0 || decode_count < 0"
    Decode => "prefill_count < 0 || decode_count <= 0"
    MixedCompatibility =>
      raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let invalid_action = match lowering.phase() {
    Prefill => "continue;"
    Decode => "return;"
    MixedCompatibility =>
      raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|#define LF_HEAD_DIMENSION \{head_dimension}
    $|
    $|static __device__ __forceinline__ float lf_direct_warp_sum(float value) {
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
    $|  const int lane = (int)threadIdx.x;
    $|  const int query_head = (int)blockIdx.y;
    $|  if (blockDim.x != 32 || query_head >= \{query_heads}) return;
    $|  const int prefill_count = counts[0];
    $|  const int decode_count = counts[1];
    $|  const int row_count = counts[2];
    $|  const int token_count = counts[3];
    $|  const int page_entry_count = counts[4];
    $|  if (\{phase_counts} || row_count != prefill_count + decode_count ||
    $|      row_count > \{maximum_query_rows} || token_count <= 0 ||
    $|      token_count > \{maximum_query_tokens} || page_entry_count <= 0 ||
    $|      page_entry_count > \{maximum_page_entries}) return;
    $|
    $|\{row_open}
    $|\{initialize_state}
    $|    const int sequence_length = sequence_lengths[row];
    $|    const int query_position = token >= 0 && token < token_count
    $|        ? query_positions[token] : -1;
    $|    const int page_start = page_table_offsets[row];
    $|    const int page_end = page_table_offsets[row + 1];
    $|\{output_offset_binding}
    $|    if (row_start < 0 || row_start >= row_end || row_end > token_count ||
    $|        query_position < 0 || query_position >= sequence_length ||
    $|        page_start < 0 || page_start >= page_end ||
    $|        page_end > page_entry_count ||
    $|        page_end - page_start !=
    $|            (sequence_length + \{tokens_per_page - 1}) / \{tokens_per_page}) {
    $|\{invalid_output}
    $|      \{invalid_action}
    $|    }
    $|    const int key_value_head = query_head / \{group_size};
    $|    const unsigned long long query_offset =
    $|        (unsigned long long)token * \{input_row_width}ULL +
    $|        (unsigned long long)query_head * LF_HEAD_DIMENSION;
    $|    const float scale = rsqrtf((float)LF_HEAD_DIMENSION);
    $|    float maximum = -CUDART_INF_F;
    $|    float denominator = 0.0f;
    $|    float accumulator[\{accumulator_items}];
    $|    #pragma unroll
    $|    for (int item = 0; item < \{accumulator_items}; ++item)
    $|      accumulator[item] = 0.0f;
    $|
    $|\{partition_bounds}
    $|    for (int key_position = key_begin; key_position < key_limit;
    $|         ++key_position) {
    $|      const int page_slot = page_start + key_position / \{tokens_per_page};
    $|      const int physical_page = page_slot < page_end
    $|          ? page_table_page_indices[page_slot] : -1;
    $|      const int token_in_page = key_position % \{tokens_per_page};
    $|      const unsigned long long cache_base =
    $|          physical_page >= 0 &&
    $|                  (unsigned long long)physical_page < \{physical_page_capacity}ULL
    $|              ? (unsigned long long)physical_page * \{page_stride_values}ULL +
    $|                  ((unsigned long long)token_in_page * \{key_value_heads}ULL +
    $|                      key_value_head) * LF_HEAD_DIMENSION
    $|              : 0ULL;
    $|      float partial = 0.0f;
    $|      for (int component = lane; component < LF_HEAD_DIMENSION;
    $|           component += 32) {
    $|        const float key = physical_page >= 0 &&
    $|                (unsigned long long)physical_page < \{physical_page_capacity}ULL
    $|            ? __bfloat162float(key_cache[cache_base + component])
    $|            : CUDART_NAN_F;
    $|        partial +=
    $|            __bfloat162float(rotated_qkv_rows[query_offset + component]) * key;
    $|      }
    $|      const float score = lf_direct_warp_sum(partial) * scale;
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
    $|        if (component < LF_HEAD_DIMENSION) {
    $|          const float value = physical_page >= 0 &&
    $|                  (unsigned long long)physical_page < \{physical_page_capacity}ULL
    $|              ? __bfloat162float(value_cache[cache_base + component])
    $|              : CUDART_NAN_F;
    $|          accumulator[item] =
    $|              accumulator[item] * old_scale + new_scale * value;
    $|        }
    $|      }
    $|    }
    $|\{write_state}
    $|    denominator = __shfl_sync(0xffffffffU, denominator, 0);
    $|    #pragma unroll
    $|    for (int item = 0; item < \{accumulator_items}; ++item) {
    $|      const int component = lane + item * 32;
    $|      if (component < LF_HEAD_DIMENSION)
    $|\{write_result}
    $|    }
    $|\{row_close}
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
