///|
fn partitioned_checked_product(
  left : Int64,
  right : Int64,
) -> Int64 raise CudaAttentionTileSourceError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise InvalidCudaAttentionTileSource(WorkspaceLimit)
  }
  left * right
}

///|
fn partitioned_workspace(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  maximum_workspace_bytes : Int64,
) -> CudaPartitionedAttentionWorkspace raise CudaAttentionTileSourceError {
  let partitions = lowering.key_value_partitions()
  let (_, _, head_dimension) = lowering.tile()
  let (query_heads, _, _, _) = abi.shape()
  let (maximum_query_rows, maximum_query_tokens, _) = abi.limits()
  let state_rows = if lowering.phase() == Decode {
    maximum_query_rows
  } else {
    maximum_query_tokens
  }
  if partitions <= 1 || maximum_workspace_bytes <= 0L {
    raise InvalidCudaAttentionTileSource(WorkspaceLimit)
  }
  let partial_state_count = partitioned_checked_product(
    partitioned_checked_product(state_rows.to_int64(), query_heads.to_int64()),
    partitions.to_int64(),
  )
  let denominator_offset_bytes = partitioned_checked_product(
    partial_state_count, 4L,
  )
  let numerator_offset_bytes = partitioned_checked_product(
    partial_state_count, 8L,
  )
  let numerator_bytes = partitioned_checked_product(
    partitioned_checked_product(partial_state_count, head_dimension.to_int64()),
    4L,
  )
  if numerator_offset_bytes > 9223372036854775807L - numerator_bytes {
    raise InvalidCudaAttentionTileSource(WorkspaceLimit)
  }
  let total_bytes = numerator_offset_bytes + numerator_bytes
  if total_bytes > maximum_workspace_bytes {
    raise InvalidCudaAttentionTileSource(WorkspaceLimit)
  }
  {
    partial_state_count,
    maximum_offset_bytes: 0L,
    denominator_offset_bytes,
    numerator_offset_bytes,
    total_bytes,
  }
}

///|
fn partitioned_merge_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  workspace : CudaPartitionedAttentionWorkspace,
  merge_symbol : String,
) -> String {
  let partitions = lowering.key_value_partitions()
  let (query_tile, _, head_dimension) = lowering.tile()
  let (_, _, block_threads, _) = lowering.launch()
  let (query_heads, _, _, _) = abi.shape()
  let (_, maximum_query_tokens, _) = abi.limits()
  let partial_state_count = workspace.partial_state_count()
  let first_row = if lowering.phase() == Decode { "prefill_count" } else { "0" }
  let final_row = if lowering.phase() == Decode {
    "row_count"
  } else {
    "prefill_count"
  }
  let state_row = if lowering.phase() == Decode { "row" } else { "token" }
  (
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|#define LF_MERGE_QUERY_TILE \{query_tile}
    $|#define LF_MERGE_HEAD_DIMENSION \{head_dimension}
    $|#define LF_MERGE_QUERY_HEADS \{query_heads}
    $|#define LF_MERGE_PARTITIONS \{partitions}
    $|#define LF_MERGE_BLOCK_THREADS \{block_threads}
    $|#define LF_MERGE_PARTIAL_STATE_COUNT \{partial_state_count}ULL
    $|
    $|extern "C" __global__ void \{merge_symbol}(
    $|    const int *counts, const int *query_row_offsets,
    $|    const float *workspace, __nv_bfloat16 *output) {
    $|  const int thread = (int)threadIdx.x;
    $|  const int query_head = (int)blockIdx.y;
    $|  if (blockDim.x != LF_MERGE_BLOCK_THREADS ||
    $|      query_head >= LF_MERGE_QUERY_HEADS) return;
    $|  const int prefill_count = counts[0];
    $|  const int decode_count = counts[1];
    $|  const int row_count = counts[2];
    $|  const int token_count = counts[3];
    $|  if (prefill_count < 0 || decode_count < 0 ||
    $|      row_count != prefill_count + decode_count || token_count <= 0 ||
    $|      token_count > \{maximum_query_tokens}) return;
    $|
    $|  for (int tile_ordinal = (int)blockIdx.x; ;
    $|       tile_ordinal += (int)gridDim.x) {
    $|    int row = -1, token_base = 0, tile_prefix = 0;
    $|    for (int candidate_row = \{first_row}; candidate_row < \{final_row}; ++candidate_row) {
    $|      const int begin = query_row_offsets[candidate_row];
    $|      const int end = query_row_offsets[candidate_row + 1];
    $|      const int row_tiles = end > begin
    $|          ? (end - begin + LF_MERGE_QUERY_TILE - 1) / LF_MERGE_QUERY_TILE : 0;
    $|      if (tile_ordinal < tile_prefix + row_tiles) {
    $|        row = candidate_row;
    $|        token_base = begin +
    $|            (tile_ordinal - tile_prefix) * LF_MERGE_QUERY_TILE;
    $|        break;
    $|      }
    $|      tile_prefix += row_tiles;
    $|    }
    $|    if (row < 0) return;
    $|    const int row_end = query_row_offsets[row + 1];
    $|    const int active_queries = min(LF_MERGE_QUERY_TILE, row_end - token_base);
    $|    if (active_queries <= 0 || row_end > token_count) return;
    $|
    $|    for (int linear = thread;
    $|         linear < active_queries * LF_MERGE_HEAD_DIMENSION;
    $|         linear += LF_MERGE_BLOCK_THREADS) {
    $|      const int query = linear / LF_MERGE_HEAD_DIMENSION;
    $|      const int component = linear % LF_MERGE_HEAD_DIMENSION;
    $|      const int token = token_base + query;
    $|      const unsigned long long first_state =
    $|          ((unsigned long long)\{state_row} * LF_MERGE_QUERY_HEADS + query_head) *
    $|          LF_MERGE_PARTITIONS;
    $|      float merged_maximum = -CUDART_INF_F;
    $|      int invalid = 0;
    $|      int nonempty = 0;
    $|      #pragma unroll
    $|      for (int split = 0; split < LF_MERGE_PARTITIONS; ++split) {
    $|        const unsigned long long state = first_state + split;
    $|        const float part_maximum = workspace[state];
    $|        const float part_denominator =
    $|            workspace[LF_MERGE_PARTIAL_STATE_COUNT + state];
    $|        if (!isfinite(part_denominator) || part_denominator < 0.0f ||
    $|            (part_denominator > 0.0f && !isfinite(part_maximum))) {
    $|          invalid = 1;
    $|        } else if (part_denominator > 0.0f) {
    $|          merged_maximum = fmaxf(merged_maximum, part_maximum);
    $|          nonempty = 1;
    $|        }
    $|      }
    $|      float merged_denominator = 0.0f;
    $|      float merged_numerator = 0.0f;
    $|      if (!invalid && nonempty) {
    $|        #pragma unroll
    $|        for (int split = 0; split < LF_MERGE_PARTITIONS; ++split) {
    $|          const unsigned long long state = first_state + split;
    $|          const float part_maximum = workspace[state];
    $|          const float part_denominator =
    $|              workspace[LF_MERGE_PARTIAL_STATE_COUNT + state];
    $|          if (part_denominator > 0.0f) {
    $|            const float merge_scale = expf(part_maximum - merged_maximum);
    $|            merged_denominator += part_denominator * merge_scale;
    $|            merged_numerator +=
    $|                workspace[2ULL * LF_MERGE_PARTIAL_STATE_COUNT +
    $|                    state * LF_MERGE_HEAD_DIMENSION + component] *
    $|                merge_scale;
    $|          }
    $|        }
    $|      }
    $|      const unsigned long long output_index =
    $|          ((unsigned long long)token * LF_MERGE_QUERY_HEADS + query_head) *
    $|          LF_MERGE_HEAD_DIMENSION + component;
    $|      output[output_index] = __float2bfloat16_rn(
    $|          !invalid && merged_denominator > 0.0f
    $|              ? merged_numerator / merged_denominator : CUDART_NAN_F);
    $|    }
    $|    __syncthreads();
    $|  }
    $|}
  )
}

///|
/// Emits the CUDA realization of the functional partition-map/local-fold/
/// associative-merge schedule. No benchmarking, allocation, or device query
/// occurs in this pure source-generation boundary.
pub fn emit_cuda_partitioned_attention_tile_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  partial_symbol : String,
  merge_symbol : String,
  maximum_workspace_bytes : Int64,
  maximum_source_bytes : Int,
) -> CudaPartitionedAttentionTileSource raise CudaAttentionTileSourceError {
  if lowering.phase() == MixedCompatibility ||
    lowering.key_value_partitions() <= 1 ||
    partial_symbol == merge_symbol ||
    !valid_cuda_symbol(partial_symbol) ||
    !valid_cuda_symbol(merge_symbol) ||
    maximum_source_bytes <= 0 ||
    maximum_source_bytes > 16777216 {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let workspace = partitioned_workspace(lowering, abi, maximum_workspace_bytes)
  let partial = match lowering.phase() {
    Prefill =>
      emit_cuda_prefill_attention_tile_source(
        lowering, abi, partial_symbol, maximum_source_bytes,
      )
    Decode =>
      if lowering.compute() == CudaGroupedQuerySplitSubgroupShuffleDot {
        emit_cuda_grouped_split_decode_attention_tile_source(
          lowering, abi, partial_symbol, maximum_source_bytes,
        )
      } else {
        emit_cuda_direct_attention_tile_source(
          lowering, abi, partial_symbol, maximum_source_bytes,
        )
      }
    MixedCompatibility =>
      raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let partial_text = @utf8.decode(partial.bytes()) catch {
    _ => raise InvalidCudaAttentionTileSource(SourceLimit)
  }
  let source = partial_text +
    "\n" +
    partitioned_merge_source(lowering, abi, workspace, merge_symbol)
  let bytes = @utf8.encode(source)
  if bytes.length() > maximum_source_bytes {
    raise InvalidCudaAttentionTileSource(SourceLimit)
  }
  {
    lowering_digest: lowering.digest(),
    abi,
    partial_symbol,
    merge_symbol,
    workspace,
    bytes,
    digest: CudaAttentionTileSourceDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    ),
  }
}
