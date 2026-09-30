///|
/// Lower a single-owner statistics map and disjoint component fold. Unlike
/// the generic row-tile merge, decode maps one row directly to each CTA;
/// no row discovery, per-component exponentials, or repeated state scans.
fn cuda_decode_partition_merge_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  workspace : CudaPartitionedAttentionWorkspace,
  symbol : String,
  program : @attention_physical_ir.PartitionMerge,
) -> String {
  let (partitions, dimensions) = program.dimensions()
  let (_, _, threads, _) = lowering.launch()
  let (heads, _, _, _) = abi.shape()
  let (max_rows, max_tokens, _) = abi.limits()
  let states = workspace.partial_state_count()
  let body = program
    .steps()
    .map(step => {
      match step {
        ReducePartitionMaximum =>
          (
            #|  if (thread == 0) {
            #|    float maximum = -CUDART_INF_F;
            #|    bool invalid = false;
            #|    #pragma unroll
            #|    for (int split = 0; split < LF_MERGE_PARTITIONS; ++split) {
            #|      const float m = workspace[first_state + split];
            #|      const float d = workspace[LF_MERGE_PARTIAL_STATE_COUNT + first_state + split];
            #|      invalid |= !isfinite(d) || d < 0.0f || (d > 0.0f && !isfinite(m));
            #|      if (d > 0.0f) maximum = fmaxf(maximum, m);
            #|    }
          )
        FormPartitionScalesAndDenominator =>
          (
            #|    float denominator = 0.0f;
            #|    #pragma unroll
            #|    for (int split = 0; split < LF_MERGE_PARTITIONS; ++split) {
            #|      const float d = workspace[LF_MERGE_PARTIAL_STATE_COUNT + first_state + split];
            #|      const float scale = !invalid && d > 0.0f
            #|          ? expf(workspace[first_state + split] - maximum) : 0.0f;
            #|      partition_scales[split] = scale;
            #|      partition_active[split] = !invalid && d > 0.0f;
            #|      denominator += d * scale;
            #|    }
            #|    partition_scales[LF_MERGE_PARTITIONS] = invalid ? CUDART_NAN_F : denominator;
            #|  }
          )
        PublishPartitionStatistics => "  __syncthreads();"
        AccumulatePartitionNumerators =>
          (
            #|  const float denominator = partition_scales[LF_MERGE_PARTITIONS];
            #|  for (int component = thread; component < LF_MERGE_HEAD_DIMENSION;
            #|       component += LF_MERGE_BLOCK_THREADS) {
            #|    float numerator = 0.0f;
            #|    if (isfinite(denominator) && denominator > 0.0f) {
            #|      #pragma unroll
            #|      for (int split = 0; split < LF_MERGE_PARTITIONS; ++split) {
            #|        const float scale = partition_scales[split];
            #|        if (partition_active[split])
            #|          numerator += workspace[2ULL * LF_MERGE_PARTIAL_STATE_COUNT +
            #|              (first_state + split) * LF_MERGE_HEAD_DIMENSION + component] * scale;
            #|      }
            #|    }
          )
        NormalizePartitionNumerators =>
          (
            #|    const unsigned long long output_index =
            #|        ((unsigned long long)token * LF_MERGE_QUERY_HEADS + query_head) *
            #|        LF_MERGE_HEAD_DIMENSION + component;
            #|    output[output_index] = __float2bfloat16_rn(
            #|        isfinite(denominator) && denominator > 0.0f
            #|            ? numerator / denominator : CUDART_NAN_F);
            #|  }
          )
      }
    })
    .join("\n")
  (
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|// partition_merge_law=\{program.numeric_law()}
    $|#define LF_MERGE_HEAD_DIMENSION \{dimensions}
    $|#define LF_MERGE_QUERY_HEADS \{heads}
    $|#define LF_MERGE_PARTITIONS \{partitions}
    $|#define LF_MERGE_BLOCK_THREADS \{threads}
    $|#define LF_MERGE_PARTIAL_STATE_COUNT \{states}ULL
    $|extern "C" __global__ void \{symbol}(
    $|    const int *counts, const int *query_row_offsets,
    $|    const float *workspace, __nv_bfloat16 *output) {
    $|  const int thread = (int)threadIdx.x;
    $|  const int query_head = (int)blockIdx.y;
    $|  const int local_row = (int)blockIdx.x;
    $|  if (blockDim.x != LF_MERGE_BLOCK_THREADS || query_head >= LF_MERGE_QUERY_HEADS) return;
    $|  const int prefill_count = counts[0], decode_count = counts[1];
    $|  const int row_count = counts[2], token_count = counts[3];
    $|  if (prefill_count < 0 || decode_count <= 0 || row_count != prefill_count + decode_count ||
    $|      row_count > \{max_rows} || token_count <= 0 || token_count > \{max_tokens} || local_row >= decode_count) return;
    $|  const int row = prefill_count + local_row;
    $|  const int token = query_row_offsets[row], end = query_row_offsets[row + 1];
    $|  if (token < 0 || token >= token_count || end != token + 1 || end > token_count) return;
    $|  const unsigned long long first_state =
    $|      ((unsigned long long)row * LF_MERGE_QUERY_HEADS + query_head) * LF_MERGE_PARTITIONS;
    $|  __shared__ float partition_scales[LF_MERGE_PARTITIONS + 1];
    $|  __shared__ unsigned char partition_active[LF_MERGE_PARTITIONS];
    $|\{body}
    $|}
    $|#undef LF_MERGE_HEAD_DIMENSION
    $|#undef LF_MERGE_QUERY_HEADS
    $|#undef LF_MERGE_PARTITIONS
    $|#undef LF_MERGE_BLOCK_THREADS
    $|#undef LF_MERGE_PARTIAL_STATE_COUNT
  )
}
