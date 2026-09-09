///|
/// CUDA chooses the WMMA microtile extent. The portable planner validates the
/// compact permutation; copy producers and matrix consumers share this map.
/// No capacity expansion or storage lifetime change is required.
fn cuda_attention_operand_layout(
  query : Int,
  keys : Int,
  dimension : Int,
) -> String raise CudaAttentionTileSourceError {
  guard @luna_attention_tile_schedule.attention_operand_tile_layout(
      query, dimension, 16,
    )
    is Some(layout) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  guard @luna_attention_tile_schedule.attention_operand_tile_layout(
      keys, dimension, 16,
    )
    is Some(_) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  guard @luna_attention_tile_schedule.attention_operand_tile_layout(
      query, keys, 16,
    )
    is Some(_) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let side = layout.microtile_side()
  (
    $|#define LF_OPERAND_TILE \{side}
    $|static __device__ __forceinline__ int lf_operand_index(int row, int column, int columns) {
    $|  return ((row / LF_OPERAND_TILE) * (columns / LF_OPERAND_TILE) +
    $|      column / LF_OPERAND_TILE) * LF_OPERAND_TILE * LF_OPERAND_TILE +
    $|      (row % LF_OPERAND_TILE) * LF_OPERAND_TILE + column % LF_OPERAND_TILE;
    $|}
  )
}
