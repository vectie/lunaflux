///|
/// CUDA chooses the matrix-load microtile extent. The portable planner validates
/// the extents; copy producers and matrix consumers share the row XOR map below.
/// No capacity expansion or storage lifetime change is required.
fn cuda_attention_operand_layout(
  query : Int,
  keys : Int,
  dimension : Int,
) -> String raise CudaAttentionTileSourceError {
  guard @luna_attention_tile_schedule.attention_operand_tile_layout(
      query, dimension, 8,
    )
    is Some(layout) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  guard @luna_attention_tile_schedule.attention_operand_tile_layout(
      keys, dimension, 8,
    )
    is Some(_) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  guard @luna_attention_tile_schedule.attention_operand_tile_layout(
      query, keys, 8,
    )
    is Some(_) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let side = layout.microtile_side()
  (
    $|#define LF_OPERAND_TILE \{side}
    $|static __device__ __forceinline__ int lf_operand_index(int row, int column, int columns) {
    $|  const int vectors = columns / 8;
    $|  const int rotations = vectors % 8 == 0 ? 8 : (vectors % 4 == 0 ? 4 : 2);
    $|  return (unsigned)row * columns + ((unsigned)column ^ ((((unsigned)row / (8 / rotations)) % rotations) * 8));
    $|}
  )
}
