///|
/// Terminal realization of FuseScoreTransform. The generic optimizer has
/// composed scale/mask with the softmax region; do not materialize its mapped
/// tensor between the matrix dot and fold consumers. Explicit rounding keeps
/// the former float-store boundary even when the device compiler contracts
/// arithmetic in the consuming exponential expression.
fn cuda_attention_score_transform() -> String {
  let source =
    #|static __device__ __forceinline__ float lf_transformed_score(
    #|    const float *scores, int query, int key, int key_base,
    #|    int partition_end, int query_position, float scale) {
    #|  const int key_position = key_base + key;
    #|  return key_position < partition_end && key_position <= query_position
    #|      ? __fmul_rn(scores[lf_float_index(query, key, LF_KEY_VALUE_TILE)], scale)
    #|      : -CUDART_INF_F;
    #|}
  source
}
