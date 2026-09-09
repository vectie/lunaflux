///|
/// Terminal map/fold interchange: a tuple of independent score accumulators
/// shares each matrix-B load. Matrix-A loads and each accumulator's reduction
/// order are unchanged. CUDA fragment layout stays entirely in this backend.
fn cuda_attention_shared_key_dot() -> String {
  let source =
    #|    if (warp < LF_KEY_VALUE_TILE / 16) {
    #|      LfMatrixAccumulator score_fragments[LF_QUERY_TILE / 16] = {};
    #|      const int key_column = warp * 16;
    #|      #pragma unroll
    #|      for (int reduction = 0; reduction < LF_HEAD_DIMENSION; reduction += 16) {
    #|        const auto key_fragment = lf_matrix_load(key_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION, key_column, reduction, LF_HEAD_DIMENSION);
    #|        #pragma unroll
    #|        for (int query_slice = 0; query_slice < LF_QUERY_TILE / 16; ++query_slice) {
    #|          const auto query_fragment = lf_matrix_load(query_tile, query_slice * 16, reduction, LF_HEAD_DIMENSION);
    #|          lf_matrix_mma(score_fragments[query_slice], query_fragment, key_fragment);
    #|        }
    #|      }
    #|      #pragma unroll
    #|      for (int query_slice = 0; query_slice < LF_QUERY_TILE / 16; ++query_slice)
    #|        lf_matrix_store_float(scores, query_slice * 16, key_column, LF_KEY_VALUE_TILE, score_fragments[query_slice]);
    #|    }
  source
}
