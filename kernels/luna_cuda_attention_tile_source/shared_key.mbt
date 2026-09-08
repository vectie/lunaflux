///|
/// Terminal map/fold interchange: a tuple of independent score accumulators
/// shares each matrix-B load. Matrix-A loads and each accumulator's reduction
/// order are unchanged. CUDA fragment layout stays entirely in this backend.
fn cuda_attention_shared_key_dot() -> String {
  let source =
    #|    if (warp < LF_KEY_VALUE_TILE / 16) {
    #|      using namespace nvcuda;
    #|      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> query_fragment;
    #|      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> key_fragment;
    #|      wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_fragments[LF_QUERY_TILE / 16];
    #|      const int key_column = warp * 16;
    #|      #pragma unroll
    #|      for (int query_slice = 0; query_slice < LF_QUERY_TILE / 16; ++query_slice)
    #|        wmma::fill_fragment(score_fragments[query_slice], 0.0f);
    #|      #pragma unroll
    #|      for (int reduction = 0; reduction < LF_HEAD_DIMENSION; reduction += 16) {
    #|        wmma::load_matrix_sync(key_fragment,
    #|            key_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +
    #|                key_column * LF_HEAD_DIMENSION + reduction,
    #|            LF_HEAD_DIMENSION);
    #|        #pragma unroll
    #|        for (int query_slice = 0; query_slice < LF_QUERY_TILE / 16; ++query_slice) {
    #|          wmma::load_matrix_sync(query_fragment,
    #|              query_tile + query_slice * 16 * LF_HEAD_DIMENSION + reduction,
    #|              LF_HEAD_DIMENSION);
    #|          wmma::mma_sync(score_fragments[query_slice], query_fragment, key_fragment, score_fragments[query_slice]);
    #|        }
    #|      }
    #|      #pragma unroll
    #|      for (int query_slice = 0; query_slice < LF_QUERY_TILE / 16; ++query_slice)
    #|        wmma::store_matrix_sync(
    #|            scores + query_slice * 16 * LF_KEY_VALUE_TILE + key_column,
    #|            score_fragments[query_slice], LF_KEY_VALUE_TILE, wmma::mem_row_major);
    #|    }
  source
}
