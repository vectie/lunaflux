///|
/// Specialize the same ordered score fold for interior and boundary tiles.
/// The interior instantiation contains no per-element causal/tail predicate.
fn cuda_query_owned_score_fold() -> String {
  let source =
    #|template<bool Boundary>
    #|__device__ __forceinline__ void lf_query_score_fold(
    #|    LfMatrixAccumulator *score, float *next, float scale,
    #|    int kb, int key_end, int p0, int p1, int lane) {
    #|  #pragma unroll
    #|  for (int key=0; key<LF_KEY_VALUE_TILE/16; ++key) {
    #|    #pragma unroll
    #|    for (int item=0; item<8; ++item) {
    #|      const int r=item%4/2;
    #|      float x=score[key].x[item]*scale;
    #|      if (Boundary) {
    #|        const int position=kb+key*16+lf_matrix_column(item,lane);
    #|        if (!(position<key_end && position<=(r==0?p0:p1))) x=-CUDART_INF_F;
    #|      }
    #|      score[key].x[item]=x;
    #|      next[r]=fmaxf(next[r],x);
    #|    }
    #|  }
    #|}
  source
}
