///|
/// Learn the opaque WMMA accumulator layout through its documented load
/// operation once, rather than encoding an architecture-specific lane map.
/// The shared terminal-output region is scratch only during this prologue.
fn cuda_resident_output_fold_declaration() -> String {
  (
    #|  using OutputFoldFragment = nvcuda::wmma::fragment<
    #|      nvcuda::wmma::accumulator, 16, 16, 16, float>;
    #|  OutputFoldFragment output_fold[LF_QUERY_TILE / 16];
    #|  unsigned char output_fold_rows[LF_QUERY_TILE / 16][OutputFoldFragment::num_elements];
    #|  if (warp < LF_HEAD_DIMENSION / 16) {
    #|    #pragma unroll
    #|    for (int slice = 0; slice < LF_QUERY_TILE / 16; ++slice) {
    #|      nvcuda::wmma::load_matrix_sync(output_fold[slice],
    #|          accumulator + slice * 16 * LF_HEAD_DIMENSION + warp * 16,
    #|          LF_HEAD_DIMENSION, nvcuda::wmma::mem_row_major);
    #|      #pragma unroll
    #|      for (int item = 0; item < OutputFoldFragment::num_elements; ++item)
    #|        output_fold_rows[slice][item] = (unsigned char)output_fold[slice].x[item];
    #|      nvcuda::wmma::fill_fragment(output_fold[slice], 0.0f);
    #|    }
    #|  }
    #|  __syncthreads();
    #|
  )
}

///|
fn cuda_resident_output_fold_step() -> String {
  (
    #|    if (warp < LF_HEAD_DIMENSION / 16) {
    #|      using namespace nvcuda;
    #|      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> probability_fragment;
    #|      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> value_fragment;
    #|      #pragma unroll
    #|      for (int slice = 0; slice < LF_QUERY_TILE / 16; ++slice) {
    #|        #pragma unroll
    #|        for (int item = 0; item < OutputFoldFragment::num_elements; ++item) {
    #|          const int query = output_fold_rows[slice][item];
    #|          const float previous_scale = query < active_queries ? previous_scales[query] : 0.0f;
    #|          output_fold[slice].x[item] = __fmul_rn(output_fold[slice].x[item], previous_scale);
    #|        }
    #|        #pragma unroll
    #|        for (int reduction = 0; reduction < LF_KEY_VALUE_TILE; reduction += 16) {
    #|          wmma::load_matrix_sync(probability_fragment,
    #|              weights + slice * 16 * LF_KEY_VALUE_TILE + reduction, LF_KEY_VALUE_TILE);
    #|          wmma::load_matrix_sync(value_fragment,
    #|              value_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +
    #|                  reduction * LF_HEAD_DIMENSION + warp * 16, LF_HEAD_DIMENSION);
    #|          wmma::mma_sync(output_fold[slice], probability_fragment, value_fragment, output_fold[slice]);
    #|        }
    #|      }
    #|    }
    #|
  )
}

///|
fn cuda_resident_output_fold_store() -> String {
  (
    #|  if (warp < LF_HEAD_DIMENSION / 16) {
    #|    #pragma unroll
    #|    for (int slice = 0; slice < LF_QUERY_TILE / 16; ++slice)
    #|      nvcuda::wmma::store_matrix_sync(
    #|          accumulator + slice * 16 * LF_HEAD_DIMENSION + warp * 16,
    #|          output_fold[slice], LF_HEAD_DIMENSION, nvcuda::wmma::mem_row_major);
    #|  }
    #|  __syncthreads();
    #|
  )
}
