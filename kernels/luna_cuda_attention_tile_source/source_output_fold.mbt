///|
/// Learn the opaque WMMA accumulator layout through its documented load
/// operation once, rather than encoding an architecture-specific lane map.
/// The shared terminal-output region is scratch only during this prologue.
fn cuda_resident_output_fold_declaration() -> String {
  (
    #|  using OutputFoldFragment = LfMatrixAccumulator;
    #|  OutputFoldFragment output_fold[LF_QUERY_TILE / 16];
    #|  unsigned char output_fold_rows[LF_QUERY_TILE / 16][OutputFoldFragment::num_elements];
    #|  if (warp < LF_HEAD_DIMENSION / 16) {
    #|    #pragma unroll
    #|    for (int slice = 0; slice < LF_QUERY_TILE / 16; ++slice) {
    #|      #pragma unroll
    #|      for (int item = 0; item < OutputFoldFragment::num_elements; ++item)
    #|        output_fold_rows[slice][item] = (unsigned char)(slice * 16 + lf_matrix_row(item, lane));
    #|      output_fold[slice] = {};
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
    #|          const auto probability_fragment = lf_matrix_load(weights, slice * 16, reduction, LF_KEY_VALUE_TILE);
    #|          const auto value_fragment = lf_matrix_load(value_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION, reduction, warp * 16, LF_HEAD_DIMENSION, true);
    #|          lf_matrix_mma(output_fold[slice], probability_fragment, value_fragment, true);
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
    #|      lf_matrix_store_float(accumulator, slice * 16, warp * 16, LF_HEAD_DIMENSION, output_fold[slice]);
    #|  }
    #|  __syncthreads();
    #|
  )
}
