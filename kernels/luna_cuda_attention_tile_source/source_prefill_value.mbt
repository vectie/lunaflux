///|
/// Preserve the ordered PV accumulation while transfer scheduling changes.
fn cuda_prefill_value_accumulation(
  compute : @luna_cuda_attention_tile_lowering.CudaAttentionComputePrimitive,
) -> String {
  if compute == CudaMatrixQkPvMultiplyAccumulate {
    "    if (warp < LF_HEAD_DIMENSION / 16) {\n" +
    "      using namespace nvcuda;\n" +
    "      const int output_column = warp * 16;\n" +
    "      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> probability_fragment;\n" +
    "      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> value_fragment;\n" +
    "      wmma::fragment<wmma::accumulator, 16, 16, 16, float> output_fragment;\n" +
    "      #pragma unroll\n" +
    "      for (int query_base = 0; query_base < LF_QUERY_TILE; query_base += 16) {\n" +
    "        wmma::load_matrix_sync(output_fragment,\n" +
    "            accumulator + query_base * LF_HEAD_DIMENSION + output_column,\n" +
    "            LF_HEAD_DIMENSION, wmma::mem_row_major);\n" +
    "        #pragma unroll\n" +
    "        for (int reduction = 0; reduction < LF_KEY_VALUE_TILE; reduction += 16) {\n" +
    "          wmma::load_matrix_sync(probability_fragment,\n" +
    "              weights + lf_operand_index(query_base, reduction, LF_KEY_VALUE_TILE),\n" +
    "              LF_OPERAND_TILE);\n" +
    "          wmma::load_matrix_sync(value_fragment,\n" +
    "              value_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +\n" +
    "                  lf_operand_index(reduction, output_column, LF_HEAD_DIMENSION),\n" +
    "              LF_OPERAND_TILE);\n" +
    "          wmma::mma_sync(output_fragment, probability_fragment, value_fragment, output_fragment);\n" +
    "        }\n" +
    "        wmma::store_matrix_sync(\n" +
    "            accumulator + query_base * LF_HEAD_DIMENSION + output_column,\n" +
    "            output_fragment, LF_HEAD_DIMENSION, wmma::mem_row_major);\n" +
    "      }\n" +
    "    }\n"
  } else {
    "    for (int linear = thread; linear < active_queries * LF_HEAD_DIMENSION;\n" +
    "         linear += LF_BLOCK_THREADS) {\n" +
    "      const int query = linear / LF_HEAD_DIMENSION;\n" +
    "      const int component = linear % LF_HEAD_DIMENSION;\n" +
    "      float partial = 0.0f;\n" +
    "      #pragma unroll 4\n" +
    "      for (int key = 0; key < LF_KEY_VALUE_TILE; ++key)\n" +
    "        partial += weights[query * LF_KEY_VALUE_TILE + key] *\n" +
    "            __bfloat162float(value_tiles[\n" +
    "                stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +\n" +
    "                lf_operand_index(key, component, LF_HEAD_DIMENSION)]);\n" +
    "      accumulator[linear] += partial;\n" +
    "    }\n"
  }
}
