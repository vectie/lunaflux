///|
/// Preserve the ordered PV accumulation while transfer scheduling changes.
fn cuda_prefill_value_accumulation(
  compute : @luna_cuda_attention_tile_lowering.CudaAttentionComputePrimitive,
) -> String {
  if compute == CudaMatrixQkPvMultiplyAccumulate {
    "    if (warp < LF_HEAD_DIMENSION / 16) {\n" +
    "      const int output_column = warp * 16;\n" +
    "      #pragma unroll\n" +
    "      for (int query_base = 0; query_base < LF_QUERY_TILE; query_base += 16) {\n" +
    "        auto output_fragment = lf_matrix_load_float(accumulator, query_base, output_column, LF_HEAD_DIMENSION);\n" +
    "        #pragma unroll\n" +
    "        for (int reduction = 0; reduction < LF_KEY_VALUE_TILE; reduction += 16) {\n" +
    "          const auto probability_fragment = lf_matrix_load(weights, query_base, reduction, LF_KEY_VALUE_TILE);\n" +
    "          const auto value_fragment = lf_matrix_load(value_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION, reduction, output_column, LF_HEAD_DIMENSION, true);\n" +
    "          lf_matrix_mma(output_fragment, probability_fragment, value_fragment, true);\n" +
    "        }\n" +
    "        lf_matrix_store_float(accumulator, query_base, output_column, LF_HEAD_DIMENSION, output_fragment);\n" +
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
    "      accumulator[lf_float_index(query, component, LF_HEAD_DIMENSION)] += partial;\n" +
    "    }\n"
  }
}
