///|
/// Realize the ordered map/reduce/pack consumer without retaining a whole
/// exponential fragment. Each pair contributes in the old element order and
/// rounds independently to BF16 before its word is published. Address origins
/// and the row XOR are shared across the separate PV products, not the MMA
/// instruction region: accumulator live ranges remain independent.
fn cuda_streaming_probability_value(
  fragments : @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding,
  value_columns : Int,
  value_quantum : Int,
) -> String {
  let packing = fragments.probabilities()
  let (words, elements) = packing.geometry()
  let (dimensions, vector, rotations, row_group) = fragments
    .value_addresses()
    .layout()
    .geometry()
  guard words == 4 &&
    elements == 2 &&
    vector == 8 &&
    value_quantum == 16 &&
    value_columns * 16 == dimensions else {
    abort("unsupported streaming probability instruction binding")
  }
  let source = StringBuilder()
  source <+ "        // ordered-pair-v1; value-address-row-xor-factored-v1\n"
  source <+ "        #pragma unroll\n"
  source <+ "        for (int key=0;key<LF_KEY_VALUE_TILE/16;++key) {\n"
  source <+ "          LfMatrixOperand probability;\n"
  for word in 0..<words {
    let low = word * elements
    let high = low + 1
    let low_owner = packing.owner(word, 0).unwrap()
    let high_owner = packing.owner(word, 1).unwrap()
    source <+ "          {\n"
    source <+
      "            const float low=lf_masked_exp(score[key].x[\{low}],score[key].x[\{low}]-next[\{low_owner}]);\n"
    source <+ "            sum[\{low_owner}]+=low;\n"
    source <+
      "            const float high=lf_masked_exp(score[key].x[\{high}],score[key].x[\{high}]-next[\{high_owner}]);\n"
    source <+ "            sum[\{high_owner}]+=high;\n"
    source <+
      "            probability.x[\{word}]=lf_probability_pair(low,high);\n"
    source <+ "          }\n"
  }
  source <+ "          const int value_row=key*\{value_quantum}+(lane&15);\n"
  source <+
    "          const unsigned value_origin=(unsigned)__cvta_generic_to_shared(current_v+value_row*LF_HEAD_DIMENSION);\n"
  source <+
    "          const unsigned value_column_xor=(lane/16*8)^((value_row/\{row_group}%\{rotations})*\{vector});\n"
  source <+ "          #pragma unroll\n"
  source <+ "          for (int column=0;column<\{value_columns};++column) {\n"
  source <+
    "            const unsigned value_address=value_origin+((column*16)^value_column_xor)*2;\n"
  source <+
    "            lf_matrix_mma_value(out[column],probability,value_address);\n"
  source <+ "          }\n"
  source <+ "        }\n"
  source.to_string()
}
