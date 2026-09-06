///|
/// A pure row-selection map lifted out of the reduction fold. The backend
/// emits local mutation only after the compiler has chosen this storage scope.
fn projection_resident_selected_rows(input_width : Int) -> String {
  "  for (int element = (int)threadIdx.x; element < \{16 * input_width}; element += (int)blockDim.x) {\n" +
  "    const int query_row = row + element / \{input_width};\n" +
  "    const int component = element % \{input_width};\n" +
  "    const int token_row = query_row < row_count ? row_offsets[query_row + 1] - 1 : -1;\n" +
  "    selected_input[element] = token_row >= 0 && token_row < token_count ? input[(unsigned long long)token_row * \{input_width}ULL + component] : __float2bfloat16_rn(0.0f);\n" +
  "  }\n  __syncthreads();\n"
}
