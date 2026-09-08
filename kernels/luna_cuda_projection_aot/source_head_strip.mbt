///|
// The functional schedule fixes a K128 storage lifetime around eight ordered
// K16 folds. This backend only lowers its gather, copy and matrix operations.
fn projection_head_strip_reduction(
  input_width : Int,
  strip : @luna_projection_tile_compiler.ProjectionSelectedRowStrip,
) -> String {
  let width = strip.reduction_elements()
  "  __nv_bfloat16 *selected_weight = selected_input + 16 * \{width};\n" +
  "  for (int strip_begin = 0; strip_begin < \{input_width}; strip_begin += \{width}) {\n" +
  "    for (int vector = lane; vector < 16 * \{width / 8}; vector += 32) {\n" +
  "      const int local_row = vector / \{width / 8};\n" +
  "      const int local_inner = (vector % \{width / 8}) * 8;\n" +
  "      const int query_row = row + local_row;\n" +
  "      const int token_row = query_row < row_count ? row_offsets[query_row + 1] - 1 : -1;\n" +
  "      uint4 gathered = make_uint4(0U, 0U, 0U, 0U);\n" +
  "      if (token_row >= 0 && token_row < token_count) {\n" +
  "        gathered = *reinterpret_cast<const uint4 *>(input + (unsigned long long)token_row * \{input_width}ULL + strip_begin + local_inner);\n" +
  "      }\n" +
  "      *reinterpret_cast<uint4 *>(selected_input + local_row * \{width} + local_inner) = gathered;\n" +
  "      *reinterpret_cast<uint4 *>(selected_weight + local_row * \{width} + local_inner) = *reinterpret_cast<const uint4 *>(weight + (unsigned long long)(column + local_row) * \{input_width}ULL + strip_begin + local_inner);\n" +
  "    }\n" +
  "    __syncwarp();\n" +
  "    #pragma unroll\n" +
  "    for (int inner = 0; inner < \{width}; inner += 16) {\n" +
  "      nvcuda::wmma::load_matrix_sync(a, selected_input + inner, \{width});\n" +
  "      nvcuda::wmma::load_matrix_sync(b, selected_weight + inner, \{width});\n" +
  "      nvcuda::wmma::mma_sync(c, a, b, c);\n" +
  "    }\n" +
  "    __syncwarp();\n" +
  "  }\n"
}
