///|
/// CUDA lowering for the functional middle-end rule
/// `ReuseInputTileAcrossOutputTiles`. One workgroup stages a 64x16 input tile
/// and a 32x16 weight tile, then maps their Cartesian product to eight 16x16
/// matrix fragments. Device geometry stays private to this backend.
fn projection_tensor_core_cooperative_dense_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_tensor_core_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    "  if (token_count == 1) {\n" +
    "    const int output_column = (int)blockIdx.x * 8 + warp; if (output_column >= " +
    parameters.output_width.to_string() +
    ") return;\n" +
    "    const unsigned long long weight_base = (unsigned long long)output_column * " +
    parameters.input_width.to_string() +
    "ULL; float value = 0.0f;\n" +
    "    for (int inner = lane; inner < " +
    parameters.input_width.to_string() +
    "; inner += 32) value = __fadd_rn(value, __fmul_rn(__bfloat162float(input[inner]), __bfloat162float(weight[weight_base + inner])));\n" +
    "    for (int delta = 16; delta > 0; delta >>= 1) value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, delta));\n" +
    "    if (lane == 0) output[output_column] = __float2bfloat16_rn(value);\n" +
    "    return;\n  }\n" +
    "  const int column_groups = " +
    ((parameters.output_width + 31) / 32).to_string() +
    ";\n" +
    "  const int column_group = (int)blockIdx.x % column_groups;\n" +
    "  const int row_begin = ((int)blockIdx.x / column_groups) * 64;\n" +
    "  if (row_begin >= token_count) return;\n" +
    "  const int warp_row = warp >> 1;\n" +
    "  const int warp_column = warp & 1;\n" +
    "  const int row = row_begin + warp_row * 16;\n" +
    "  const int column = column_group * 32 + warp_column * 16;\n" +
    "  extern __shared__ unsigned char projection_shared[];\n" +
    "  __nv_bfloat16 *input_tile = reinterpret_cast<__nv_bfloat16 *>(projection_shared);\n" +
    "  __nv_bfloat16 *weight_tile = input_tile + 1024;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;\n" +
    "  nvcuda::wmma::fill_fragment(c, 0.0f);\n" +
    "  for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; inner += 16) {\n" +
    "    for (int element = (int)threadIdx.x; element < 1024; element += (int)blockDim.x) {\n" +
    "      const int local_row = element >> 4; const int local_inner = element & 15;\n" +
    "      input_tile[element] = row_begin + local_row < token_count ? input[(unsigned long long)(row_begin + local_row) * " +
    parameters.input_width.to_string() +
    "ULL + inner + local_inner] : __float2bfloat16_rn(0.0f);\n" +
    "    }\n" +
    "    for (int element = (int)threadIdx.x; element < 512; element += (int)blockDim.x) {\n" +
    "      const int local_column = element >> 4; const int local_inner = element & 15;\n" +
    "      const int output_column = column_group * 32 + local_column;\n" +
    "      weight_tile[element] = output_column < " +
    parameters.output_width.to_string() +
    " ? weight[(unsigned long long)output_column * " +
    parameters.input_width.to_string() +
    "ULL + inner + local_inner] : __float2bfloat16_rn(0.0f);\n" +
    "    }\n" +
    "    __syncthreads();\n" +
    "    nvcuda::wmma::load_matrix_sync(a, input_tile + warp_row * 256, 16);\n" +
    "    nvcuda::wmma::load_matrix_sync(b, weight_tile + warp_column * 256, 16);\n" +
    "    nvcuda::wmma::mma_sync(c, a, b, c);\n" +
    "    __syncthreads();\n" +
    "  }\n" +
    "  if (row + 15 < token_count && column + 15 < " +
    parameters.output_width.to_string() +
    ") {\n" +
    "    nvcuda::wmma::store_matrix_sync(output + (unsigned long long)row * " +
    parameters.output_width.to_string() +
    "ULL + column, c, " +
    parameters.output_width.to_string() +
    ", nvcuda::wmma::mem_row_major);\n" +
    "  } else {\n" +
    "    float *warp_output = reinterpret_cast<float *>(projection_shared) + warp * 256;\n" +
    "    nvcuda::wmma::store_matrix_sync(warp_output, c, 16, nvcuda::wmma::mem_row_major);\n" +
    "    __syncwarp();\n" +
    "    for (int element = lane; element < 256; element += 32) {\n" +
    "      const int local_row = element >> 4; const int local_column = element & 15;\n" +
    "      if (row + local_row < token_count && column + local_column < " +
    parameters.output_width.to_string() +
    ") output[(unsigned long long)(row + local_row) * " +
    parameters.output_width.to_string() +
    "ULL + column + local_column] = __float2bfloat16_rn(warp_output[element]);\n" +
    "    }\n" +
    "  }\n}\n",
  )
}

///|
/// The QKV variant uses the same tile schedule while preserving three distinct
/// immutable weight operands. Weight selection happens during cooperative
/// staging, before fragments consume the backend-local tile.
fn projection_tensor_core_cooperative_qkv_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_tensor_core_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, const __nv_bfloat16 *value_weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    "  if (token_count == 1) {\n" +
    "    const int output_column = (int)blockIdx.x * 8 + warp; if (output_column >= " +
    parameters.output_width.to_string() +
    ") return;\n" +
    "    const __nv_bfloat16 *selected_weight = query_weight; int selected_column = output_column;\n" +
    "    if (output_column >= " +
    parameters.query_width.to_string() +
    ") { selected_column -= " +
    parameters.query_width.to_string() +
    "; if (selected_column >= " +
    parameters.intermediate_width.to_string() +
    ") { selected_column -= " +
    parameters.intermediate_width.to_string() +
    "; selected_weight = value_weight; } else selected_weight = key_weight; }\n" +
    "    const unsigned long long weight_base = (unsigned long long)selected_column * " +
    parameters.input_width.to_string() +
    "ULL; float value = 0.0f;\n" +
    "    for (int inner = lane; inner < " +
    parameters.input_width.to_string() +
    "; inner += 32) value = __fadd_rn(value, __fmul_rn(__bfloat162float(input[inner]), __bfloat162float(selected_weight[weight_base + inner])));\n" +
    "    for (int delta = 16; delta > 0; delta >>= 1) value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, delta));\n" +
    "    if (lane == 0) output[output_column] = __float2bfloat16_rn(value);\n" +
    "    return;\n  }\n" +
    "  const int column_groups = " +
    ((parameters.output_width + 31) / 32).to_string() +
    ";\n" +
    "  const int column_group = (int)blockIdx.x % column_groups;\n" +
    "  const int row_begin = ((int)blockIdx.x / column_groups) * 64;\n" +
    "  if (row_begin >= token_count) return;\n" +
    "  const int warp_row = warp >> 1; const int warp_column = warp & 1;\n" +
    "  const int row = row_begin + warp_row * 16;\n" +
    "  const int column = column_group * 32 + warp_column * 16;\n" +
    "  extern __shared__ unsigned char projection_shared[];\n" +
    "  __nv_bfloat16 *input_tile = reinterpret_cast<__nv_bfloat16 *>(projection_shared);\n" +
    "  __nv_bfloat16 *weight_tile = input_tile + 1024;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;\n" +
    "  nvcuda::wmma::fill_fragment(c, 0.0f);\n" +
    "  for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; inner += 16) {\n" +
    "    for (int element = (int)threadIdx.x; element < 1024; element += (int)blockDim.x) {\n" +
    "      const int local_row = element >> 4; const int local_inner = element & 15;\n" +
    "      input_tile[element] = row_begin + local_row < token_count ? input[(unsigned long long)(row_begin + local_row) * " +
    parameters.input_width.to_string() +
    "ULL + inner + local_inner] : __float2bfloat16_rn(0.0f);\n" +
    "    }\n" +
    "    for (int element = (int)threadIdx.x; element < 512; element += (int)blockDim.x) {\n" +
    "      const int local_column = element >> 4; const int local_inner = element & 15;\n" +
    "      const int output_column = column_group * 32 + local_column;\n" +
    "      const __nv_bfloat16 *selected_weight = query_weight; int selected_column = output_column;\n" +
    "      if (output_column >= " +
    parameters.query_width.to_string() +
    ") { selected_column -= " +
    parameters.query_width.to_string() +
    "; if (selected_column >= " +
    parameters.intermediate_width.to_string() +
    ") { selected_column -= " +
    parameters.intermediate_width.to_string() +
    "; selected_weight = value_weight; } else selected_weight = key_weight; }\n" +
    "      weight_tile[element] = output_column < " +
    parameters.output_width.to_string() +
    " ? selected_weight[(unsigned long long)selected_column * " +
    parameters.input_width.to_string() +
    "ULL + inner + local_inner] : __float2bfloat16_rn(0.0f);\n" +
    "    }\n" +
    "    __syncthreads();\n" +
    "    nvcuda::wmma::load_matrix_sync(a, input_tile + warp_row * 256, 16);\n" +
    "    nvcuda::wmma::load_matrix_sync(b, weight_tile + warp_column * 256, 16);\n" +
    "    nvcuda::wmma::mma_sync(c, a, b, c);\n" +
    "    __syncthreads();\n" +
    "  }\n" +
    "  if (row + 15 < token_count && column + 15 < " +
    parameters.output_width.to_string() +
    ") {\n" +
    "    nvcuda::wmma::store_matrix_sync(output + (unsigned long long)row * " +
    parameters.output_width.to_string() +
    "ULL + column, c, " +
    parameters.output_width.to_string() +
    ", nvcuda::wmma::mem_row_major);\n" +
    "  } else {\n" +
    "    float *warp_output = reinterpret_cast<float *>(projection_shared) + warp * 256;\n" +
    "    nvcuda::wmma::store_matrix_sync(warp_output, c, 16, nvcuda::wmma::mem_row_major);\n" +
    "    __syncwarp();\n" +
    "    for (int element = lane; element < 256; element += 32) {\n" +
    "      const int local_row = element >> 4; const int local_column = element & 15;\n" +
    "      if (row + local_row < token_count && column + local_column < " +
    parameters.output_width.to_string() +
    ") output[(unsigned long long)(row + local_row) * " +
    parameters.output_width.to_string() +
    "ULL + column + local_column] = __float2bfloat16_rn(warp_output[element]);\n" +
    "    }\n" +
    "  }\n}\n",
  )
}
