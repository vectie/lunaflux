///|
fn projection_mlp_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, " +
    "const __nv_bfloat16 *down_weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const unsigned long long live = (unsigned long long)token_count * " +
    parameters.output_width.to_string() +
    "ULL;\n" +
    "  if (index >= live) return;\n" +
    "  const unsigned long long row = index / " +
    parameters.output_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long output_column = index % " +
    parameters.output_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long input_base = row * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "  const unsigned long long down_base = output_column * " +
    parameters.intermediate_width.to_string() +
    "ULL;\n" +
    "  float output_accumulator = 0.0f;\n" +
    "  for (int intermediate = 0; intermediate < " +
    parameters.intermediate_width.to_string() +
    "; ++intermediate) {\n" +
    "    const unsigned long long projection_base = " +
    "(unsigned long long)intermediate * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "    float gate = 0.0f;\n    float up = 0.0f;\n" +
    "    for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; ++inner) {\n" +
    "      const float input_value = __bfloat162float(input[input_base + inner]);\n" +
    "      gate = __fadd_rn(gate, __fmul_rn(input_value, " +
    "__bfloat162float(gate_weight[projection_base + inner])));\n" +
    "      up = __fadd_rn(up, __fmul_rn(input_value, " +
    "__bfloat162float(up_weight[projection_base + inner])));\n" +
    "    }\n" +
    "    const float silu = gate / __fadd_rn(1.0f, expf(-gate));\n" +
    "    const float combined = __fmul_rn(silu, up);\n" +
    "    const float down = __bfloat162float(down_weight[down_base + intermediate]);\n" +
    "    output_accumulator = __fadd_rn(output_accumulator, __fmul_rn(combined, down));\n" +
    "  }\n" +
    "  output[index] = __float2bfloat16_rn(output_accumulator);\n}\n",
  )
}

///|
/// Executes a full 16-row tile with BF16 WMMA warps. Gate/up uses sixteen warps
/// per 256-column group. Prompt down projection uses four live warps per
/// 64-column group so narrow hidden states occupy the whole GPU; a grid-stride
/// loop still covers the complete admitted token envelope. Decode retains the
/// 16-warp GEMV path. Both launches share one model-wide BF16 workspace.
fn projection_tensor_core_mlp_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_tensor_core_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, " +
    "const __nv_bfloat16 *down_weight, __nv_bfloat16 *output, __nv_bfloat16 *workspace) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    "  if (blockDim.x != 512) return;\n" +
    "  if (token_count == 1) {\n" +
    "    const int intermediate = (int)blockIdx.x * 16 + warp;\n" +
    "    if (intermediate >= " +
    parameters.intermediate_width.to_string() +
    ") return;\n" +
    "    const unsigned long long weight_base = (unsigned long long)intermediate * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "    float gate = 0.0f, up = 0.0f;\n" +
    "    for (int inner = lane; inner < " +
    parameters.input_width.to_string() +
    "; inner += 32) {\n" +
    "      const float input_value = __bfloat162float(input[inner]);\n" +
    "      gate = __fadd_rn(gate, __fmul_rn(input_value, __bfloat162float(gate_weight[weight_base + inner])));\n" +
    "      up = __fadd_rn(up, __fmul_rn(input_value, __bfloat162float(up_weight[weight_base + inner])));\n" +
    "    }\n" +
    "    for (int delta = 16; delta > 0; delta >>= 1) { gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffU, gate, delta)); up = __fadd_rn(up, __shfl_down_sync(0xffffffffU, up, delta)); }\n" +
    "    if (lane == 0) workspace[intermediate] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);\n" +
    "    return;\n  }\n" +
    "  const int group_count = " +
    (parameters.intermediate_width / 256).to_string() +
    ";\n" +
    "  const int group_index = (int)blockIdx.x % group_count;\n" +
    "  const int row_start = ((int)blockIdx.x / group_count) * 16;\n" +
    "  if (row_start >= token_count) return;\n" +
    "  const int column = group_index * 256 + warp * 16;\n" +
    "  extern __shared__ unsigned char shared_bytes[];\n" +
    "  float *tile_output = reinterpret_cast<float *>(shared_bytes);\n" +
    "  float *warp_output = tile_output + warp * 256;\n" +
    "    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a;\n" +
    "    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> bg, bu;\n" +
    "    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cg, cu;\n" +
    "    nvcuda::wmma::fill_fragment(cg, 0.0f); nvcuda::wmma::fill_fragment(cu, 0.0f);\n" +
    "    for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; inner += 16) {\n" +
    "      nvcuda::wmma::load_matrix_sync(a, input + (unsigned long long)row_start * " +
    parameters.input_width.to_string() +
    "ULL + inner, " +
    parameters.input_width.to_string() +
    ");\n" +
    "      nvcuda::wmma::load_matrix_sync(bg, gate_weight + (unsigned long long)column * " +
    parameters.input_width.to_string() +
    "ULL + inner, " +
    parameters.input_width.to_string() +
    ");\n" +
    "      nvcuda::wmma::load_matrix_sync(bu, up_weight + (unsigned long long)column * " +
    parameters.input_width.to_string() +
    "ULL + inner, " +
    parameters.input_width.to_string() +
    ");\n" +
    "      nvcuda::wmma::mma_sync(cg, a, bg, cg); nvcuda::wmma::mma_sync(cu, a, bu, cu);\n" +
    "    }\n" +
    "    nvcuda::wmma::store_matrix_sync(warp_output, cg, 16, nvcuda::wmma::mem_row_major);\n" +
    "    __syncwarp();\n" +
    "    float gate_values[8];\n" +
    "    #pragma unroll\n" +
    "    for (int item = 0; item < 8; ++item) gate_values[item] = warp_output[lane + item * 32];\n" +
    "    __syncwarp();\n" +
    "    nvcuda::wmma::store_matrix_sync(warp_output, cu, 16, nvcuda::wmma::mem_row_major);\n" +
    "    __syncwarp();\n" +
    "    #pragma unroll\n" +
    "    for (int item = 0; item < 8; ++item) { const int element = lane + item * 32;\n" +
    "      const int tile_row = element >> 4, tile_column = element & 15;\n" +
    "      if (row_start + tile_row < token_count) { const float gate = gate_values[item], up = warp_output[element];\n" +
    "        workspace[(unsigned long long)(row_start + tile_row) * " +
    parameters.intermediate_width.to_string() +
    "ULL + column + tile_column] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up); } }\n" +
    "}\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "_down(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output, __nv_bfloat16 *workspace) {\n" +
    projection_count_guard(maximum_tokens) +
    "  if (blockDim.x != 512) return;\n" +
    "  const int warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;\n" +
    "  if (token_count == 1) {\n" +
    "    const int output_column = (int)blockIdx.x * 16 + warp;\n" +
    "    if (output_column >= " +
    parameters.output_width.to_string() +
    ") return;\n" +
    "    const unsigned long long weight_base = (unsigned long long)output_column * " +
    parameters.intermediate_width.to_string() +
    "ULL;\n" +
    "    float value = 0.0f;\n" +
    "    for (int inner = lane; inner < " +
    parameters.intermediate_width.to_string() +
    "; inner += 32) value = __fadd_rn(value, __fmul_rn(__bfloat162float(workspace[inner]), __bfloat162float(down_weight[weight_base + inner])));\n" +
    "    for (int delta = 16; delta > 0; delta >>= 1) value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, delta));\n" +
    "    if (lane == 0) output[output_column] = __float2bfloat16_rn(value);\n" +
    "    return;\n  }\n" +
    "  if (warp >= 4) return;\n" +
    "  const int group_count = " +
    (parameters.output_width / 64).to_string() +
    "; const int live_groups = ((token_count + 15) / 16) * group_count;\n" +
    "  for (int work_group = (int)blockIdx.x; work_group < live_groups; work_group += (int)gridDim.x) {\n" +
    "  const int group_index = work_group % group_count;\n" +
    "  const int row_start = (work_group / group_count) * 16;\n" +
    "  const int column = group_index * 64 + warp * 16;\n" +
    "  extern __shared__ unsigned char shared_bytes[]; float *warp_output = reinterpret_cast<float *>(shared_bytes) + warp * 256;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a; nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b; nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c; nvcuda::wmma::fill_fragment(c, 0.0f);\n" +
    "  for (int inner = 0; inner < " +
    parameters.intermediate_width.to_string() +
    "; inner += 16) { nvcuda::wmma::load_matrix_sync(a, workspace + (unsigned long long)row_start * " +
    parameters.intermediate_width.to_string() +
    "ULL + inner, " +
    parameters.intermediate_width.to_string() +
    "); nvcuda::wmma::load_matrix_sync(b, down_weight + (unsigned long long)column * " +
    parameters.intermediate_width.to_string() +
    "ULL + inner, " +
    parameters.intermediate_width.to_string() +
    "); nvcuda::wmma::mma_sync(c, a, b, c); }\n" +
    "  nvcuda::wmma::store_matrix_sync(warp_output, c, 16, nvcuda::wmma::mem_row_major); __syncwarp();\n  #pragma unroll\n" +
    "  for (int item = 0; item < 8; ++item) { const int element = lane + item * 32; const int tile_row = element >> 4, tile_column = element & 15; if (row_start + tile_row < token_count) output[(unsigned long long)(row_start + tile_row) * " +
    parameters.output_width.to_string() +
    "ULL + column + tile_column] = __float2bfloat16_rn(warp_output[element]); }\n  }\n}\n",
  )
}

///|
/// Computes the gate/up intermediate once per token and retains the exact
/// declared accumulation order for each dot product and the down projection.
/// One token owns one block; the shared F32 intermediate removes the previous
/// output-width-fold redundant gate/up work without changing the operation ABI.
fn projection_mlp_shared_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    projection_cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, " +
    "const __nv_bfloat16 *down_weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const int token = (int)blockIdx.x;\n" +
    "  if (token >= token_count) return;\n" +
    "  extern __shared__ float combined[];\n" +
    "  const unsigned long long input_base = (unsigned long long)token * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "  for (int intermediate = (int)threadIdx.x; intermediate < " +
    parameters.intermediate_width.to_string() +
    "; intermediate += (int)blockDim.x) {\n" +
    "    const unsigned long long projection_base = " +
    "(unsigned long long)intermediate * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    "    float gate = 0.0f;\n    float up = 0.0f;\n" +
    "    for (int inner = 0; inner < " +
    parameters.input_width.to_string() +
    "; ++inner) {\n" +
    "      const float input_value = __bfloat162float(input[input_base + inner]);\n" +
    "      gate = __fadd_rn(gate, __fmul_rn(input_value, " +
    "__bfloat162float(gate_weight[projection_base + inner])));\n" +
    "      up = __fadd_rn(up, __fmul_rn(input_value, " +
    "__bfloat162float(up_weight[projection_base + inner])));\n" +
    "    }\n" +
    "    const float silu = gate / __fadd_rn(1.0f, expf(-gate));\n" +
    "    combined[intermediate] = __fmul_rn(silu, up);\n" +
    "  }\n" +
    "  __syncthreads();\n" +
    "  for (int output_column = (int)threadIdx.x; output_column < " +
    parameters.output_width.to_string() +
    "; output_column += (int)blockDim.x) {\n" +
    "    const unsigned long long down_base = " +
    "(unsigned long long)output_column * " +
    parameters.intermediate_width.to_string() +
    "ULL;\n" +
    "    float output_accumulator = 0.0f;\n" +
    "    for (int intermediate = 0; intermediate < " +
    parameters.intermediate_width.to_string() +
    "; ++intermediate) {\n" +
    "      const float down = __bfloat162float(down_weight[down_base + intermediate]);\n" +
    "      output_accumulator = __fadd_rn(output_accumulator, " +
    "__fmul_rn(combined[intermediate], down));\n" +
    "    }\n" +
    "    output[(unsigned long long)token * " +
    parameters.output_width.to_string() +
    "ULL + output_column] = __float2bfloat16_rn(output_accumulator);\n" +
    "  }\n" +
    "}\n",
  )
}

///|
fn projection_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  maximum_rows : Int,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  embed_greedy_sampling : Bool,
  use_shared_mlp : Bool,
) -> Bytes raise ProjectionAotError {
  match parameters.family {
    QkvProjection =>
      if projection_tensor_core_eligible(parameters, dispatch) {
        projection_tensor_core_qkv_source(symbol, parameters, maximum_tokens)
      } else {
        projection_qkv_source(symbol, parameters, maximum_tokens)
      }
    DenseProjection =>
      if projection_tensor_core_eligible(parameters, dispatch) {
        projection_tensor_core_dense_source(
          symbol, parameters, maximum_tokens, maximum_rows,
        )
      } else {
        projection_dense_source(
          symbol, parameters, maximum_tokens, maximum_rows,
        )
      }
    LanguageModelHead => {
      let base = if projection_tensor_core_eligible(parameters, dispatch) {
        projection_tensor_core_dense_source(
          symbol, parameters, maximum_tokens, maximum_rows,
        )
      } else {
        projection_dense_source(
          symbol, parameters, maximum_tokens, maximum_rows,
        )
      }
      if !embed_greedy_sampling {
        return base
      }
      let source = @utf8.decode(base) catch { _ => raise Invalid(Source) }
      let reducer = @luna_cuda_sampling_aot.embedded_greedy_sampling_cuda_source(
        parameters.output_width,
        maximum_rows,
      ) catch {
        _ => raise Invalid(Source)
      }
      @utf8.encode(source + reducer)
    }
    GatedMlp =>
      if projection_tensor_core_mlp_eligible(parameters, dispatch) {
        projection_tensor_core_mlp_source(symbol, parameters, maximum_tokens)
      } else if use_shared_mlp {
        projection_mlp_shared_source(symbol, parameters, maximum_tokens)
      } else {
        projection_mlp_source(symbol, parameters, maximum_tokens)
      }
  }
}
