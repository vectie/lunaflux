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
fn projection_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  maximum_rows : Int,
  embed_greedy_sampling : Bool,
) -> Bytes raise ProjectionAotError {
  match parameters.family {
    QkvProjection => projection_qkv_source(symbol, parameters, maximum_tokens)
    DenseProjection =>
      projection_dense_source(symbol, parameters, maximum_tokens)
    LanguageModelHead => {
      let base = projection_dense_source(symbol, parameters, maximum_tokens)
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
    GatedMlp => projection_mlp_source(symbol, parameters, maximum_tokens)
  }
}
