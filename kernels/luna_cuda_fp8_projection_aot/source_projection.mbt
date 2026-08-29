///|
fn fp8_dense_source(
  symbol : String,
  parameters : Fp8ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    fp8_cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const unsigned char *weight, const float *weight_scale, " +
    "__nv_bfloat16 *output, float *activation_scale_workspace) {\n" +
    fp8_count_guard(maximum_tokens) +
    fp8_common_index(parameters) +
    "  const unsigned long long weight_base = (unsigned long long)column * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    fp8_dot_loop(parameters.input_width),
  )
}

///|
fn fp8_qkv_source(
  symbol : String,
  parameters : Fp8ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes {
  @utf8.encode(
    fp8_cuda_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const unsigned char *query_weight, const float *query_scale, " +
    "const unsigned char *key_weight, const float *key_scale, " +
    "const unsigned char *value_weight, const float *value_scale, " +
    "__nv_bfloat16 *output, float *activation_scale_workspace) {\n" +
    fp8_count_guard(maximum_tokens) +
    fp8_common_index(parameters) +
    "  const unsigned char *weight = query_weight; const float *weight_scale = query_scale; int local_column = column;\n" +
    "  if (column >= " +
    parameters.input_width.to_string() +
    ") {\n" +
    "    local_column = column - " +
    parameters.input_width.to_string() +
    ";\n" +
    "    if (local_column >= " +
    parameters.intermediate_width.to_string() +
    ") { weight = value_weight; weight_scale = value_scale; local_column -= " +
    parameters.intermediate_width.to_string() +
    "; }\n" +
    "    else { weight = key_weight; weight_scale = key_scale; }\n  }\n" +
    "  const unsigned long long weight_base = (unsigned long long)local_column * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    fp8_dot_loop(parameters.input_width),
  )
}
