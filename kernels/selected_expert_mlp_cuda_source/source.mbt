///|
fn valid_selected_expert_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn selected_expert_swiglu_bf16_source(
  hidden_size~ : Int,
  intermediate_size~ : Int,
  routed_expert_count~ : Int,
  experts_per_token~ : Int,
  activation~ : SelectedExpertActivation,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> SelectedExpertMlpCudaSource raise SelectedExpertMlpCudaSourceError {
  if hidden_size <= 0 ||
    intermediate_size <= 0 ||
    routed_expert_count <= 0 ||
    experts_per_token <= 0 ||
    experts_per_token > routed_expert_count ||
    experts_per_token > 256 {
    raise InvalidGeometry
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  let clamp = match activation {
    StandardSwiGlu => None
    BoundedClampedSwiGlu(limit) => {
      if limit.is_nan() || limit.is_inf() || limit <= 0.0 {
        raise InvalidActivation
      }
      Some(@luna_cuda_pointwise_aot.canonical_cuda_float_literal(limit))
    }
  }
  if !valid_selected_expert_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let rows = maximum_rows.to_int64()
  let hidden = hidden_size.to_int64()
  let intermediate = intermediate_size.to_int64()
  let experts = routed_expert_count.to_int64()
  if hidden > 9223372036854775552L / rows ||
    intermediate > 4611686018427387903L / hidden ||
    experts > 4611686018427387903L / (intermediate * hidden) {
    raise DimensionOverflow
  }
  let cells = rows * hidden
  let expert_matrix = experts * intermediate * hidden
  if cells <= 0L ||
    cells > 9223372036854775552L ||
    expert_matrix <= 0L ||
    expert_matrix > 4611686018427387903L {
    raise DimensionOverflow
  }
  let clamp_code = match clamp {
    None =>
      "const __nv_bfloat16 gate = gate_raw; const __nv_bfloat16 up = up_raw;"
    Some(limit) =>
      "const float gate_clamped = fminf(__bfloat162float(gate_raw), " +
      limit +
      "); const float up_clamped = fmaxf(-" +
      limit +
      ", fminf(__bfloat162float(up_raw), " +
      limit +
      ")); const __nv_bfloat16 gate = __float2bfloat16_rn(gate_clamped); const __nv_bfloat16 up = __float2bfloat16_rn(up_clamped);"
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const int32_t *selected_indices, const float *selected_weights, const __nv_bfloat16 *gate_weights, const __nv_bfloat16 *up_weights, const __nv_bfloat16 *down_weights, __nv_bfloat16 *output) {\n" +
    "  if (input == output) return; const int32_t live_rows = counts[3]; if (live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long cell = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; const unsigned long long live = (unsigned long long)live_rows * " +
    hidden_size.to_string() +
    "ULL; if (cell >= live) return; const unsigned long long row = cell / " +
    hidden_size.to_string() +
    "ULL; const int32_t out = (int32_t)(cell % " +
    hidden_size.to_string() +
    "ULL); const unsigned long long selected_base = row * " +
    experts_per_token.to_string() +
    "ULL;\n" +
    "  bool valid = true; for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) { const int32_t expert = selected_indices[selected_base + slot]; if (expert < 0 || expert >= " +
    routed_expert_count.to_string() +
    " || !isfinite(selected_weights[selected_base + slot])) valid = false; for (int32_t previous = 0; previous < slot; ++previous) if (selected_indices[selected_base + previous] == expert) valid = false; } if (!valid) { output[cell] = __float2bfloat16_rn(0.0f); return; }\n" +
    "  __nv_bfloat16 routed_sum = __float2bfloat16_rn(0.0f); const unsigned long long input_base = row * " +
    hidden_size.to_string() +
    "ULL; const unsigned long long expert_matrix_stride = (unsigned long long)" +
    intermediate_size.to_string() +
    " * " +
    hidden_size.to_string() +
    "ULL; const unsigned long long down_stride = (unsigned long long)" +
    hidden_size.to_string() +
    " * " +
    intermediate_size.to_string() +
    "ULL;\n" +
    "  for (int32_t expert = 0; expert < " +
    routed_expert_count.to_string() +
    "; ++expert) { int32_t selected_slot = -1; for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) if (selected_indices[selected_base + slot] == expert) selected_slot = slot; if (selected_slot < 0) continue; float down_sum = 0.0f;\n" +
    "    for (int32_t inner = 0; inner < " +
    intermediate_size.to_string() +
    "; ++inner) { float gate_sum = 0.0f; float up_sum = 0.0f; const unsigned long long weight_base = (unsigned long long)expert * expert_matrix_stride + (unsigned long long)inner * " +
    hidden_size.to_string() +
    "ULL; for (int32_t column = 0; column < " +
    hidden_size.to_string() +
    "; ++column) { const float x = __bfloat162float(input[input_base + column]); gate_sum = __fadd_rn(gate_sum, __fmul_rn(x, __bfloat162float(gate_weights[weight_base + column]))); up_sum = __fadd_rn(up_sum, __fmul_rn(x, __bfloat162float(up_weights[weight_base + column]))); } const __nv_bfloat16 gate_raw = __float2bfloat16_rn(gate_sum); const __nv_bfloat16 up_raw = __float2bfloat16_rn(up_sum); " +
    clamp_code +
    " const float gate_f32 = __bfloat162float(gate); const __nv_bfloat16 activated = __float2bfloat16_rn(__fdiv_rn(gate_f32, __fadd_rn(1.0f, expf(-gate_f32)))); const __nv_bfloat16 product = __float2bfloat16_rn(__fmul_rn(__bfloat162float(activated), __bfloat162float(up))); const unsigned long long down_index = (unsigned long long)expert * down_stride + (unsigned long long)out * " +
    intermediate_size.to_string() +
    "ULL + inner; down_sum = __fadd_rn(down_sum, __fmul_rn(__bfloat162float(product), __bfloat162float(down_weights[down_index]))); }\n" +
    "    const __nv_bfloat16 expert_output = __float2bfloat16_rn(down_sum); const __nv_bfloat16 weighted = __float2bfloat16_rn(__fmul_rn(__bfloat162float(expert_output), selected_weights[selected_base + selected_slot])); routed_sum = __float2bfloat16_rn(__fadd_rn(__bfloat162float(routed_sum), __bfloat162float(weighted))); } output[cell] = routed_sum;\n}\n",
  )
  {
    numerical_contract: ExpertAscendingBf16SwiGluWeightedAccumulateV1,
    activation,
    hidden_size,
    intermediate_size,
    routed_expert_count,
    experts_per_token,
    maximum_rows,
    function_symbol,
    source,
  }
}
