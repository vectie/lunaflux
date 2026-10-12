///|
/// Emits the first exact staged mHC boundary. Streams are BF16; the immutable
/// control law selects BF16 weights with normalized inputs, or F32 weights with
/// projection before normalization. Base/post/matrix controls are F32.
/// Collapsed output rounds once to BF16 after the ordered stream reduction.
pub fn staged_function_and_collapse_bf16_f32_source(
  stream_width~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  control_epsilon~ : Double,
  normalization_epsilon~ : Double,
  function_symbol~ : String,
  control_law? : @precision.HyperConnectionControlLaw = NormalizedBf16Projection,
  control_square_law? : @precision.SquareReductionLaw = OrderedFloat32,
  control_dot_law? : @precision.DotReductionLaw = OrderedFloat32,
  control_vector_affine_law? : @precision.Float32AffineLaw = SeparateMultiplyAdd,
  control_sigmoid_law? : @precision.SigmoidEvaluationLaw = StableBranches,
  control_root_law? : @precision.ReciprocalRootLaw = DivideRoundedSqrtFloat32,
) -> DecoderHyperConnectionFunctionCudaSource raise DecoderHyperConnectionCudaSourceError {
  if stream_width <= 1 || stream_width > 16 {
    raise InvalidStreamWidth
  }
  if hidden_width <= 0 || hidden_width > 65536 {
    raise InvalidHiddenWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if control_epsilon.is_nan() ||
    control_epsilon.is_inf() ||
    control_epsilon <= 0.0 ||
    normalization_epsilon.is_nan() ||
    normalization_epsilon.is_inf() ||
    normalization_epsilon <= 0.0 {
    raise InvalidEpsilon
  }
  if !valid_hyper_connection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let width = stream_width.to_string()
  let hidden = hidden_width.to_string()
  let flattened = (stream_width * hidden_width).to_string()
  let mix = ((2 + stream_width) * stream_width).to_string()
  let control = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    control_epsilon,
  )
  let normalization = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    normalization_epsilon,
  )
  let (weight_type, projection, operand_name, publication, numerical_contract) = match
    control_law {
    NormalizedBf16Projection =>
      (
        "__nv_bfloat16",
        "const float normalized = __fmul_rn(__bfloat162float(streams[input_base + (unsigned long long)column]), inverse); const float weight = __bfloat162float(function_weights[weight_base + (unsigned long long)column]);",
        "normalized",
        "mixes[output] = dot;",
        OrderedF32FunctionBf16CollapseV1,
      )
    F32ProjectionThenNormalize =>
      (
        "float",
        "const float operand = __bfloat162float(streams[input_base + (unsigned long long)column]); const float weight = function_weights[weight_base + (unsigned long long)column];",
        "operand",
        "mixes[output] = __fmul_rn(dot, inverse);",
        OrderedF32ProjectionThenNormalizeBf16CollapseV1,
      )
  }
  let (square_init, square_update, square_finish) = control_square_source(
    control_square_law,
  )
  let (dot_init, dot_update, dot_finish) = control_dot_source(
    control_dot_law,
    operand_name,
    stream_width * hidden_width,
  )
  let numerical_contract = if control_square_law is OrderedFloat32 &&
    control_dot_law is OrderedFloat32 &&
    control_vector_affine_law is SeparateMultiplyAdd &&
    control_sigmoid_law is StableBranches &&
    control_root_law is DivideRoundedSqrtFloat32 {
    numerical_contract
  } else {
    RefinedControlFunctionBf16CollapseV1(
      control=control_law,
      square=control_square_law,
      dot=control_dot_law,
      vector_affine=control_vector_affine_law,
      sigmoid=control_sigmoid_law,
      root=control_root_law,
    )
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "__device__ __forceinline__ float lf_hc_stage_sigmoid(float x) { " +
    control_sigmoid_source(control_sigmoid_law) +
    " }\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *streams, const " +
    weight_type +
    " *function_weights, const float *base, const float *scale, float *post, float *initial_combination, __nv_bfloat16 *collapsed) {\n" +
    "  const int32_t live_rows = counts[3]; if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return; const int32_t row = (int32_t)blockIdx.x; if (row >= live_rows || threadIdx.x != 0) return;\n" +
    "  const void *buffers[8] = {(const void*)counts, (const void*)streams, (const void*)function_weights, (const void*)base, (const void*)scale, (const void*)post, (const void*)initial_combination, (const void*)collapsed}; for (int32_t left = 0; left < 8; ++left) for (int32_t right = left + 1; right < 8; ++right) if (buffers[left] == buffers[right]) return;\n" +
    "  const unsigned long long input_base = (unsigned long long)row * " +
    flattened +
    "ULL; " +
    square_init +
    " for (int32_t column = 0; column < " +
    flattened +
    "; ++column) { const float x = __bfloat162float(streams[input_base + (unsigned long long)column]); if (!isfinite(x)) return; " +
    square_update +
    " }\n" +
    "  " +
    (if square_finish.is_empty() { "" } else { square_finish + " " }) +
    "const float inverse = " +
    control_root_source(
      "__fadd_rn(__fdiv_rn(square_sum, " +
      flattened +
      ".0f), " +
      normalization +
      ")",
      control_root_law,
    ) +
    "; if (!isfinite(inverse)) return; float mixes[" +
    mix +
    "];\n" +
    "  for (int32_t output = 0; output < " +
    mix +
    "; ++output) { " +
    dot_init +
    " const unsigned long long weight_base = (unsigned long long)output * " +
    flattened +
    "ULL; for (int32_t column = 0; column < " +
    flattened +
    "; ++column) { " +
    projection +
    " " +
    dot_update +
    " } " +
    (if dot_finish.is_empty() { "" } else { dot_finish + " " }) +
    publication +
    " }\n" +
    "  for (int32_t i = 0; i < " +
    mix +
    "; ++i) if (!isfinite(base[i])) return; for (int32_t i = 0; i < 3; ++i) if (!isfinite(scale[i])) return;\n" +
    "  float pre[" +
    width +
    "]; const unsigned long long vector_base = (unsigned long long)row * " +
    width +
    "ULL; for (int32_t stream = 0; stream < " +
    width +
    "; ++stream) { const float pre_logit = " +
    control_vector_affine_source("stream", "0", control_vector_affine_law) +
    "; pre[stream] = __fadd_rn(lf_hc_stage_sigmoid(pre_logit), " +
    control +
    "); const int32_t p = " +
    width +
    " + stream; const float post_logit = " +
    control_vector_affine_source("p", "1", control_vector_affine_law) +
    "; post[vector_base + (unsigned long long)stream] = __fmul_rn(2.0f, lf_hc_stage_sigmoid(post_logit)); }\n" +
    "  const unsigned long long matrix_base = (unsigned long long)row * " +
    width +
    "ULL * " +
    width +
    "ULL; for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) { int32_t control_index = 2 * " +
    width +
    " + destination * " +
    width +
    "; float maximum = __fadd_rn(__fmul_rn(mixes[control_index], scale[2]), base[control_index]); for (int32_t source = 1; source < " +
    width +
    "; ++source) { control_index = 2 * " +
    width +
    " + destination * " +
    width +
    " + source; const float value = __fadd_rn(__fmul_rn(mixes[control_index], scale[2]), base[control_index]); if (value > maximum) maximum = value; } float sum = 0.0f; for (int32_t source = 0; source < " +
    width +
    "; ++source) { control_index = 2 * " +
    width +
    " + destination * " +
    width +
    " + source; const float value = expf(__fadd_rn(__fadd_rn(__fmul_rn(mixes[control_index], scale[2]), base[control_index]), -maximum)); const unsigned long long index = matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; initial_combination[index] = value; sum = __fadd_rn(sum, value); } for (int32_t source = 0; source < " +
    width +
    "; ++source) { const unsigned long long index = matrix_base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; initial_combination[index] = __fadd_rn(__fdiv_rn(initial_combination[index], sum), " +
    control +
    "); } }\n" +
    "  const unsigned long long collapsed_base = (unsigned long long)row * " +
    hidden +
    "ULL; for (int32_t column = 0; column < " +
    hidden +
    "; ++column) { float sum = 0.0f; for (int32_t stream = 0; stream < " +
    width +
    "; ++stream) sum = __fadd_rn(sum, __fmul_rn(pre[stream], __bfloat162float(streams[input_base + (unsigned long long)stream * " +
    hidden +
    "ULL + (unsigned long long)column]))); if (isfinite(sum)) collapsed[collapsed_base + (unsigned long long)column] = __float2bfloat16_rn(sum); }\n}\n",
  )
  {
    operation: StagedFunctionAndCollapseV1,
    numerical_contract,
    stream_width,
    hidden_width,
    maximum_rows,
    control_epsilon,
    normalization_epsilon,
    function_symbol,
    source,
  }
}

///|
/// Emits the second exact staged mHC boundary. It performs the initial column
/// normalization, followed by `iterations - 1` row/column normalizations.
pub fn staged_positive_sinkhorn_f32_source(
  stream_width~ : Int,
  maximum_rows~ : Int,
  sinkhorn_iterations~ : Int,
  control_epsilon~ : Double,
  function_symbol~ : String,
) -> DecoderHyperConnectionSinkhornCudaSource raise DecoderHyperConnectionCudaSourceError {
  if stream_width <= 1 || stream_width > 16 {
    raise InvalidStreamWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if sinkhorn_iterations <= 0 || sinkhorn_iterations > 64 {
    raise InvalidIterationCount
  }
  if control_epsilon.is_nan() ||
    control_epsilon.is_inf() ||
    control_epsilon <= 0.0 {
    raise InvalidEpsilon
  }
  if !valid_hyper_connection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let width = stream_width.to_string()
  let epsilon = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    control_epsilon,
  )
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\nextern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const float *initial_combination, float *combination) {\n" +
    "  const int32_t live_rows = counts[3]; if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return; const int32_t row = (int32_t)blockIdx.x; if (row >= live_rows || threadIdx.x != 0 || (const void*)counts == (const void*)initial_combination || (const void*)counts == (const void*)combination || initial_combination == combination) return; const unsigned long long base = (unsigned long long)row * " +
    width +
    "ULL * " +
    width +
    "ULL;\n" +
    "  for (int32_t i = 0; i < " +
    width +
    " * " +
    width +
    "; ++i) { const float value = initial_combination[base + (unsigned long long)i]; if (!isfinite(value) || value <= 0.0f) return; combination[base + (unsigned long long)i] = value; }\n" +
    "  for (int32_t source = 0; source < " +
    width +
    "; ++source) { float sum = 0.0f; for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) sum = __fadd_rn(sum, combination[base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source]); const float denominator = __fadd_rn(sum, " +
    epsilon +
    "); for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) { const unsigned long long index = base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; combination[index] = __fdiv_rn(combination[index], denominator); } }\n" +
    "  for (int32_t iteration = 1; iteration < " +
    sinkhorn_iterations.to_string() +
    "; ++iteration) { for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) { float sum = 0.0f; for (int32_t source = 0; source < " +
    width +
    "; ++source) sum = __fadd_rn(sum, combination[base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source]); const float denominator = __fadd_rn(sum, " +
    epsilon +
    "); for (int32_t source = 0; source < " +
    width +
    "; ++source) { const unsigned long long index = base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; combination[index] = __fdiv_rn(combination[index], denominator); } } for (int32_t source = 0; source < " +
    width +
    "; ++source) { float sum = 0.0f; for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) sum = __fadd_rn(sum, combination[base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source]); const float denominator = __fadd_rn(sum, " +
    epsilon +
    "); for (int32_t destination = 0; destination < " +
    width +
    "; ++destination) { const unsigned long long index = base + (unsigned long long)destination * " +
    width +
    "ULL + (unsigned long long)source; combination[index] = __fdiv_rn(combination[index], denominator); } } }\n}\n",
  )
  {
    operation: StagedSinkhornV1,
    numerical_contract: OrderedF32PositiveSinkhornV1,
    stream_width,
    maximum_rows,
    sinkhorn_iterations,
    control_epsilon,
    function_symbol,
    source,
  }
}
