///|
/// Emits exact V1 head-reduction source. Layouts are row-major
/// `streams[rows,width,hidden]`, `function_weights[width,width*hidden]`,
/// `base[width]`, scalar `scale[1]`, and `output[rows,hidden]`. Thread zero
/// derives normalized sigmoid controls; all threads then reduce streams in
/// ascending order and round the output once to BF16-RNE.
pub fn head_reduce_v1_bf16_f32_source(
  stream_width~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  control_epsilon~ : Double,
  normalization_epsilon~ : Double,
  function_symbol~ : String,
) -> DecoderHyperConnectionHeadReduceCudaSource raise DecoderHyperConnectionCudaSourceError {
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
  // Fixed typed expressions and folds are total for the geometry validated
  // above. Keep the legacy public error variants for incompatible callers.
  let plan = try! @precision.LearnedStreamCollapsePrecision::new(
    stream_count=stream_width,
    hidden_width~,
    maximum_rows~,
    control_epsilon~,
    normalization_epsilon~,
  )
  fn expression(
    op : @precision.LearnedStreamExpression,
    bindings : ArrayView[String],
  ) -> String {
    try! @arithmetic.scalar_expression(plan.expression(op), bindings)
  }
  fn seed(op : @precision.LearnedStreamExpression) -> String {
    try! @arithmetic.scalar_expression(
      @precision.NumericExpression::new(
        [],
        Float32Bits(plan.fold(op).initial_bits),
      ),
      [],
    )
  }
  let control_literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    control_epsilon,
  )
  let normalization_literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    normalization_epsilon,
  )
  let width = (try! plan.fold(CollapseUpdate)).extent.to_string()
  let hidden = hidden_width.to_string()
  let flattened = (try! plan.fold(DotUpdate)).extent.to_string()
  let square_extent = (try! plan.fold(SquareUpdate)).extent.to_string()
  let square_seed = seed(SquareUpdate)
  let dot_seed = seed(DotUpdate)
  let collapse_seed = seed(CollapseUpdate)
  let positive = expression(SigmoidNonnegative, ["value"])
  let negative = expression(SigmoidNegativeExponential, ["exponential"])
  let exponential = expression(Exponential, ["value"])
  let square = expression(SquareUpdate, ["square_sum", "value"])
  let mean = expression(Mean, ["square_sum", flattened + ".0f"])
  let inverse = expression(Inverse, ["mean", normalization_literal])
  let dot = expression(DotUpdate, [
    "dot", "streams[input_base + (unsigned long long)column]", "weight",
  ])
  let mix = expression(Mix, ["dot", "inverse"])
  let affine = expression(Affine, ["mix", "scalar", "bias"])
  let collapse = expression(CollapseUpdate, ["sum", "controls[stream]", "value"])
  let publish = expression(Publish, ["isfinite(sum) ? sum : NAN"])
  // Private lowering helpers belong to their entry-point namespace so two
  // independently lowered heads can inhabit one composed AOT module.
  let sigmoid_symbol = function_symbol + "_sigmoid"
  let control = expression(Control, [
    sigmoid_symbol + "(logit)",
    control_literal,
  ])
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "__device__ __forceinline__ float " +
    sigmoid_symbol +
    "(float value) {\n" +
    "  if (value >= 0.0f) return " +
    positive +
    ";\n" +
    "  const float exponential = " +
    exponential +
    "; return " +
    negative +
    ";\n}\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *streams, const float *function_weights, const float *base, const float *scale, __nv_bfloat16 *output) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (row >= live_rows) return;\n" +
    "  __shared__ float controls[" +
    width +
    "]; __shared__ int32_t valid;\n" +
    "  const unsigned long long input_base = (unsigned long long)row * " +
    flattened +
    "ULL;\n" +
    "  if (threadIdx.x == 0) { valid = 1; float square_sum = " +
    square_seed +
    "; for (int32_t column = 0; column < " +
    square_extent +
    "; ++column) { const float value = __bfloat162float(streams[input_base + (unsigned long long)column]); if (!isfinite(value)) { valid = 0; break; } square_sum = " +
    square +
    "; } if (valid) { const float mean = " +
    mean +
    "; const float inverse = " +
    inverse +
    "; const float scalar = scale[0]; if (!isfinite(inverse) || !isfinite(scalar)) valid = 0; for (int32_t stream = 0; stream < " +
    width +
    " && valid; ++stream) { float dot = " +
    dot_seed +
    "; const unsigned long long weight_base = (unsigned long long)stream * " +
    flattened +
    "ULL; for (int32_t column = 0; column < " +
    flattened +
    "; ++column) { const float weight = function_weights[weight_base + (unsigned long long)column]; if (!isfinite(weight)) { valid = 0; break; } dot = " +
    dot +
    "; } const float bias = base[stream]; if (!isfinite(bias)) { valid = 0; break; } const float mix = " +
    mix +
    "; const float logit = " +
    affine +
    "; controls[stream] = " +
    control +
    "; if (!isfinite(controls[stream])) valid = 0; } } }\n" +
    "  __syncthreads();\n" +
    "  const unsigned long long output_base = (unsigned long long)row * " +
    hidden +
    "ULL;\n" +
    "  if (!valid) { for (int32_t column = (int32_t)threadIdx.x; column < " +
    hidden +
    "; column += (int32_t)blockDim.x) output[output_base + (unsigned long long)column] = __float2bfloat16_rn(NAN); return; }\n" +
    "  for (int32_t column = (int32_t)threadIdx.x; column < " +
    hidden +
    "; column += (int32_t)blockDim.x) { float sum = " +
    collapse_seed +
    "; for (int32_t stream = 0; stream < " +
    width +
    "; ++stream) { const float value = __bfloat162float(streams[input_base + (unsigned long long)stream * " +
    hidden +
    "ULL + (unsigned long long)column]); sum = " +
    collapse +
    "; } output[output_base + (unsigned long long)column] = " +
    publish +
    "; }\n" +
    "}\n",
  )
  {
    operation: HeadReduceV1,
    numerical_contract: OrderedF32HeadControlReductionBf16RoundV1,
    stream_width,
    hidden_width,
    maximum_rows,
    control_epsilon,
    normalization_epsilon,
    function_symbol,
    source,
  }
}
