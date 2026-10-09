///|
fn valid_decay_control_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn safe_lower_bound_sigmoid_decay_f32_source(
  head_count~ : Int,
  head_dimension~ : Int,
  lower_bound~ : Double,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecayControlCudaSource raise DecayControlCudaSourceError {
  if head_count <= 0 || head_count > 65536 {
    raise InvalidHeadCount
  }
  if head_dimension <= 0 || head_dimension > 65536 {
    raise InvalidHeadDimension
  }
  if lower_bound.is_nan() || lower_bound.is_inf() || lower_bound >= 0.0 {
    raise InvalidLowerBound
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_decay_control_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let row_width = head_count.to_int64() * head_dimension.to_int64()
  if row_width <= 0L ||
    row_width > 2147483647L ||
    row_width * maximum_rows.to_int64() > 9223372036854775807L {
    raise DimensionOverflow
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *raw_forget, const float *dt_bias, const float *a_log, float *log_decay) {\n" +
    "  const int row_count = counts[3]; if (row_count <= 0 || row_count > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const unsigned long long row_width = " +
    row_width.to_string() +
    "ULL; const unsigned long long live = (unsigned long long)row_count * row_width; if (index >= live) return;\n" +
    "  const unsigned long long component = index % row_width; const unsigned long long head = component / " +
    head_dimension.to_string() +
    "ULL;\n" +
    "  const float biased_logit = __fadd_rn(__bfloat162float(raw_forget[index]), dt_bias[component]);\n" +
    "  const float decay_rate = expf(a_log[head]); const float scaled = __fmul_rn(decay_rate, biased_logit);\n" +
    "  const float sigmoid = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-scaled)));\n" +
    "  log_decay[index] = __fmul_rn(" +
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(lower_bound) +
    ", sigmoid);\n}\n",
  )
  {
    operation: SafeLowerBoundSigmoidDecayF32,
    numerical_contract: Bf16LogitF32BiasExpRateOrderedSigmoidF32OutputV1,
    head_count,
    head_dimension,
    lower_bound,
    maximum_rows,
    function_symbol,
    source,
  }
}
