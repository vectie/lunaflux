///|
fn two_stage_product(left : Int64, right : Int64) -> Int64? {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    None
  } else {
    Some(left * right)
  }
}

///|
fn valid_two_stage_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Serial correctness renderer for two bias-free dense projections with an
/// explicit BF16 intermediate round. All matrices are output-major row-major.
pub fn bias_free_bf16_low_rank_projection_source(
  input_width~ : Int,
  intermediate_width~ : Int,
  output_width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> TwoStageProjectionCudaSource raise TwoStageProjectionCudaSourceError {
  if input_width <= 0 {
    raise InvalidInputWidth
  }
  if intermediate_width <= 0 {
    raise InvalidIntermediateWidth
  }
  if output_width <= 0 {
    raise InvalidOutputWidth
  }
  if maximum_rows <= 0 || maximum_rows > 2147483647 {
    raise InvalidRowCount
  }
  guard two_stage_product(maximum_rows.to_int64(), input_width.to_int64())
    is Some(_) &&
    two_stage_product(maximum_rows.to_int64(), intermediate_width.to_int64())
    is Some(_) &&
    two_stage_product(maximum_rows.to_int64(), output_width.to_int64())
    is Some(_) &&
    two_stage_product(intermediate_width.to_int64(), input_width.to_int64())
    is Some(_) &&
    two_stage_product(output_width.to_int64(), intermediate_width.to_int64())
    is Some(_) else {
    raise DimensionOverflow
  }
  if !valid_two_stage_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *first_weight, const __nv_bfloat16 *second_weight, __nv_bfloat16 *intermediate, __nv_bfloat16 *output) {\n" +
    "  if (blockIdx.x != 0 || threadIdx.x != 0) return; const int32_t live_rows = counts[3];\n" +
    "  if (counts[0] < 0 || counts[1] < 0 || counts[2] <= 0 || counts[0] > counts[2] || counts[1] != counts[2] - counts[0] || live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  for (int32_t row = 0; row < live_rows; row += 1) { const unsigned long long input_base = (unsigned long long)row * " +
    input_width.to_string() +
    "ULL; const unsigned long long intermediate_base = (unsigned long long)row * " +
    intermediate_width.to_string() +
    "ULL; const unsigned long long output_base = (unsigned long long)row * " +
    output_width.to_string() +
    "ULL;\n" +
    "    for (int32_t out = 0; out < " +
    intermediate_width.to_string() +
    "; out += 1) { float sum = 0.0f; const unsigned long long weight_base = (unsigned long long)out * " +
    input_width.to_string() +
    "ULL; for (int32_t in = 0; in < " +
    input_width.to_string() +
    "; in += 1) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(input[input_base + in]), __bfloat162float(first_weight[weight_base + in]))); intermediate[intermediate_base + out] = __float2bfloat16_rn(sum); }\n" +
    "    for (int32_t out = 0; out < " +
    output_width.to_string() +
    "; out += 1) { float sum = 0.0f; const unsigned long long weight_base = (unsigned long long)out * " +
    intermediate_width.to_string() +
    "ULL; for (int32_t in = 0; in < " +
    intermediate_width.to_string() +
    "; in += 1) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(intermediate[intermediate_base + in]), __bfloat162float(second_weight[weight_base + in]))); output[output_base + out] = __float2bfloat16_rn(sum); }\n" +
    "  }\n}\n",
  )
  {
    operation: BiasFreeBf16LowRankProjection,
    numerical_contract: OrderedFloat32AccumulationBf16IntermediateAndOutputV1,
    input_width,
    intermediate_width,
    output_width,
    maximum_rows,
    function_symbol,
    source,
  }
}

///|
pub fn TwoStageProjectionCudaSource::operation(
  self : Self,
) -> TwoStageProjectionOperation {
  self.operation
}

///|
pub fn TwoStageProjectionCudaSource::numerical_contract(
  self : Self,
) -> TwoStageProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn TwoStageProjectionCudaSource::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn TwoStageProjectionCudaSource::intermediate_width(self : Self) -> Int {
  self.intermediate_width
}

///|
pub fn TwoStageProjectionCudaSource::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn TwoStageProjectionCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn TwoStageProjectionCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn TwoStageProjectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
