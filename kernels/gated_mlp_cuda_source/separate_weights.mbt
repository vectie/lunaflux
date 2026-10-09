///|
/// Renders bias-free BF16 SwiGLU over separate row-major gate/up/down weights.
/// Gate and up are `[intermediate,hidden]`; down is `[hidden,intermediate]`.
pub fn separate_gate_up_swiglu_dense_bf16_source(
  hidden_size~ : Int,
  intermediate_size~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> SeparateWeightGatedMlpCudaSource raise GatedMlpCudaSourceError {
  if hidden_size <= 0 || intermediate_size <= 0 {
    raise InvalidWidth
  }
  if maximum_rows <= 0 {
    raise InvalidRowCount
  }
  if hidden_size > 1073741823 ||
    intermediate_size > 1073741823 ||
    maximum_rows.to_int64() > 2147483647L / hidden_size.to_int64() ||
    hidden_size.to_int64() > 9223372036854775807L / intermediate_size.to_int64() {
    raise DimensionOverflow
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output) {\n" +
    "  if (input == output) return; const int32_t live_rows = counts[3]; if (live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; const unsigned long long live_elements = (unsigned long long)live_rows * " +
    hidden_size.to_string() +
    "ULL; if (i >= live_elements) return;\n" +
    "  const unsigned long long row = i / " +
    hidden_size.to_string() +
    "ULL; const int out = (int)(i % " +
    hidden_size.to_string() +
    "ULL); const unsigned long long input_base = row * " +
    hidden_size.to_string() +
    "ULL; float down_sum = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    intermediate_size.to_string() +
    "; inner += 1) { float gate_sum = 0.0f; float up_sum = 0.0f; const unsigned long long weight_base = (unsigned long long)inner * " +
    hidden_size.to_string() +
    "ULL;\n" +
    "    for (int column = 0; column < " +
    hidden_size.to_string() +
    "; column += 1) { const float x = __bfloat162float(input[input_base + column]); gate_sum = __fadd_rn(gate_sum, __fmul_rn(x, __bfloat162float(gate_weight[weight_base + column]))); up_sum = __fadd_rn(up_sum, __fmul_rn(x, __bfloat162float(up_weight[weight_base + column]))); }\n" +
    "    const __nv_bfloat16 gate = __float2bfloat16_rn(gate_sum); const __nv_bfloat16 up = __float2bfloat16_rn(up_sum); const float gate_f32 = __bfloat162float(gate); const __nv_bfloat16 activated = __float2bfloat16_rn(__fdiv_rn(gate_f32, __fadd_rn(1.0f, expf(-gate_f32)))); const __nv_bfloat16 product = __float2bfloat16_rn(__fmul_rn(__bfloat162float(activated), __bfloat162float(up))); down_sum = __fadd_rn(down_sum, __fmul_rn(__bfloat162float(product), __bfloat162float(down_weight[(unsigned long long)out * " +
    intermediate_size.to_string() +
    "ULL + inner]))); }\n" +
    "  output[i] = __float2bfloat16_rn(down_sum);\n}\n",
  )
  {
    numerical_contract: OrderedF32AccumulationBf16SwiGluDownBf16V1,
    hidden_size,
    intermediate_size,
    maximum_rows,
    function_symbol,
    source,
  }
}

///|
pub fn SeparateWeightGatedMlpCudaSource::numerical_contract(
  self : Self,
) -> GatedMlpNumericalContract {
  self.numerical_contract
}

///|
pub fn SeparateWeightGatedMlpCudaSource::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn SeparateWeightGatedMlpCudaSource::intermediate_size(self : Self) -> Int {
  self.intermediate_size
}

///|
pub fn SeparateWeightGatedMlpCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn SeparateWeightGatedMlpCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn SeparateWeightGatedMlpCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
