///|
fn valid_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 128 &&
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Renders bias-free BF16 packed-up/gate SwiGLU followed by a BF16 down
/// projection. Packed weight is row-major `[2*intermediate,hidden]`, first up
/// then gate; down weight is row-major `[hidden,intermediate]`.
pub fn packed_swiglu_dense_bf16_rows_source(
  rows~ : Int,
  hidden_size~ : Int,
  intermediate_size~ : Int,
  function_symbol~ : String,
) -> GatedMlpCudaSource raise GatedMlpCudaSourceError {
  if rows <= 0 {
    raise InvalidRowCount
  }
  if hidden_size <= 0 || intermediate_size <= 0 {
    raise InvalidWidth
  }
  if intermediate_size > 1073741823 ||
    rows.to_int64() > 2147483647L / hidden_size.to_int64() {
    raise DimensionOverflow
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let elements = rows.to_int64() * hidden_size.to_int64()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const __nv_bfloat16 *input, const __nv_bfloat16 *packed_up_gate_weight, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    elements.to_string() +
    "ULL) return;\n" +
    "  const unsigned long long row = i / " +
    hidden_size.to_string() +
    "ULL; const int out = (int)(i % " +
    hidden_size.to_string() +
    "ULL); float down_sum = 0.0f;\n" +
    "  for (int inner = 0; inner < " +
    intermediate_size.to_string() +
    "; inner += 1) { float up_sum = 0.0f; float gate_sum = 0.0f; const unsigned long long up_base = (unsigned long long)inner * " +
    hidden_size.to_string() +
    "ULL; const unsigned long long gate_base = (unsigned long long)(inner + " +
    intermediate_size.to_string() +
    ") * " +
    hidden_size.to_string() +
    "ULL; const unsigned long long input_base = row * " +
    hidden_size.to_string() +
    "ULL;\n" +
    "    for (int column = 0; column < " +
    hidden_size.to_string() +
    "; column += 1) { const float x = __bfloat162float(input[input_base + column]); up_sum = __fadd_rn(up_sum, __fmul_rn(x, __bfloat162float(packed_up_gate_weight[up_base + column]))); gate_sum = __fadd_rn(gate_sum, __fmul_rn(x, __bfloat162float(packed_up_gate_weight[gate_base + column]))); }\n" +
    "    const __nv_bfloat16 up = __float2bfloat16_rn(up_sum); const __nv_bfloat16 gate = __float2bfloat16_rn(gate_sum); const float gate_f32 = __bfloat162float(gate); const __nv_bfloat16 activated = __float2bfloat16_rn(__fdiv_rn(gate_f32, __fadd_rn(1.0f, expf(-gate_f32)))); const __nv_bfloat16 product = __float2bfloat16_rn(__fmul_rn(__bfloat162float(up), __bfloat162float(activated))); down_sum = __fadd_rn(down_sum, __fmul_rn(__bfloat162float(product), __bfloat162float(down_weight[(unsigned long long)out * " +
    intermediate_size.to_string() +
    "ULL + inner]))); }\n" +
    "  output[i] = __float2bfloat16_rn(down_sum);\n}\n",
  )
  {
    numerical_contract: OrderedF32AccumulationBf16SwiGluDownBf16V1,
    row_count: rows,
    hidden_size,
    intermediate_size,
    function_symbol,
    source,
  }
}

///|
pub fn GatedMlpCudaSource::numerical_contract(
  self : Self,
) -> GatedMlpNumericalContract {
  self.numerical_contract
}

///|
pub fn GatedMlpCudaSource::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GatedMlpCudaSource::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn GatedMlpCudaSource::intermediate_size(self : Self) -> Int {
  self.intermediate_size
}

///|
pub fn GatedMlpCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GatedMlpCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
