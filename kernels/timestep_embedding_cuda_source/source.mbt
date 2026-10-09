///|
fn valid_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Renders MiniMax-compatible generic sinusoidal timestep rows: unscaled F32
/// input, period 10000, downscale shift zero, and flipped `[cos,sin]` halves.
pub fn cosine_sine_timestep_rows_source(
  rows~ : Int,
  frequency_width~ : Int,
  function_symbol~ : String,
) -> TimestepEmbeddingCudaSource raise TimestepEmbeddingCudaSourceError {
  if rows <= 0 {
    raise InvalidRowCount
  }
  if frequency_width <= 0 || frequency_width % 2 != 0 {
    raise InvalidFrequencyWidth
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let half = frequency_width / 2
  let elements = rows.to_int64() * frequency_width.to_int64()
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const float *timestep, float *frequency_rows) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  if (i >= " +
    elements.to_string() +
    "ULL) return;\n" +
    "  const int column = (int)(i % " +
    frequency_width.to_string() +
    "ULL);\n" +
    "  const int frequency = column % " +
    half.to_string() +
    ";\n" +
    "  const float exponent = __fdiv_rn(__fmul_rn(-9.2103404998779296875f, (float)frequency), " +
    half.to_string() +
    ".0f);\n" +
    "  const float angle = __fmul_rn(timestep[i / " +
    frequency_width.to_string() +
    "ULL], expf(exponent));\n" +
    "  frequency_rows[i] = column < " +
    half.to_string() +
    " ? cosf(angle) : sinf(angle);\n}\n",
  )
  {
    numerical_contract: Float32CosSinExpfPeriod10000V1,
    row_count: rows,
    frequency_width,
    function_symbol,
    source,
  }
}

///|
/// Renders a workspace-free correctness kernel for two biased F32 dense
/// layers with SiLU between them. Weights are row-major `[output,input]`.
pub fn dense_silu_dense_f32_rows_source(
  rows~ : Int,
  input_width~ : Int,
  hidden_width~ : Int,
  output_width~ : Int,
  function_symbol~ : String,
) -> TimestepEmbeddingCudaSource raise TimestepEmbeddingCudaSourceError {
  if rows <= 0 {
    raise InvalidRowCount
  }
  if input_width <= 0 || hidden_width <= 0 || output_width <= 0 {
    raise InvalidFrequencyWidth
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let elements = rows.to_int64() * output_width.to_int64()
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const float *input, const float *weight1, const float *bias1, const float *weight2, const float *bias2, float *output) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  if (i >= " +
    elements.to_string() +
    "ULL) return;\n" +
    "  const unsigned long long row = i / " +
    output_width.to_string() +
    "ULL; const int out = (int)(i % " +
    output_width.to_string() +
    "ULL);\n" +
    "  float result = bias2[out];\n" +
    "  for (int hidden = 0; hidden < " +
    hidden_width.to_string() +
    "; hidden += 1) {\n" +
    "    float first = bias1[hidden]; const unsigned long long w1 = (unsigned long long)hidden * " +
    input_width.to_string() +
    "ULL;\n" +
    "    for (int column = 0; column < " +
    input_width.to_string() +
    "; column += 1) first = __fadd_rn(first, __fmul_rn(input[row * " +
    input_width.to_string() +
    "ULL + column], weight1[w1 + column]));\n" +
    "    const float activated = __fdiv_rn(first, __fadd_rn(1.0f, expf(-first)));\n" +
    "    result = __fadd_rn(result, __fmul_rn(activated, weight2[(unsigned long long)out * " +
    hidden_width.to_string() +
    "ULL + hidden]));\n  }\n" +
    "  output[i] = result;\n}\n",
  )
  {
    numerical_contract: OrderedFloat32DenseSiluDenseV1,
    row_count: rows,
    frequency_width: output_width,
    function_symbol,
    source,
  }
}

///|
pub fn TimestepEmbeddingCudaSource::numerical_contract(
  self : Self,
) -> TimestepEmbeddingNumericalContract {
  self.numerical_contract
}

///|
pub fn TimestepEmbeddingCudaSource::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn TimestepEmbeddingCudaSource::frequency_width(self : Self) -> Int {
  self.frequency_width
}

///|
pub fn TimestepEmbeddingCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn TimestepEmbeddingCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
