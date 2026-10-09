///|
pub fn f32_weight_normalized_conv_transpose1d_source(
  batch_count~ : Int,
  input_channels~ : Int,
  output_channels~ : Int,
  input_time~ : Int,
  kernel_size~ : Int,
  padding~ : Int,
  stride~ : Int,
  normalize_symbol~ : String,
  convolution_symbol~ : String,
) -> WeightNormalizedConvTranspose1dCudaSource raise ChannelMajorProjectionCudaSourceError {
  if batch_count <= 0 {
    raise InvalidBatchCount
  }
  if input_channels <= 0 || output_channels <= 0 || kernel_size <= 0 {
    raise InvalidChannelCount
  }
  if input_time <= 0 || padding < 0 || stride <= 0 {
    raise InvalidTimeCount
  }
  if !valid_symbol(normalize_symbol) ||
    !valid_symbol(convolution_symbol) ||
    normalize_symbol == convolution_symbol {
    raise InvalidFunctionSymbol
  }
  let input_span = if input_time == 1 {
    0L
  } else {
    checked_mul(input_time.to_int64() - 1L, stride.to_int64())
  }
  let twice_padding = if padding == 0 {
    0L
  } else {
    checked_mul(padding.to_int64(), 2L)
  }
  if input_span > 9223372036854775807L - kernel_size.to_int64() {
    raise DimensionOverflow
  }
  let unpadded = input_span + kernel_size.to_int64()
  if unpadded <= twice_padding {
    raise InvalidTimeCount
  }
  let output_time64 = unpadded - twice_padding
  if output_time64 <= 0L || output_time64 > 2147483647L {
    raise InvalidTimeCount
  }
  let output_time = output_time64.to_int()
  let input_elements = checked_mul(
    checked_mul(batch_count.to_int64(), input_channels.to_int64()),
    input_time.to_int64(),
  )
  let per_input = checked_mul(
    output_channels.to_int64(),
    kernel_size.to_int64(),
  )
  let weight_elements = checked_mul(input_channels.to_int64(), per_input)
  let output_elements = checked_mul(
    checked_mul(batch_count.to_int64(), output_channels.to_int64()),
    output_time64,
  )
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\nextern \"C\" __global__ void " +
    normalize_symbol +
    "(const float *weight_g, const float *weight_v, float *normalized_weight) {\n" +
    "  const int input_channel = (int)blockIdx.x; if (input_channel >= " +
    input_channels.to_string() +
    " || threadIdx.x != 0) return;\n" +
    "  const unsigned long long base = (unsigned long long)input_channel * " +
    output_channels.to_string() +
    "ULL * " +
    kernel_size.to_string() +
    "ULL; float sum_squares = 0.0f;\n" +
    "  for (int output_channel = 0; output_channel < " +
    output_channels.to_string() +
    "; output_channel += 1) { for (int kernel_index = 0; kernel_index < " +
    kernel_size.to_string() +
    "; kernel_index += 1) { const unsigned long long index = base + (unsigned long long)output_channel * " +
    kernel_size.to_string() +
    "ULL + kernel_index; const float value = weight_v[index]; sum_squares = __fadd_rn(sum_squares, __fmul_rn(value, value)); }}\n" +
    "  const float scale = __fdiv_rn(weight_g[input_channel], sqrtf(sum_squares)); for (unsigned long long offset = 0; offset < " +
    per_input.to_string() +
    "ULL; offset += 1ULL) normalized_weight[base + offset] = __fmul_rn(weight_v[base + offset], scale);\n}\n" +
    "extern \"C\" __global__ void " +
    convolution_symbol +
    "(const float *input, const float *normalized_weight, const float *bias, float *output) {\n" +
    "  const unsigned long long flat = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (flat >= " +
    output_elements.to_string() +
    "ULL) return;\n" +
    "  const unsigned long long output_time_index = flat % " +
    output_time.to_string() +
    "ULL; const unsigned long long outer = flat / " +
    output_time.to_string() +
    "ULL; const unsigned long long output_channel = outer % " +
    output_channels.to_string() +
    "ULL; const unsigned long long batch_index = outer / " +
    output_channels.to_string() +
    "ULL; float sum = bias[output_channel];\n" +
    "  for (int input_channel = 0; input_channel < " +
    input_channels.to_string() +
    "; input_channel += 1) { const unsigned long long weight_base = ((unsigned long long)input_channel * " +
    output_channels.to_string() +
    "ULL + output_channel) * " +
    kernel_size.to_string() +
    "ULL; const unsigned long long input_base = (batch_index * " +
    input_channels.to_string() +
    "ULL + input_channel) * " +
    input_time.to_string() +
    "ULL; for (int input_time_index = 0; input_time_index < " +
    input_time.to_string() +
    "; input_time_index += 1) { const long long kernel_index = (long long)output_time_index - (long long)input_time_index * " +
    stride.to_string() +
    "LL + " +
    padding.to_string() +
    "LL; if (kernel_index >= 0 && kernel_index < " +
    kernel_size.to_string() +
    "LL) sum = __fadd_rn(sum, __fmul_rn(input[input_base + input_time_index], normalized_weight[weight_base + (unsigned long long)kernel_index])); }} output[flat] = sum;\n}\n",
  )
  {
    operation: F32WeightNormalizedConvTranspose1d,
    numerical_contract: OrderedF32PerInputWeightNormConvTranspose1dV1,
    batch_count,
    input_channels,
    output_channels,
    input_time,
    output_time,
    kernel_size,
    padding,
    stride,
    input_elements,
    weight_elements,
    output_elements,
    normalize_symbol,
    convolution_symbol,
    source,
  }
}
