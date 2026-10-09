///|
pub fn f32_weight_normalized_dilated_conv1d_source(
  batch_count~ : Int,
  channel_count~ : Int,
  time_count~ : Int,
  kernel_size~ : Int,
  dilation~ : Int,
  padding~ : Int,
  normalize_symbol~ : String,
  convolution_symbol~ : String,
) -> WeightNormalizedDilatedConv1dCudaSource raise ChannelMajorProjectionCudaSourceError {
  if batch_count <= 0 {
    raise InvalidBatchCount
  }
  if channel_count <= 0 || kernel_size <= 1 {
    raise InvalidChannelCount
  }
  if time_count <= 0 || dilation <= 0 || padding < 0 {
    raise InvalidTimeCount
  }
  if !valid_symbol(normalize_symbol) ||
    !valid_symbol(convolution_symbol) ||
    normalize_symbol == convolution_symbol {
    raise InvalidFunctionSymbol
  }
  let receptive = checked_mul((kernel_size - 1).to_int64(), dilation.to_int64())
  if padding.to_int64() > 9223372036854775807L / 2L {
    raise DimensionOverflow
  }
  let padding_extent = padding.to_int64() * 2L
  if time_count.to_int64() > 9223372036854775807L - padding_extent {
    raise DimensionOverflow
  }
  let padded_time = time_count.to_int64() + padding_extent
  if receptive > 9223372036854775807L - 1L ||
    receptive + 1L > padded_time ||
    padded_time - receptive != time_count.to_int64() {
    raise InvalidTimeCount
  }
  let element_count = checked_mul(
    checked_mul(batch_count.to_int64(), channel_count.to_int64()),
    time_count.to_int64(),
  )
  let per_output = checked_mul(channel_count.to_int64(), kernel_size.to_int64())
  let weight_elements = checked_mul(channel_count.to_int64(), per_output)
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\nextern \"C\" __global__ void " +
    normalize_symbol +
    "(const float *weight_g, const float *weight_v, float *normalized_weight) {\n" +
    "  const int output_channel = (int)blockIdx.x; if (output_channel >= " +
    channel_count.to_string() +
    " || threadIdx.x != 0) return; const unsigned long long base = (unsigned long long)output_channel * " +
    per_output.to_string() +
    "ULL; float sum_squares = 0.0f;\n" +
    "  for (int input_channel = 0; input_channel < " +
    channel_count.to_string() +
    "; input_channel += 1) { for (int kernel_index = 0; kernel_index < " +
    kernel_size.to_string() +
    "; kernel_index += 1) { const unsigned long long index = base + (unsigned long long)input_channel * " +
    kernel_size.to_string() +
    "ULL + kernel_index; const float value = weight_v[index]; sum_squares = __fadd_rn(sum_squares, __fmul_rn(value, value)); }}\n" +
    "  const float scale = __fdiv_rn(weight_g[output_channel], sqrtf(sum_squares)); for (unsigned long long offset = 0; offset < " +
    per_output.to_string() +
    "ULL; offset += 1ULL) normalized_weight[base + offset] = __fmul_rn(weight_v[base + offset], scale);\n}\n" +
    "extern \"C\" __global__ void " +
    convolution_symbol +
    "(const float *input, const float *normalized_weight, const float *bias, float *output) {\n" +
    "  const unsigned long long flat = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (flat >= " +
    element_count.to_string() +
    "ULL) return; const unsigned long long time_index = flat % " +
    time_count.to_string() +
    "ULL; const unsigned long long outer = flat / " +
    time_count.to_string() +
    "ULL; const unsigned long long output_channel = outer % " +
    channel_count.to_string() +
    "ULL; const unsigned long long batch_index = outer / " +
    channel_count.to_string() +
    "ULL; float sum = bias[output_channel];\n" +
    "  const unsigned long long weight_base = output_channel * " +
    per_output.to_string() +
    "ULL; const unsigned long long input_batch_base = batch_index * " +
    channel_count.to_string() +
    "ULL * " +
    time_count.to_string() +
    "ULL; for (int input_channel = 0; input_channel < " +
    channel_count.to_string() +
    "; input_channel += 1) { for (int kernel_index = 0; kernel_index < " +
    kernel_size.to_string() +
    "; kernel_index += 1) { const long long input_time_index = (long long)time_index + (long long)kernel_index * " +
    dilation.to_string() +
    "LL - " +
    padding.to_string() +
    "LL; if (input_time_index >= 0 && input_time_index < " +
    time_count.to_string() +
    "LL) { const unsigned long long input_index = input_batch_base + (unsigned long long)input_channel * " +
    time_count.to_string() +
    "ULL + (unsigned long long)input_time_index; const unsigned long long weight_index = weight_base + (unsigned long long)input_channel * " +
    kernel_size.to_string() +
    "ULL + kernel_index; sum = __fadd_rn(sum, __fmul_rn(input[input_index], normalized_weight[weight_index])); } }} output[flat] = sum;\n}\n",
  )
  {
    operation: F32WeightNormalizedDilatedConv1d,
    numerical_contract: OrderedF32PerOutputWeightNormDilatedConv1dV1,
    batch_count,
    channel_count,
    time_count,
    kernel_size,
    dilation,
    padding,
    element_count,
    weight_elements,
    normalize_symbol,
    convolution_symbol,
    source,
  }
}
