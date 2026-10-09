///|
pub fn f32_weight_normalized_conv1d_source(
  batch_count~ : Int,
  input_channels~ : Int,
  output_channels~ : Int,
  input_time~ : Int,
  kernel_size~ : Int,
  padding~ : Int,
  stride~ : Int,
  normalize_symbol~ : String,
  convolution_symbol~ : String,
  use_bias? : Bool = true,
) -> WeightNormalizedConv1dCudaSource raise ChannelMajorProjectionCudaSourceError {
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
  let padded = input_time.to_int64() + padding.to_int64() * 2L
  if padded < kernel_size.to_int64() {
    raise InvalidTimeCount
  }
  let output_time = ((padded - kernel_size.to_int64()) / stride.to_int64() + 1L).to_int()
  let input_elements = checked_mul(
    checked_mul(batch_count.to_int64(), input_channels.to_int64()),
    input_time.to_int64(),
  )
  let per_output = checked_mul(
    input_channels.to_int64(),
    kernel_size.to_int64(),
  )
  let weight_elements = checked_mul(output_channels.to_int64(), per_output)
  let output_elements = checked_mul(
    checked_mul(batch_count.to_int64(), output_channels.to_int64()),
    output_time.to_int64(),
  )
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\nextern \"C\" __global__ void " +
    normalize_symbol +
    "(const float *weight_g, const float *weight_v, float *normalized_weight) {\n" +
    "  const int output_channel = (int)blockIdx.x; if (output_channel >= " +
    output_channels.to_string() +
    " || threadIdx.x != 0) return;\n" +
    "  const unsigned long long base = (unsigned long long)output_channel * " +
    input_channels.to_string() +
    "ULL * " +
    kernel_size.to_string() +
    "ULL; float sum_squares = 0.0f;\n" +
    "  for (int input_channel = 0; input_channel < " +
    input_channels.to_string() +
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
    "ULL; float sum = " +
    (if use_bias { "bias[output_channel]" } else { "0.0f" }) +
    ";\n" +
    "  const unsigned long long weight_base = output_channel * " +
    input_channels.to_string() +
    "ULL * " +
    kernel_size.to_string() +
    "ULL; const unsigned long long input_batch_base = batch_index * " +
    input_channels.to_string() +
    "ULL * " +
    input_time.to_string() +
    "ULL;\n" +
    "  for (int input_channel = 0; input_channel < " +
    input_channels.to_string() +
    "; input_channel += 1) { for (int kernel_index = 0; kernel_index < " +
    kernel_size.to_string() +
    "; kernel_index += 1) { const long long input_time_index = (long long)output_time_index * " +
    stride.to_string() +
    "LL + kernel_index - " +
    padding.to_string() +
    "LL; if (input_time_index >= 0 && input_time_index < " +
    input_time.to_string() +
    "LL) { const unsigned long long input_index = input_batch_base + (unsigned long long)input_channel * " +
    input_time.to_string() +
    "ULL + (unsigned long long)input_time_index; const unsigned long long weight_index = weight_base + (unsigned long long)input_channel * " +
    kernel_size.to_string() +
    "ULL + kernel_index; sum = __fadd_rn(sum, __fmul_rn(input[input_index], normalized_weight[weight_index])); } }} output[flat] = sum;\n}\n",
  )
  {
    operation: F32WeightNormalizedConv1d,
    numerical_contract: if use_bias {
      OrderedF32PerOutputWeightNormConv1dV1
    } else {
      OrderedF32PerOutputWeightNormBiasFreeConv1dV1
    },
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
