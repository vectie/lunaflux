///|
fn valid_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn checked_mul(
  left : Int64,
  right : Int64,
) -> Int64 raise ChannelMajorProjectionCudaSourceError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise DimensionOverflow
  }
  left * right
}

///|
/// Renders a correctness-grade F32 pointwise Conv1d over contiguous
/// `[batch,input_channel,time]` input and `[batch,output_channel,time]`
/// output. Weight storage is `[output_channel,input_channel,1]`; padding is
/// zero and stride is one. Bias is the initial F32 accumulator, followed by
/// input-channel-ascending separately rounded multiply and add operations.
pub fn f32_pointwise_conv1d_source(
  batch_count~ : Int,
  input_channels~ : Int,
  output_channels~ : Int,
  time_count~ : Int,
  function_symbol~ : String,
  autocast? : @precision.AutocastFloat? = None,
) -> ChannelMajorProjectionCudaSource raise ChannelMajorProjectionCudaSourceError {
  if batch_count <= 0 {
    raise InvalidBatchCount
  }
  if input_channels <= 0 || output_channels <= 0 {
    raise InvalidChannelCount
  }
  if time_count <= 0 {
    raise InvalidTimeCount
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let batch = batch_count.to_int64()
  let time = time_count.to_int64()
  let input_elements = checked_mul(
    checked_mul(batch, input_channels.to_int64()),
    time,
  )
  let output_elements = checked_mul(
    checked_mul(batch, output_channels.to_int64()),
    time,
  )
  fn rounded(value : String) -> String {
    match autocast {
      Some(dtype) => @autocast.round_source(value, dtype)
      None => value
    }
  }
  let source = @utf8.encode(
    (match autocast {
      Some(dtype) => @autocast.include_source(dtype)
      None => ""
    }) +
    "#include <stdint.h>\nextern \"C\" __global__ void " +
    function_symbol +
    "(const float *input, const float *weight, const float *bias, float *output) {\n" +
    "  const unsigned long long flat = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  if (flat >= " +
    output_elements.to_string() +
    "ULL) return;\n" +
    "  const unsigned long long time_index = flat % " +
    time_count.to_string() +
    "ULL;\n" +
    "  const unsigned long long outer = flat / " +
    time_count.to_string() +
    "ULL;\n" +
    "  const unsigned long long output_channel = outer % " +
    output_channels.to_string() +
    "ULL;\n" +
    "  const unsigned long long batch_index = outer / " +
    output_channels.to_string() +
    "ULL;\n" +
    (if autocast is Some(_) {
      "  float sum = 0.0f;\n"
    } else {
      "  float sum = bias[output_channel];\n"
    }) +
    "  const unsigned long long weight_base = output_channel * " +
    input_channels.to_string() +
    "ULL;\n" +
    "  const unsigned long long input_batch_base = batch_index * " +
    input_channels.to_string() +
    "ULL * " +
    time_count.to_string() +
    "ULL;\n" +
    "  for (int channel = 0; channel < " +
    input_channels.to_string() +
    "; channel += 1) {\n" +
    "    const float product = __fmul_rn(" +
    rounded(
      "input[input_batch_base + (unsigned long long)channel * " +
      time_count.to_string() +
      "ULL + time_index]",
    ) +
    ", " +
    rounded("weight[weight_base + channel]") +
    ");\n" +
    "    sum = __fadd_rn(sum, product);\n  }\n" +
    (if autocast is Some(_) {
      "  sum = __fadd_rn(sum, " + rounded("bias[output_channel]") + ");\n"
    } else {
      ""
    }) +
    "  output[flat] = " +
    rounded("sum") +
    ";\n}\n",
  )
  {
    operation: F32PointwiseConv1d,
    numerical_contract: match autocast {
      Some(dtype) => AutocastPointwiseF32AccumulationV1(dtype)
      None => OrderedF32BiasAccumulationF32OutputV1
    },
    batch_count,
    input_channels,
    output_channels,
    time_count,
    input_elements,
    output_elements,
    function_symbol,
    source,
  }
}
