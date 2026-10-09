///|
fn valid_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Renders channelwise F32 inverse normalization for contiguous
/// `[batch,channel,...]` storage. Multiplication and addition are separately
/// round-to-nearest; coefficient buffers are ordered by channel.
pub fn f32_channel_major_destandardization_source(
  batch_count~ : Int,
  channels~ : Int,
  elements_per_channel~ : Int64,
  function_symbol~ : String,
) -> AffineModulationCudaSource raise AffineModulationCudaSourceError {
  if batch_count <= 0 || channels <= 0 {
    raise InvalidWidth
  }
  if elements_per_channel <= 0L {
    raise InvalidRowCount
  }
  if channels.to_int64() > 9223372036854775807L / elements_per_channel {
    raise DimensionOverflow
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let channel_elements = channels.to_int64() * elements_per_channel
  if batch_count.to_int64() > 9223372036854775807L / channel_elements {
    raise DimensionOverflow
  }
  let elements = batch_count.to_int64() * channel_elements
  let source = @utf8.encode(
    "#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const float *normalized, const float *standard_deviations, const float *means, float *output) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  if (i >= " +
    elements.to_string() +
    "ULL) return;\n" +
    "  const unsigned long long channel = (i / " +
    elements_per_channel.to_string() +
    "ULL) % " +
    channels.to_string() +
    "ULL;\n" +
    "  const float scaled = __fmul_rn(normalized[i], standard_deviations[channel]);\n" +
    "  output[i] = __fadd_rn(scaled, means[channel]);\n}\n",
  )
  {
    operation: F32ChannelMajorDestandardization,
    numerical_contract: OrderedF32ChannelMajorMultiplyAddV1,
    row_count: elements_per_channel,
    width: channels,
    function_symbol,
    source,
  }
}

///|
/// Produces one BF16 shift/scale parameter row from one F32 conditioning row.
/// Weight layout is row-major `[2 * hidden_size,input_width]`; the first half
/// of outputs is shift and the second half is scale.
pub fn final_adaln_parameter_row_source(
  input_width~ : Int,
  hidden_size~ : Int,
  function_symbol~ : String,
) -> AffineModulationCudaSource raise AffineModulationCudaSourceError {
  if input_width <= 0 || hidden_size <= 0 {
    raise InvalidWidth
  }
  // The emitted kernel addresses `2 * hidden_size` outputs with CUDA `int`.
  // Reject before multiplying so neither MoonBit construction nor the CUDA
  // literal can wrap into a different launch/weight geometry.
  if hidden_size > 1073741823 {
    raise DimensionOverflow
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let outputs = hidden_size * 2
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const float *temb, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias, __nv_bfloat16 *shift, __nv_bfloat16 *scale) {\n" +
    "  const int output_index = (int)blockIdx.x;\n  if (output_index >= " +
    outputs.to_string() +
    ") return;\n" +
    "  if (threadIdx.x != 0) return;\n  float sum = __bfloat162float(bias[output_index]);\n" +
    "  const unsigned long long weight_base = (unsigned long long)output_index * " +
    input_width.to_string() +
    "ULL;\n" +
    "  for (int column = 0; column < " +
    input_width.to_string() +
    "; column += 1) {\n" +
    "    const float x = temb[column];\n" +
    "    const float activated_f32 = __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));\n" +
    "    const __nv_bfloat16 activated_bf16 = __float2bfloat16_rn(activated_f32);\n" +
    "    sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(activated_bf16), __bfloat162float(weight[weight_base + column])));\n  }\n" +
    "  const __nv_bfloat16 value = __float2bfloat16_rn(sum); if (output_index < " +
    hidden_size.to_string() +
    ") shift[output_index] = value; else scale[output_index - " +
    hidden_size.to_string() +
    "] = value;\n}\n",
  )
  {
    operation: FinalAdaLnParameterRow,
    numerical_contract: SiluF32CastBf16OrderedDenseBf16OutputV1,
    row_count: 1L,
    width: hidden_size,
    function_symbol,
    source,
  }
}

///|
/// Produces one BF16 shift/scale table row for every F32 conditioning row.
/// Weight layout is shared row-major `[2 * hidden_size,input_width]`; output
/// tables are row-major `[rows,hidden_size]`.
pub fn final_adaln_parameter_rows_source(
  rows~ : Int,
  input_width~ : Int,
  hidden_size~ : Int,
  function_symbol~ : String,
) -> AffineModulationCudaSource raise AffineModulationCudaSourceError {
  if rows <= 0 {
    raise InvalidRowCount
  }
  if input_width <= 0 || hidden_size <= 0 {
    raise InvalidWidth
  }
  if hidden_size > 1073741823 {
    raise DimensionOverflow
  }
  let outputs = hidden_size * 2
  if rows.to_int64() > 2147483647L / outputs.to_int64() {
    raise DimensionOverflow
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let blocks = rows.to_int64() * outputs.to_int64()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const float *temb_rows, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias, __nv_bfloat16 *shift_rows, __nv_bfloat16 *scale_rows) {\n" +
    "  const unsigned long long flat = (unsigned long long)blockIdx.x;\n  if (flat >= " +
    blocks.to_string() +
    "ULL || threadIdx.x != 0) return;\n" +
    "  const unsigned long long row = flat / " +
    outputs.to_string() +
    "ULL; const int output_index = (int)(flat % " +
    outputs.to_string() +
    "ULL);\n" +
    "  float sum = __bfloat162float(bias[output_index]); const unsigned long long weight_base = (unsigned long long)output_index * " +
    input_width.to_string() +
    "ULL; const unsigned long long input_base = row * " +
    input_width.to_string() +
    "ULL;\n" +
    "  for (int column = 0; column < " +
    input_width.to_string() +
    "; column += 1) { const float x = temb_rows[input_base + column]; const float activated_f32 = __fdiv_rn(x, __fadd_rn(1.0f, expf(-x))); const __nv_bfloat16 activated_bf16 = __float2bfloat16_rn(activated_f32); sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(activated_bf16), __bfloat162float(weight[weight_base + column]))); }\n" +
    "  const __nv_bfloat16 value = __float2bfloat16_rn(sum); const unsigned long long output_base = row * " +
    hidden_size.to_string() +
    "ULL; if (output_index < " +
    hidden_size.to_string() +
    ") shift_rows[output_base + output_index] = value; else scale_rows[output_base + output_index - " +
    hidden_size.to_string() +
    "] = value;\n}\n",
  )
  {
    operation: FinalAdaLnParameterRow,
    numerical_contract: SiluF32CastBf16OrderedDenseBf16OutputV1,
    row_count: rows.to_int64(),
    width: hidden_size,
    function_symbol,
    source,
  }
}

///|
/// Gathers host-validated packed BF16 shift/scale rows into modality-aligned
/// video and audio buffers. Indices are trusted products of request admission.
pub fn packed_modality_adaln_gather_source(
  packed_rows~ : Int,
  video_rows~ : Int,
  audio_rows~ : Int,
  hidden_size~ : Int,
  function_symbol~ : String,
) -> AffineModulationCudaSource raise AffineModulationCudaSourceError {
  if packed_rows <= 0 || video_rows <= 0 || audio_rows <= 0 {
    raise InvalidRowCount
  }
  if hidden_size <= 0 {
    raise InvalidWidth
  }
  if video_rows > packed_rows - audio_rows || audio_rows > packed_rows {
    raise InvalidRowCount
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let selected_rows = video_rows + audio_rows
  let elements = selected_rows.to_int64() * hidden_size.to_int64()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const __nv_bfloat16 *packed_shift, const __nv_bfloat16 *packed_scale, const int32_t *video_indices, const int32_t *audio_indices, __nv_bfloat16 *video_shift, __nv_bfloat16 *video_scale, __nv_bfloat16 *audio_shift, __nv_bfloat16 *audio_scale) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  if (i >= " +
    elements.to_string() +
    "ULL) return;\n" +
    "  const unsigned long long selected_row = i / " +
    hidden_size.to_string() +
    "ULL;\n" +
    "  const unsigned long long column = i % " +
    hidden_size.to_string() +
    "ULL;\n" +
    "  if (selected_row < " +
    video_rows.to_string() +
    "ULL) { const unsigned long long source = (unsigned long long)video_indices[selected_row] * " +
    hidden_size.to_string() +
    "ULL + column; video_shift[i] = packed_shift[source]; video_scale[i] = packed_scale[source]; }\n" +
    "  else { const unsigned long long row = selected_row - " +
    video_rows.to_string() +
    "ULL; const unsigned long long source = (unsigned long long)audio_indices[row] * " +
    hidden_size.to_string() +
    "ULL + column; const unsigned long long output = row * " +
    hidden_size.to_string() +
    "ULL + column; audio_shift[output] = packed_shift[source]; audio_scale[output] = packed_scale[source]; }\n}\n",
  )
  {
    operation: PackedModalityAdaLnGather,
    numerical_contract: BitExactBf16PackedModalityGatherV1,
    row_count: selected_rows.to_int64(),
    width: hidden_size,
    function_symbol,
    source,
  }
}

///|
/// Renders the exact row-aligned BF16 AdaLN affine stage followed by the F32
/// output-head boundary. Each BF16 arithmetic result is rounded before the
/// next operation: `hidden * (1 + scale) + shift`, then widened to F32.
pub fn bf16_shift_scale_f32_fixed_rows_source(
  rows~ : Int64,
  width~ : Int,
  function_symbol~ : String,
) -> AffineModulationCudaSource raise AffineModulationCudaSourceError {
  if rows <= 0L || rows > 2147483647L {
    raise InvalidRowCount
  }
  if width <= 0 {
    raise InvalidWidth
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let elements = rows * width.to_int64()
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const __nv_bfloat16 *normalized, const __nv_bfloat16 *shift, const __nv_bfloat16 *scale, float *output) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  if (i >= " +
    elements.to_string() +
    "ULL) return;\n" +
    "  const __nv_bfloat16 one_plus_scale = __float2bfloat16_rn(__fadd_rn(1.0f, __bfloat162float(scale[i])));\n" +
    "  const __nv_bfloat16 scaled = __float2bfloat16_rn(__fmul_rn(__bfloat162float(normalized[i]), __bfloat162float(one_plus_scale)));\n" +
    "  const __nv_bfloat16 modulated = __float2bfloat16_rn(__fadd_rn(__bfloat162float(scaled), __bfloat162float(shift[i])));\n" +
    "  output[i] = __bfloat162float(modulated);\n}\n",
  )
  {
    operation: Bf16ShiftScaleToF32,
    numerical_contract: StagedBf16ShiftScaleF32OutputV1,
    row_count: rows,
    width,
    function_symbol,
    source,
  }
}
