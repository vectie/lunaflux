///|
fn valid_short_convolution_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn positive_product(left : Int64, right : Int64) -> Int64? {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    None
  } else {
    Some(left * right)
  }
}

///|
/// Renders one reusable bias-free depthwise causal convolution invocation.
/// Input/output are token-major `[maximum_rows, channel_count]`; weights are
/// `[channel_count, kernel_size]`, oldest lag first and current token last.
/// Initial/final BF16 state is `[maximum_sequences, channel_count,
/// kernel_size-1]`, also oldest first. `sequence_offsets` is an I32 CSR-style
/// array of length `maximum_sequences+1`; live entries partition the exact
/// `counts[3]` token prefix into `counts[2]` non-empty sequences.
pub fn depthwise_causal_silu_bf16_source(
  channel_count~ : Int,
  kernel_size~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  function_symbol~ : String,
) -> CausalShortConvolutionCudaSource raise CausalShortConvolutionCudaSourceError {
  if channel_count <= 0 {
    raise InvalidChannelCount
  }
  if kernel_size <= 1 {
    raise InvalidKernelSize
  }
  if maximum_rows <= 0 || maximum_rows > 2147483647 {
    raise InvalidRowCount
  }
  if maximum_sequences <= 0 || maximum_sequences > maximum_rows {
    raise InvalidSequenceCount
  }
  let history = kernel_size - 1
  guard positive_product(maximum_rows.to_int64(), channel_count.to_int64())
    is Some(_) &&
    positive_product(channel_count.to_int64(), kernel_size.to_int64())
    is Some(_) &&
    positive_product(maximum_sequences.to_int64(), channel_count.to_int64())
    is Some(sequence_channels) &&
    positive_product(sequence_channels, history.to_int64()) is Some(_) else {
    raise DimensionOverflow
  }
  if !valid_short_convolution_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *sequence_offsets, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, const __nv_bfloat16 *initial_state, __nv_bfloat16 *output, __nv_bfloat16 *final_state) {\n" +
    "  const int32_t sequence_count = counts[2]; const int32_t live_rows = counts[3];\n" +
    "  if (counts[0] < 0 || counts[1] < 0 || sequence_count <= 0 || sequence_count > " +
    maximum_sequences.to_string() +
    " || counts[0] > sequence_count || counts[1] != sequence_count - counts[0] || live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  if (sequence_offsets[0] != 0 || sequence_offsets[sequence_count] != live_rows) return;\n" +
    "  for (int32_t s = 0; s < sequence_count; s += 1) { const int32_t begin = sequence_offsets[s]; const int32_t end = sequence_offsets[s + 1]; if (begin < 0 || end <= begin || end > live_rows) return; }\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const unsigned long long output_cells = (unsigned long long)live_rows * " +
    channel_count.to_string() +
    "ULL;\n" +
    "  if (i < output_cells) { const int32_t row = (int32_t)(i / " +
    channel_count.to_string() +
    "ULL); const int32_t channel = (int32_t)(i % " +
    channel_count.to_string() +
    "ULL); int32_t sequence = -1;\n" +
    "    for (int32_t s = 0; s < sequence_count; s += 1) { if (row >= sequence_offsets[s] && row < sequence_offsets[s + 1]) { sequence = s; break; } } if (sequence < 0) return;\n" +
    "    const int32_t local_row = row - sequence_offsets[sequence]; float sum = 0.0f;\n" +
    "    for (int32_t offset = 0; offset < " +
    kernel_size.to_string() +
    "; offset += 1) { const int32_t source = local_row - " +
    history.to_string() +
    " + offset; const __nv_bfloat16 value = source < 0 ? initial_state[((unsigned long long)sequence * " +
    channel_count.to_string() +
    "ULL + channel) * " +
    history.to_string() +
    "ULL + (" +
    history.to_string() +
    " + source)] : input[(unsigned long long)(sequence_offsets[sequence] + source) * " +
    channel_count.to_string() +
    "ULL + channel]; sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(value), __bfloat162float(weight[(unsigned long long)channel * " +
    kernel_size.to_string() +
    "ULL + offset]))); }\n" +
    "    output[i] = __float2bfloat16_rn(__fdiv_rn(sum, __fadd_rn(1.0f, expf(-sum)))); }\n" +
    "  const unsigned long long state_cells = (unsigned long long)sequence_count * " +
    channel_count.to_string() +
    "ULL * " +
    history.to_string() +
    "ULL;\n" +
    "  if (i < state_cells) { const int32_t history_offset = (int32_t)(i % " +
    history.to_string() +
    "ULL); const unsigned long long sequence_channel = i / " +
    history.to_string() +
    "ULL; const int32_t channel = (int32_t)(sequence_channel % " +
    channel_count.to_string() +
    "ULL); const int32_t sequence = (int32_t)(sequence_channel / " +
    channel_count.to_string() +
    "ULL); const int32_t length = sequence_offsets[sequence + 1] - sequence_offsets[sequence]; const int32_t source = length + history_offset - " +
    history.to_string() +
    "; final_state[i] = source < 0 ? initial_state[i + length] : input[(unsigned long long)(sequence_offsets[sequence] + source) * " +
    channel_count.to_string() +
    "ULL + channel]; }\n" +
    "}\n",
  )
  {
    operation: DepthwiseCausalSiluBf16,
    numerical_contract: OrderedFloat32AccumulationSiluBf16RoundV1,
    channel_count,
    kernel_size,
    maximum_rows,
    maximum_sequences,
    function_symbol,
    source,
  }
}

///|
pub fn CausalShortConvolutionCudaSource::operation(
  self : Self,
) -> CausalShortConvolutionOperation {
  self.operation
}

///|
pub fn CausalShortConvolutionCudaSource::numerical_contract(
  self : Self,
) -> CausalShortConvolutionNumericalContract {
  self.numerical_contract
}

///|
pub fn CausalShortConvolutionCudaSource::channel_count(self : Self) -> Int {
  self.channel_count
}

///|
pub fn CausalShortConvolutionCudaSource::kernel_size(self : Self) -> Int {
  self.kernel_size
}

///|
pub fn CausalShortConvolutionCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn CausalShortConvolutionCudaSource::maximum_sequences(self : Self) -> Int {
  self.maximum_sequences
}

///|
pub fn CausalShortConvolutionCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn CausalShortConvolutionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
