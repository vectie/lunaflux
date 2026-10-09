///|
fn valid_compressed_kv_prepare_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Emits one serial correctness block per batch row. Mode zero is official
/// prefill: the first `sequence_length` ordinary rows are followed immediately
/// by `floor(sequence_length / ratio)` compressed rows. Mode one is official
/// decode storage: the configured circular window is copied in physical slot
/// order, followed by the completed compressed-cache prefix. No sort, dedup,
/// learned compression, cache ownership, index production, or attention occurs.
pub fn non_overlapping_shared_kv_prepare_bf16_source(
  window_size~ : Int,
  compression_ratio~ : Int,
  head_dimension~ : Int,
  maximum_position_count~ : Int,
  maximum_batches~ : Int,
  function_symbol~ : String,
) -> DecoderCompressedKvPrepareCudaSource raise DecoderCompressedKvPrepareCudaSourceError {
  if window_size <= 0 || compression_ratio <= 1 {
    raise InvalidCompressionRatio
  }
  if head_dimension <= 0 || head_dimension > 65536 {
    raise InvalidHeadDimension
  }
  if maximum_position_count <= 0 ||
    maximum_position_count > 2147483647 ||
    window_size > maximum_position_count ||
    compression_ratio > maximum_position_count {
    raise InvalidMaximumPositionCount
  }
  if maximum_batches <= 0 || maximum_batches > 1048576 {
    raise InvalidMaximumBatches
  }
  if !valid_compressed_kv_prepare_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let maximum_compressed_slots = maximum_position_count / compression_ratio
  if maximum_position_count > 2147483647 - maximum_compressed_slots {
    raise SizeOverflow
  }
  let maximum_key_slots = maximum_position_count + maximum_compressed_slots
  let batches64 = maximum_batches.to_int64()
  let slots64 = maximum_key_slots.to_int64()
  let head64 = head_dimension.to_int64()
  if batches64 > 9223372036854775807L / slots64 ||
    batches64 * slots64 > 9223372036854775807L / head64 ||
    batches64 * slots64 * head64 > 9223372036854775807L / 2L {
    raise SizeOverflow
  }
  let source = @utf8.encode(
    "#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *modes, const int32_t *sequence_lengths, const uint16_t *__restrict__ ordinary_kv, const uint16_t *__restrict__ compressed_kv, int32_t *prepared_lengths, uint16_t *__restrict__ prepared_kv) {\n" +
    "  if (threadIdx.x != 0) return; const int32_t batch = (int32_t)blockIdx.x; if (batch >= " +
    maximum_batches.to_string() +
    ") return;\n" +
    "  const unsigned long long output_base = (unsigned long long)batch * " +
    maximum_key_slots.to_string() +
    "ULL * " +
    head_dimension.to_string() +
    "ULL; for (unsigned long long i = 0; i < " +
    maximum_key_slots.to_string() +
    "ULL * " +
    head_dimension.to_string() +
    "ULL; ++i) prepared_kv[output_base + i] = (uint16_t)0; prepared_lengths[batch] = 0;\n" +
    "  const int32_t live_batches = counts[3]; if (live_batches <= 0 || live_batches > " +
    maximum_batches.to_string() +
    " || batch >= live_batches) return; const int32_t mode = modes[batch]; const int32_t length = sequence_lengths[batch]; if ((mode != 0 && mode != 1) || length <= 0 || length > " +
    maximum_position_count.to_string() +
    ") return;\n" +
    "  const int32_t ordinary_slots = mode == 0 ? length : " +
    window_size.to_string() +
    "; const int32_t compressed_slots = length / " +
    compression_ratio.to_string() +
    "; const unsigned long long ordinary_base = (unsigned long long)batch * " +
    maximum_position_count.to_string() +
    "ULL * " +
    head_dimension.to_string() +
    "ULL; const unsigned long long compressed_base = (unsigned long long)batch * " +
    maximum_compressed_slots.to_string() +
    "ULL * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "  for (int32_t slot = 0; slot < ordinary_slots; ++slot) for (int32_t feature = 0; feature < " +
    head_dimension.to_string() +
    "; ++feature) prepared_kv[output_base + (unsigned long long)slot * " +
    head_dimension.to_string() +
    "ULL + (unsigned long long)feature] = ordinary_kv[ordinary_base + (unsigned long long)slot * " +
    head_dimension.to_string() +
    "ULL + (unsigned long long)feature];\n" +
    "  for (int32_t slot = 0; slot < compressed_slots; ++slot) for (int32_t feature = 0; feature < " +
    head_dimension.to_string() +
    "; ++feature) prepared_kv[output_base + (unsigned long long)(ordinary_slots + slot) * " +
    head_dimension.to_string() +
    "ULL + (unsigned long long)feature] = compressed_kv[compressed_base + (unsigned long long)slot * " +
    head_dimension.to_string() +
    "ULL + (unsigned long long)feature]; prepared_lengths[batch] = ordinary_slots + compressed_slots;\n" +
    "}\n",
  )
  {
    operation: NonOverlappingSharedKeyValuePrepareBf16V1,
    numerical_contract: BitExactBf16SegmentCopyV1,
    window_size,
    compression_ratio,
    head_dimension,
    maximum_position_count,
    maximum_compressed_slots,
    maximum_key_slots,
    maximum_batches,
    function_symbol,
    source,
  }
}
