///|
fn valid_selected_attention_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn checked_selected_attention_product(left : Int64, right : Int64) -> Int64? {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    None
  } else {
    Some(left * right)
  }
}

///|
/// Emits a serial correctness kernel over flattened live query rows. ABI:
///
/// 1. I32 step counts `[5]`; `counts[2]` is the live batch count and
///    `counts[3]` the live row count.
/// 2. BF16 query `[maximum_rows,query_head_count,head_dimension]`.
/// 3. I32 query batch ids `[maximum_rows]`.
/// 4. BF16 shared K/V `[maximum_batches,maximum_key_slots,head_dimension]`.
/// 5. I32 cache lengths `[maximum_batches]`.
/// 6. F32 attention sink `[query_head_count]`.
/// 7. I32 joined selected indices `[maximum_rows,selected_width]` with `-1`
///    padding; valid indices are relative to their row's batch.
/// 8. BF16 output `[maximum_rows,query_head_count,head_dimension]`.
///
/// Finite inputs are a precondition. The sink contributes only to the stable
/// softmax denominator. Invalid count, batch, cache-length, or index evidence
/// produces a deterministic zero row without an out-of-bounds access.
pub fn shared_key_value_selected_attention_bf16_source(
  query_head_count~ : Int,
  head_dimension~ : Int,
  selected_width~ : Int,
  maximum_key_slots~ : Int,
  maximum_rows~ : Int,
  maximum_batches~ : Int,
  function_symbol~ : String,
) -> DecoderSelectedSparseAttentionCudaSource raise DecoderSelectedSparseAttentionCudaSourceError {
  if query_head_count <= 0 || query_head_count > 1024 {
    raise InvalidQueryHeadCount
  }
  if head_dimension <= 0 || head_dimension > 65536 {
    raise InvalidHeadDimension
  }
  if selected_width <= 0 || selected_width > 65536 {
    raise InvalidSelectedWidth
  }
  if maximum_key_slots <= 0 || maximum_key_slots > 2147483647 {
    raise InvalidMaximumKeySlots
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if maximum_batches <= 0 || maximum_batches > maximum_rows {
    raise InvalidMaximumBatches
  }
  if !valid_selected_attention_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  guard checked_selected_attention_product(
      maximum_rows.to_int64(),
      query_head_count.to_int64(),
    )
    is Some(row_heads) else {
    raise SizeOverflow
  }
  guard checked_selected_attention_product(row_heads, head_dimension.to_int64())
    is Some(_) else {
    raise SizeOverflow
  }
  guard checked_selected_attention_product(
      maximum_batches.to_int64(),
      maximum_key_slots.to_int64(),
    )
    is Some(batch_slots) else {
    raise SizeOverflow
  }
  guard checked_selected_attention_product(
      batch_slots,
      head_dimension.to_int64(),
    )
    is Some(_) else {
    raise SizeOverflow
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *query, const int32_t *query_batch_ids, const __nv_bfloat16 *key_value, const int32_t *cache_lengths, const float *attention_sink, const int32_t *selected_indices, __nv_bfloat16 *output) {\n" +
    "  const int32_t row = (int32_t)blockIdx.x; if (threadIdx.x != 0 || row >= " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long output_base = (unsigned long long)row * " +
    query_head_count.to_string() +
    "ULL * " +
    head_dimension.to_string() +
    "ULL;\n" +
    "  for (unsigned long long cell = 0; cell < " +
    (query_head_count * head_dimension).to_string() +
    "ULL; ++cell) output[output_base + cell] = __float2bfloat16_rn(0.0f);\n" +
    "  const int32_t batch_count = counts[2]; const int32_t live_rows = counts[3];\n" +
    "  if (batch_count <= 0 || batch_count > " +
    maximum_batches.to_string() +
    " || live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    " || row >= live_rows) return;\n" +
    "  const int32_t batch = query_batch_ids[row]; if (batch < 0 || batch >= batch_count) return;\n" +
    "  const int32_t cache_length = cache_lengths[batch]; if (cache_length < 0 || cache_length > " +
    maximum_key_slots.to_string() +
    ") return;\n" +
    "  const unsigned long long selected_base = (unsigned long long)row * " +
    selected_width.to_string() +
    "ULL; int32_t valid_row = 1;\n" +
    "  for (int32_t rank = 0; rank < " +
    selected_width.to_string() +
    "; ++rank) { const int32_t key = selected_indices[selected_base + (unsigned long long)rank]; if (key < -1 || key >= cache_length) { valid_row = 0; break; } }\n" +
    "  if (!valid_row) return; const float scale = rsqrtf(" +
    head_dimension.to_string() +
    ".0f);\n" +
    "  for (int32_t head = 0; head < " +
    query_head_count.to_string() +
    "; ++head) {\n" +
    "    const unsigned long long query_base = ((unsigned long long)row * " +
    query_head_count.to_string() +
    "ULL + (unsigned long long)head) * " +
    head_dimension.to_string() +
    "ULL; float maximum = attention_sink[head];\n" +
    "    for (int32_t rank = 0; rank < " +
    selected_width.to_string() +
    "; ++rank) { const int32_t key = selected_indices[selected_base + (unsigned long long)rank]; if (key == -1) continue; const unsigned long long kv_base = ((unsigned long long)batch * " +
    maximum_key_slots.to_string() +
    "ULL + (unsigned long long)key) * " +
    head_dimension.to_string() +
    "ULL; float dot = 0.0f; for (int32_t component = 0; component < " +
    head_dimension.to_string() +
    "; ++component) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(query[query_base + (unsigned long long)component]), __bfloat162float(key_value[kv_base + (unsigned long long)component]))); const float score = __fmul_rn(dot, scale); if (score > maximum) maximum = score; }\n" +
    "    float denominator = expf(__fsub_rn(attention_sink[head], maximum));\n" +
    "    for (int32_t rank = 0; rank < " +
    selected_width.to_string() +
    "; ++rank) { const int32_t key = selected_indices[selected_base + (unsigned long long)rank]; if (key == -1) continue; const unsigned long long kv_base = ((unsigned long long)batch * " +
    maximum_key_slots.to_string() +
    "ULL + (unsigned long long)key) * " +
    head_dimension.to_string() +
    "ULL; float dot = 0.0f; for (int32_t component = 0; component < " +
    head_dimension.to_string() +
    "; ++component) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(query[query_base + (unsigned long long)component]), __bfloat162float(key_value[kv_base + (unsigned long long)component]))); denominator = __fadd_rn(denominator, expf(__fsub_rn(__fmul_rn(dot, scale), maximum))); }\n" +
    "    for (int32_t component = 0; component < " +
    head_dimension.to_string() +
    "; ++component) { float numerator = 0.0f; for (int32_t rank = 0; rank < " +
    selected_width.to_string() +
    "; ++rank) { const int32_t key = selected_indices[selected_base + (unsigned long long)rank]; if (key == -1) continue; const unsigned long long kv_base = ((unsigned long long)batch * " +
    maximum_key_slots.to_string() +
    "ULL + (unsigned long long)key) * " +
    head_dimension.to_string() +
    "ULL; float dot = 0.0f; for (int32_t inner = 0; inner < " +
    head_dimension.to_string() +
    "; ++inner) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(query[query_base + (unsigned long long)inner]), __bfloat162float(key_value[kv_base + (unsigned long long)inner]))); const float probability_numerator = expf(__fsub_rn(__fmul_rn(dot, scale), maximum)); numerator = __fadd_rn(numerator, __fmul_rn(probability_numerator, __bfloat162float(key_value[kv_base + (unsigned long long)component]))); } output[query_base + (unsigned long long)component] = __float2bfloat16_rn(__fdiv_rn(numerator, denominator)); }\n" +
    "  }\n}\n",
  )
  {
    operation: SharedKeyValueSelectedAttentionBf16,
    numerical_contract: SerialOrderedF32SinkDenominatorBf16OutputV1,
    query_head_count,
    head_dimension,
    selected_width,
    maximum_key_slots,
    maximum_rows,
    maximum_batches,
    function_symbol,
    source,
  }
}
