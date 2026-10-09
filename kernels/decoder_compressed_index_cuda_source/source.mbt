///|
fn valid_compressed_index_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Emits the exact accepted-domain scalar-reference contract for each live row:
/// `count = floor((query_position + 1) / compression_ratio)` and
/// `indices[row, slot] = cache_offset + slot` for `slot < count`.
///
/// Outputs are a dense row-major I32 matrix
/// `[maximum_rows, maximum_slots_per_row]` plus one I32 count per row. Every
/// non-prefix slot is deterministically `-1`; inactive rows have count zero.
/// Invalid positions, negative offsets, or I32 addition overflow also produce
/// an empty sentinel row without reading or writing out of bounds. Runtime
/// `counts[3]` selects the live prefix. Launch geometry is
/// `[maximum_rows,1,1]` blocks of 256 threads.
pub fn causal_complete_slots_i32_source(
  compression_ratio~ : Int,
  maximum_position_count~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecoderCompressedIndexCudaSource raise DecoderCompressedIndexCudaSourceError {
  if compression_ratio <= 1 {
    raise InvalidCompressionRatio
  }
  if maximum_position_count <= 0 ||
    maximum_position_count > 2147483647 ||
    compression_ratio > maximum_position_count {
    raise InvalidMaximumPositionCount
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_compressed_index_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let maximum_slots_per_row = maximum_position_count / compression_ratio
  let source = @utf8.encode(
    "#include <stdint.h>\n" +
    "#include <limits.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *query_positions, const int32_t *cache_offsets, int32_t *slot_counts, int32_t *slot_indices) {\n" +
    "  const int32_t declared_live_rows = counts[3];\n" +
    "  const int32_t live_rows = (declared_live_rows >= 0 && declared_live_rows <= " +
    maximum_rows.to_string() +
    ") ? declared_live_rows : 0;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  const int32_t lane = (int32_t)threadIdx.x;\n" +
    "  int32_t selected = 0;\n" +
    "  int32_t offset = 0;\n" +
    "  if (row < live_rows) {\n" +
    "    const int32_t position = query_positions[row];\n" +
    "    offset = cache_offsets[row];\n" +
    "    if (position >= 0 && position < " +
    maximum_position_count.to_string() +
    " && offset >= 0) {\n" +
    "      selected = (position + 1) / " +
    compression_ratio.to_string() +
    ";\n" +
    "      if (selected > 0 && offset > INT32_MAX - (selected - 1)) selected = 0;\n" +
    "    }\n" +
    "  }\n" +
    "  if (lane == 0) slot_counts[row] = selected;\n" +
    "  const unsigned long long base = (unsigned long long)row * " +
    maximum_slots_per_row.to_string() +
    "ULL;\n" +
    "  for (int32_t slot = lane; slot < " +
    maximum_slots_per_row.to_string() +
    "; slot += 256) {\n" +
    "    slot_indices[base + (unsigned long long)slot] = slot < selected ? offset + slot : -1;\n" +
    "  }\n" +
    "}\n",
  )
  {
    operation: CausalCompleteSlotsI32,
    numerical_contract: BitExactCausalCompleteSlotsI32V1,
    compression_ratio,
    maximum_position_count,
    maximum_slots_per_row,
    maximum_rows,
    function_symbol,
    source,
  }
}
