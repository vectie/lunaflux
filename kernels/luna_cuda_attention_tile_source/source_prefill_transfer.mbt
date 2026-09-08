///|
/// Cooperative synchronous realization of the same paged row map used by the
/// lookahead pipeline. Kept separately so the baseline remains byte-stable.
fn emit_prefill_paged_key_value_load(
  abi : CudaPagedAttentionTileAbi,
  hoist_page_lookup : Bool,
) -> String {
  let (_, key_value_heads, head_dimension, tokens_per_page) = abi.shape()
  let (_, page_stride_values, physical_page_capacity) = abi.layout()
  let vectors_per_key = head_dimension / 8
  let paged_key_value_load = if hoist_page_lookup {
    "      int physical_page = -1;\n" +
    "      if (key_position < partition_end && key_position < sequence_length &&\n" +
    "          key_position <= maximum_query_position && component == 0) {\n" +
    "        const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "        const int page_slot = page_start + logical_page;\n" +
    "        physical_page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;\n" +
    "        if (physical_page < 0 || (unsigned long long)physical_page >= " +
    physical_page_capacity.to_string() +
    "ULL) {\n" +
    "          atomicExch(&tile_valid, 0);\n" +
    "          physical_page = -1;\n" +
    "        }\n" +
    "      }\n" +
    "      const int page_source_lane =\n" +
    "          (lane / " +
    vectors_per_key.to_string() +
    ") * " +
    vectors_per_key.to_string() +
    ";\n" +
    "      physical_page = __shfl_sync(0xffffffffU, physical_page, page_source_lane);\n" +
    "      if (physical_page >= 0) {\n" +
    "        const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "        const unsigned long long cache_index =\n" +
    "            (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL +\n" +
    "            ((unsigned long long)token_in_page * " +
    key_value_heads.to_string() +
    "ULL + key_value_head) *\n" +
    "            LF_HEAD_DIMENSION + component;\n" +
    "        packed_key = *reinterpret_cast<const uint4 *>(key_cache + cache_index);\n" +
    "        packed_value = *reinterpret_cast<const uint4 *>(value_cache + cache_index);\n" +
    "      }\n"
  } else {
    "      if (key_position < partition_end && key_position < sequence_length &&\n" +
    "          key_position <= maximum_query_position) {\n" +
    "        const int logical_page = key_position / " +
    tokens_per_page.to_string() +
    ";\n" +
    "        const int page_slot = page_start + logical_page;\n" +
    "        const int physical_page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;\n" +
    "        if (physical_page < 0 || (unsigned long long)physical_page >= " +
    physical_page_capacity.to_string() +
    "ULL) {\n" +
    "          atomicExch(&tile_valid, 0);\n" +
    "        } else {\n" +
    "          const int token_in_page = key_position % " +
    tokens_per_page.to_string() +
    ";\n" +
    "          const unsigned long long cache_index =\n" +
    "              (unsigned long long)physical_page * " +
    page_stride_values.to_string() +
    "ULL +\n" +
    "              ((unsigned long long)token_in_page * " +
    key_value_heads.to_string() +
    "ULL + key_value_head) *\n" +
    "              LF_HEAD_DIMENSION + component;\n" +
    "          packed_key = *reinterpret_cast<const uint4 *>(key_cache + cache_index);\n" +
    "          packed_value = *reinterpret_cast<const uint4 *>(value_cache + cache_index);\n" +
    "        }\n" +
    "      }\n"
  }
  paged_key_value_load
}
