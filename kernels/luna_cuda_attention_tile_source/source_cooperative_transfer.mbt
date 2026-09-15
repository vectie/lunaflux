///|
fn cuda_cooperative_decode_transfer(
  tokens_per_page : Int,
  physical_page_capacity : Int64,
  page_stride_values : Int64,
  key_value_heads : Int,
) -> String {
  (
    $|    for (int linear = thread;
    $|         linear < LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|         linear += LF_BLOCK_THREADS) {
    $|      const int key = linear / LF_HEAD_DIMENSION;
    $|      const int component = linear % LF_HEAD_DIMENSION;
    $|      __nv_bfloat16 staged_key = __float2bfloat16_rn(0.0f);
    $|      __nv_bfloat16 staged_value = __float2bfloat16_rn(0.0f);
    $|      if (key < active_keys) {
    $|        const int key_position = key_base + key;
    $|        const int page_slot = page_start + key_position / \{tokens_per_page};
    $|        const int physical_page =
    $|            page_slot < page_end ? page_table_page_indices[page_slot] : -1;
    $|        if (physical_page < 0 ||
    $|            (unsigned long long)physical_page >= \{physical_page_capacity}ULL) {
    $|          atomicExch(tile_valid, 0);
    $|        } else {
    $|          const int token_in_page = key_position % \{tokens_per_page};
    $|          const unsigned long long cache_index =
    $|              (unsigned long long)physical_page * \{page_stride_values}ULL +
    $|              ((unsigned long long)token_in_page * \{key_value_heads}ULL +
    $|                  key_value_head) * LF_HEAD_DIMENSION + component;
    $|          staged_key = key_cache[cache_index];
    $|          staged_value = value_cache[cache_index];
    $|        }
    $|      }
    $|      const unsigned long long stage_offset =
    $|          (unsigned long long)stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION +
    $|          linear;
    $|      key_tiles[stage_offset] = staged_key;
    $|      value_tiles[stage_offset] = staged_value;
    $|    }
  )
}
