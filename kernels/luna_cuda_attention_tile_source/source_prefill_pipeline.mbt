///|
/// Double buffering realizes a pure ordered fold with a one-element lookahead.
/// The future K/V transfer owns a disjoint ring slot until its next consumer;
/// current QK/softmax/PV order and BF16 rounding are unchanged. All lanes issue
/// equal-sized transfers, including zero-filled tail vectors, before commit.
fn emit_prefill_async_stage(
  abi : CudaPagedAttentionTileAbi,
  hoist_page_lookup : Bool,
  lookahead : Bool,
) -> String {
  let (_, heads, dimension, page_tokens) = abi.shape()
  let (_, stride, capacity) = abi.layout()
  let lookup_lane = if hoist_page_lookup { " && component == 0" } else { "" }
  let page_broadcast = if hoist_page_lookup {
    "      page = __shfl_sync(0xffffffffU, page, (lane / " +
    (dimension / 8).to_string() +
    ") * " +
    (dimension / 8).to_string() +
    ");\n"
  } else {
    ""
  }
  (
    $|  auto stage_prefill_tile = [&](int key_base, int slot) {
    $|    __nv_bfloat16 *stage_keys = key_tiles + slot * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    __nv_bfloat16 *stage_values = value_tiles + slot * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    for (int vector = thread; vector < key_value_vector_count; vector += LF_BLOCK_THREADS) {
    $|      const int linear = vector * LF_VALUES_PER_VECTOR;
    $|      const int key = linear / LF_HEAD_DIMENSION;
    $|      const int component = linear % LF_HEAD_DIMENSION;
    $|      const int position = key_base + key;
    $|      int page = -1;
    $|      if (position < partition_end && position < sequence_length &&
    $|          position <= maximum_query_position\{lookup_lane}) {
    $|        const int page_slot = page_start + position / \{page_tokens};
    $|        page = page_slot < page_end ? page_table_page_indices[page_slot] : -1;
    $|        if (page < 0 || (unsigned long long)page >= \{capacity}ULL) {
    $|          atomicExch(&tile_valid, 0);
    $|          page = -1;
    $|        }
    $|      }
    $|\{page_broadcast}
    $|      unsigned long long cache_index = 0;
    $|      const int source_bytes = page >= 0 ? 16 : 0;
    $|      if (page >= 0)
    $|        cache_index = (unsigned long long)page * \{stride}ULL +
    $|            ((unsigned long long)(position % \{page_tokens}) * \{heads}ULL + key_value_head) *
    $|                LF_HEAD_DIMENSION + component;
    $|      const unsigned int key_address = (unsigned int)__cvta_generic_to_shared(stage_keys + lf_operand_index(key, component, LF_HEAD_DIMENSION));
    $|      const unsigned int value_address = (unsigned int)__cvta_generic_to_shared(stage_values + lf_operand_index(key, component, LF_HEAD_DIMENSION));
    $|      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
    $|          "r"(key_address), "l"(key_cache + cache_index), "r"(source_bytes) : "memory");
    $|      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
    $|          "r"(value_address), "l"(value_cache + cache_index), "r"(source_bytes) : "memory");
    $|    }
    $|    asm volatile("cp.async.commit_group;" ::: "memory");
    $|  };
  ) +
  (if lookahead {
    "\n  if (partition_begin < partition_end) stage_prefill_tile(partition_begin, 0);\n"
  } else {
    "\n"
  })
}
