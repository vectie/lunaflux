///|
/// Terminal realization of the portable KV interval substitution. Both
/// synchronous LDG and asynchronous copies consume these same addresses.
/// The current view is optional; paged reads remain a valid realization for
/// every schedule because the ABI promises identical committed K/V values.
fn cuda_prefill_key_value_view_address(
  abi : CudaPagedAttentionTileAbi,
  query_owned : Bool,
  dense? : Bool = true,
) -> String {
  let (heads, kv_heads, dimension, page_tokens) = abi.shape()
  let (row_width, page_stride, pages) = abi.layout()
  let (
    position,
    component,
    row_begin,
    row_end,
    positions,
    current,
    page_begin,
    page_indices,
    key_head,
    keys,
    values,
    invalid,
    valid,
  ) = if query_owned {
    (
      "position",
      "c",
      "begin",
      "end",
      "positions",
      "input",
      "page_begin",
      "page_indices",
      "head/\{heads/kv_heads}",
      "keys",
      "values",
      "atomicExch(control,0)",
      "position<key_end",
    )
  } else {
    (
      "key_position", "component", "row_start", "row_end", "query_positions", "rotated_qkv_rows",
      "page_start", "page_table_page_indices", "key_value_head", "key_cache", "value_cache",
      "atomicExch(&tile_valid,0)", "key_position<partition_end && key_position<sequence_length && key_position<=maximum_query_position",
    )
  }
  let current_delta = if dense {
    "(long long)\{position}-(long long)\{positions}[\{row_begin}]"
  } else {
    "0"
  }
  let from_current = if dense {
    "\{valid} && current_delta>=0 && current_delta<\{row_end}-\{row_begin} && \{positions}[\{row_begin}+current_delta]==\{position}"
  } else {
    "false"
  }
  (
    $|      const __nv_bfloat16 *kv_key_source=\{keys}, *kv_value_source=\{values};
    $|      bool kv_source_valid=false;
    $|      const long long current_delta=\{current_delta};
    $|      const bool from_current=\{from_current};
    $|      int kv_page=-1;
    $|      if (\{valid} && !from_current && \{component}==0) {
    $|        kv_page=\{page_indices}[\{page_begin}+\{position}/\{page_tokens}];
    $|        if (kv_page<0 || (unsigned long long)kv_page>=\{pages}ULL) { \{invalid}; kv_page=-1; }
    $|      }
    $|      kv_page=__shfl_sync(0xffffffffU,kv_page,(lane/\{dimension/8})*\{dimension/8});
    $|      if (from_current) {
    $|        const unsigned long long source=(unsigned long long)(\{row_begin}+current_delta)*\{row_width}ULL+
    $|            (\{heads}ULL+\{key_head})*LF_HEAD_DIMENSION+\{component};
    $|        kv_key_source=\{current}+source;
    $|        kv_value_source=kv_key_source+\{kv_heads*dimension};
    $|        kv_source_valid=true;
    $|      } else if (kv_page>=0) {
    $|        const unsigned long long source=(unsigned long long)kv_page*\{page_stride}ULL+
    $|            ((unsigned long long)(\{position}%\{page_tokens})*\{kv_heads}ULL+\{key_head})*LF_HEAD_DIMENSION+\{component};
    $|        kv_key_source=\{keys}+source; kv_value_source=\{values}+source;
    $|        kv_source_valid=true;
    $|      }
  )
}

///|
fn cuda_prefill_view_sync_load(abi : CudaPagedAttentionTileAbi) -> String {
  cuda_prefill_key_value_view_address(abi, false) +
  "\n      if (kv_source_valid) { packed_key=*reinterpret_cast<const uint4 *>(kv_key_source); packed_value=*reinterpret_cast<const uint4 *>(kv_value_source); }\n"
}

///|
fn cuda_prefill_view_async_stage(
  abi : CudaPagedAttentionTileAbi,
  lookahead : Bool,
) -> String {
  (
    $|  auto stage_prefill_tile = [&](int key_base, int slot) {
    $|    auto *stage_keys=key_tiles+slot*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION;
    $|    auto *stage_values=value_tiles+slot*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION;
    $|    for (int vector=thread;vector<key_value_vector_count;vector+=LF_BLOCK_THREADS) {
    $|      const int key=vector*8/LF_HEAD_DIMENSION, component=vector*8%LF_HEAD_DIMENSION;
    $|      const int key_position=key_base+key;
    $|\{cuda_prefill_key_value_view_address(abi, false)}
    $|      const int source_bytes=kv_source_valid?16:0;
    $|      const unsigned ka=(unsigned)__cvta_generic_to_shared(stage_keys+lf_operand_index(key,component,LF_HEAD_DIMENSION));
    $|      const unsigned va=(unsigned)__cvta_generic_to_shared(stage_values+lf_operand_index(key,component,LF_HEAD_DIMENSION));
    $|      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(ka),"l"(kv_key_source),"r"(source_bytes):"memory");
    $|      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(va),"l"(kv_value_source),"r"(source_bytes):"memory");
    $|    }
    $|    asm volatile("cp.async.commit_group;" ::: "memory");
    $|  };
  ) +
  (if lookahead {
    "\n  if (partition_begin<partition_end) stage_prefill_tile(partition_begin,0);\n"
  } else {
    "\n"
  })
}
