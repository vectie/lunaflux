///|
/// Two-stage realization of the schedule's asynchronous shared-memory map.
/// Only storage lifetimes change: the ordered softmax fold remains untouched.
fn emit_decode_async_stage(
  abi : CudaPagedAttentionTileAbi,
  effects : @luna_attention_tile_schedule.AttentionGroupedEffectPlan,
  hoist_page_lookup : Bool,
  retention : @attention_physical_ir.RetainedCopyOwnership,
  vector_elements? : Int = 4,
  tile_rows~ : Int,
  start_alignment~ : Int,
) -> String {
  let (_, _, _, page_rows) = abi.shape()
  let epochs = if hoist_page_lookup {
    @attention_physical_ir.PagedTileEpochs::new(
      tile_rows~,
      page_rows~,
      start_alignment~,
      owners=32,
    )
  } else {
    None
  }
  let page_acquire = match epochs {
    Some(plan) =>
      "    // Aligned page epochs: one table load per page and subgroup.\n" +
      "    const int epoch_slot = page_start + key_base / \{page_rows} + lane;\n" +
      "    const int epoch_page = lane < \{plan.page_count()} && lane * \{page_rows} < active_keys && epoch_slot < page_end\n" +
      "        ? page_table_page_indices[epoch_slot] : -1;"
    None => ""
  }
  let retained_page = if epochs is Some(_) {
    "      const int retained_page = __shfl_sync(0xffffffffU, epoch_page, key / \{page_rows});"
  } else {
    ""
  }
  let address = cuda_decode_row_address(
    abi,
    vector_elements,
    hoist_page_lookup,
    "atomicExch(stage_valid + slot, 0)",
    retained_page~,
  )
  let vector_bytes = vector_elements * 2
  let (vectors, owners, slots) = retention.dimensions()
  let key_copies = Array::make(slots, "")
  let value_copies = Array::make(slots, "")
  for slot in 0..<slots {
    // The binding name and vector offset are compile-time constants. The
    // inactive tail guard is uniform inside each participating subgroup.
    key_copies[slot] = (
      $|    unsigned long long retained_address_\{slot} = 0;
      $|    int retained_size_\{slot} = 0;
      $|    {
      $|      const int vector = thread + \{slot * owners};
      $|      if (vector < \{vectors}) {
      $|      const int linear = vector * \{vector_elements};
      $|      const int key = linear / LF_HEAD_DIMENSION;
      $|      const int component = linear % LF_HEAD_DIMENSION;
      $|\{address}
      $|      const int source_bytes = row_valid ? \{vector_bytes} : 0;
      $|      const unsigned int key_address = (unsigned int)__cvta_generic_to_shared(key_tile + linear);
      $|      retained_address_\{slot} = cache_index;
      $|      retained_size_\{slot} = source_bytes;
      $|      asm volatile("cp.async.ca.shared.global [%0], [%1], \{vector_bytes}, %2;" ::
      $|          "r"(key_address), "l"(key_cache + cache_index), "r"(source_bytes) : "memory");
      $|      }
      $|    }
    )
    value_copies[slot] = (
      $|    {
      $|      const int vector = thread + \{slot * owners};
      $|      if (vector < \{vectors}) {
      $|      const unsigned int value_address = (unsigned int)__cvta_generic_to_shared(value_tile + vector * \{vector_elements});
      $|      asm volatile("cp.async.ca.shared.global [%0], [%1], \{vector_bytes}, %2;" ::
      $|          "r"(value_address), "l"(value_cache + retained_address_\{slot}), "r"(retained_size_\{slot}) : "memory");
      $|      }
      $|    }
    )
  }
  (
    $|  auto stage_tile = [&](int key_base, int slot) {
    $|    const int active_keys = min(LF_KEY_VALUE_TILE, key_limit - key_base);
    $|    __nv_bfloat16 *key_tile = stage_arena + slot * 2 * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    __nv_bfloat16 *value_tile = key_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|\{page_acquire}
    $|    // Retained copy ownership: \{vectors} vectors, \{owners} owners, \{slots} scalar slots.
    $|\{key_copies.join("\n")}
    $|    asm volatile("cp.async.commit_group;" ::: "memory");
    $|\{value_copies.join("\n")}
    $|    asm volatile("cp.async.commit_group;" ::: "memory");
    $|  };
    $|\{cuda_grouped_effects(effects, InitializeTransfers, "", "  ")}
  )
}

///|
fn emit_decode_pipeline_iteration(
  effects : @luna_attention_tile_schedule.AttentionGroupedEffectPlan,
) -> String {
  (
    $|    const int slot = ((key_base - key_begin) / LF_KEY_VALUE_TILE) & 1;
    $|    // K is ready before V: QK does not depend on the value transfer.
    $|\{cuda_grouped_effects(effects, AcquireKeyOperands, "", "    ")}
    $|    key_tile = stage_arena + slot * 2 * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|    value_tile = key_tile + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
  )
}
