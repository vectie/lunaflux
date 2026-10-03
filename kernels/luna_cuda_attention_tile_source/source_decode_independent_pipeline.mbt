///|
/// Two independently retired operand rings. Retain page identities, not an
/// array of wide vector addresses: the latter grows register pressure and can
/// spill when the slot index is dynamic. Aligned page epochs amortize the
/// table lookup across both operands and every vector of the future tile.
fn emit_decode_independent_stage(
  abi : CudaPagedAttentionTileAbi,
  effects : @luna_attention_tile_schedule.AttentionGroupedEffectPlan,
  hoist_page_lookup : Bool,
  retention : @attention_physical_ir.RetainedCopyOwnership,
  vector_elements : Int,
  tile_rows : Int,
  start_alignment : Int,
  storage_index : String,
) -> String {
  let (_, heads, dimensions, page_rows) = abi.shape()
  let (_, stride, capacity) = abi.layout()
  let origins = @attention_physical_ir.PagedCopyOrigin::new(
    page_stride=stride,
    page_capacity=capacity,
    within_page_offset=(heads - 1).to_int64() * dimensions.to_int64(),
  )
  let origin_type = if origins is Some(plan) && plan.is_narrow() {
    "unsigned int"
  } else {
    "unsigned long long"
  }
  let epochs = if hoist_page_lookup && origins is Some(_) {
    @attention_physical_ir.PagedLookupLookahead::new(
      tile_rows~,
      page_rows~,
      start_alignment~,
      owners=32,
    )
  } else {
    None
  }
  let declaration = if epochs is Some(_) {
    "  \{origin_type} future_origin_0 = 0, future_origin_1 = 0;\n" +
    "  int future_valid_0 = 0, future_valid_1 = 0;\n" +
    "  int lookup_epoch_begin = -1, lookup_epoch_page = -1;"
  } else {
    ""
  }
  let prepare = match epochs {
    Some(plan) =>
      "    // Future page epochs span \{plan.tiles_per_epoch()} tiles; both operands share the lookup.\n" +
      "    if (key_base < key_limit && (lookup_epoch_begin < 0 || key_base >= lookup_epoch_begin + \{plan.rows_per_epoch()})) {\n" +
      "      lookup_epoch_begin = key_base;\n" +
      "      const int epoch_slot = page_start + key_base / \{page_rows} + lane;\n" +
      "      lookup_epoch_page = lane * \{page_rows} < key_limit - key_base && epoch_slot < page_end\n" +
      "          ? page_table_page_indices[epoch_slot] : -1;\n" +
      "    }\n" +
      "    const int epoch_owner = (key_base - lookup_epoch_begin) / \{page_rows} + lane;\n" +
      "    const int epoch_page = __shfl_sync(0xffffffffU, lookup_epoch_page, min(31, epoch_owner));\n" +
      "    const int page_valid = epoch_page >= 0 && (unsigned long long)epoch_page < \{capacity}ULL;\n" +
      "    const \{origin_type} page_origin = page_valid\n" +
      "        ? (\{origin_type})((unsigned long long)epoch_page * \{stride}ULL + key_value_head * LF_HEAD_DIMENSION) : 0;\n" +
      "    if (slot == 0) { future_origin_0 = page_origin; future_valid_0 = page_valid; }\n" +
      "    else { future_origin_1 = page_origin; future_valid_1 = page_valid; }\n" +
      "    if (lane < \{tile_rows / page_rows} && lane * \{page_rows} < active_keys && !page_valid) atomicExch(stage_valid + slot, 0);"
    None => ""
  }
  let vector_bytes = vector_elements * 2
  let (vectors, owners, slots) = retention.dimensions()
  let bodies = [
    for operand in 0..<2 => {
      let name = if operand == 0 { "key" } else { "value" }
      let address = if epochs is Some(_) {
        (
          $|      const int page_owner = key / \{page_rows};
          $|      const int page_valid = __shfl_sync(0xffffffffU, slot == 0 ? future_valid_0 : future_valid_1, page_owner);
          $|      const \{origin_type} page_origin = __shfl_sync(0xffffffffU, slot == 0 ? future_origin_0 : future_origin_1, page_owner);
          $|      const int row_valid = key < active_keys && page_valid;
          $|      const unsigned long long cache_index = row_valid
          $|          ? (unsigned long long)page_origin + (key % \{page_rows}) * \{heads * dimensions}ULL + component : 0;
        )
      } else {
        cuda_decode_row_address(
          abi,
          vector_elements,
          hoist_page_lookup,
          "atomicExch(stage_valid + \{operand * 2} + slot, 0)",
        )
      }
      let copies = [
        for vector_slot in 0..<slots => {
          (
            $|    {
            $|      const int vector = thread + \{vector_slot * owners};
            $|      if (vector < \{vectors}) {
            $|      const int linear = vector * \{vector_elements};
            $|      const int key = linear / LF_HEAD_DIMENSION;
            $|      const int component = linear % LF_HEAD_DIMENSION;
            $|\{address}
            $|      const unsigned int destination = (unsigned int)__cvta_generic_to_shared(operand_tile + \{storage_index});
            $|      asm volatile("cp.async.ca.shared.global [%0], [%1], \{vector_bytes}, %2;" ::
            $|          "r"(destination), "l"(\{name}_cache + cache_index), "r"(row_valid ? \{vector_bytes} : 0) : "memory");
            $|      }
            $|    }
          )
        }
      ]
      let prepare_operand = if operand == 0 {
        prepare
      } else if epochs is Some(_) {
        "    const int page_valid = slot == 0 ? future_valid_0 : future_valid_1;\n" +
        "    if (lane < \{tile_rows / page_rows} && lane * \{page_rows} < active_keys && !page_valid) atomicExch(stage_valid + 2 + slot, 0);"
      } else {
        ""
      }
      (
        $|  auto stage_\{name} = [&](int key_base, int slot) {
        $|    const int active_keys = max(0, min(LF_KEY_VALUE_TILE, key_limit - key_base));
        $|    __nv_bfloat16 *operand_tile = stage_arena + (slot * 2 + \{operand}) * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
        $|\{prepare_operand}
        $|\{copies.join("\n")}
        $|    // Empty tail transfers still commit: the readiness law stays four groups deep.
        $|    asm volatile("cp.async.commit_group;" ::: "memory");
        $|  };
      )
    }
  ]
  (
    $|  // Retained copy ownership: \{vectors} vectors, \{owners} owners, \{slots} scalar slots.
    $|\{declaration}
    $|\{bodies.join("\n")}
    $|\{cuda_grouped_effects(effects, InitializeTransfers, "", "  ")}
  )
}
