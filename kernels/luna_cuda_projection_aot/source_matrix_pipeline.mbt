///|
/// Lower an ordered fold over immutable operands. Gather/scatter implements
/// sparse result demand; segmented weight lookup implements Q/K/V concatenation.
fn projection_matrix_pipeline_source(
  parameters : ProjectionParameters,
  compilation : @luna_projection_tile_compiler.PhysicalProjectionTileCompilation,
  maximum_rows : Int,
) -> String {
  guard compilation.matrix is Some(physical) else { return "" }
  let plan = physical.schedule
  let selected = compilation.schedule().demand() == QueryRowEnds
  let rows = plan.row_width()
  let columns = plan.output_tiles() * 16
  let width = plan.transfer_width()
  let input_width = parameters.input_width
  let output_width = parameters.output_width
  // The row-preserving permutation is compact. Both producer and consumer
  // use that map; an unrelated padded-row allocation only wastes storage.
  let storage_plan = physical.operands
  let input_region = storage_plan.region(0).unwrap()
  let weight_region = storage_plan.region(1).unwrap()
  let slot = storage_plan.slot_elements()
  let weight_plan = compilation.matrix_weights.unwrap()
  let (input_elements, _, _) = weight_region.geometry()
  let distribution = physical.distribution
  let fold_rows = distribution.rows_per_group()
  let consumers = distribution.consumer_groups()
  let row_count = if selected { "counts[2]" } else { "token_count" }
  let input_row = if selected {
    "(logical_row < pipeline_rows ? row_offsets[logical_row + 1] - 1 : -1)"
  } else {
    "logical_row"
  }
  let weight_selection = if parameters.family == QkvProjection {
    let (binding, branch) = projection_weight_selection_source(
      weight_plan.row_selection(),
      ["query_weight", "key_weight", "value_weight"],
      "logical_column",
      "base",
      "weight_row",
    )
    binding + "\n          " + branch + "\n"
  } else {
    "const __nv_bfloat16 *base = weight; const int weight_row = logical_column;\n"
  }
  let input_copy =
    $|        \{projection_physical_transfer_map_loop(input_region.ownership())} {
    $|          const int local_row = vector / \{width / 8}, k = (vector % \{width / 8}) * 8;
    $|          const int logical_row = row_begin + local_row;
    $|          const int token_row = \{input_row};
    $|          const bool valid = logical_row < pipeline_rows && token_row >= 0 && token_row < token_count;
    $|          const __nv_bfloat16 *source = valid ? input + (unsigned long long)token_row * \{input_width}ULL + k_base + k : input;
    $|          const unsigned address = (unsigned)__cvta_generic_to_shared(destination + operand_index(local_row, k));
    $|          if (overlap) {
    $|            \{projection_operand_transfer_source(asynchronous=physical.effects.has_pending_transfers())}
    $|          } else {
    $|            \{projection_operand_transfer_source(asynchronous=false)}
    $|          }
    $|        }
  let weight_copy =
    $|        \{projection_physical_transfer_map_loop(weight_region.ownership())} {
    $|          const int local_row = vector / \{weight_plan.row_vectors()}, k = (vector % \{weight_plan.row_vectors()}) * 8;
    $|          const int logical_column = column_begin + local_row;
    $|          \{weight_selection}
    $|          const bool valid = \{if output_width % columns == 0 { "true" } else { "logical_column < " + output_width.to_string() }};
    $|          const __nv_bfloat16 *source = valid ? base + (unsigned long long)weight_row * \{input_width}ULL + k_base + k : base;
    $|          const unsigned address = (unsigned)__cvta_generic_to_shared(destination + \{input_elements} + operand_index(local_row, k));
    $|          \{projection_operand_transfer_source(asynchronous=physical.effects.has_pending_transfers())}
    $|        }
  let fragment_declarations =
    $|      float c[\{fold_rows}][8] = {};
  // Selection changes logical row addresses, not matrix-fragment ownership.
  // Both row maps lower through the same packed transport permutation.
  let body = projection_storage_fold_body(
    physical,
    storage="matrix_storage",
    consumers~,
  )
  let lifetime = projection_fold_lifetime_source(
    physical,
    body,
    phase_argument=true,
  )
  let source =
    $|  const int pipeline_rows = \{row_count};
    $|  if (pipeline_rows >= \{plan.minimum_rows()}\{if selected { " && pipeline_rows <= " + maximum_rows.to_string() } else { "" }}) {
    $|    __shared__ __align__(32) __nv_bfloat16 matrix_storage[\{storage_plan.allocation_elements()}];
    $|    \{projection_operand_address_source(storage_plan.layout())}
    $|    const int column_groups = \{1 + (output_width - 1) / columns};
    $|    const int work_count = ((pipeline_rows + \{rows - 1}) / \{rows}) * column_groups;
    $|    for (int work = blockIdx.x; work < work_count; work += gridDim.x) {
    $|      const int row_begin = (work / column_groups) * \{rows};
    $|      const int column_begin = (work % column_groups) * \{columns};
    $|      const int consumer_row = (warp / \{distribution.output_groups()}) * \{fold_rows};
    $|      const int consumer_column = warp % \{distribution.output_groups()};
    $|      auto stage_operands = [&](int k_base, int stage, bool overlap) {
    $|        __nv_bfloat16 *destination = matrix_storage + stage * \{slot};
    $|\{input_copy}
    $|\{weight_copy}
    $|        \{if physical.effects.has_pending_transfers() { "asm volatile(\"cp.async.commit_group;\" ::: \"memory\");" } else { "" }}
    $|      };
    $|\{fragment_declarations}
    $|\{lifetime}
    $|      if (warp < \{consumers}) {
    $|        #pragma unroll
    $|        for (int row = 0; row < \{fold_rows}; ++row) {
    $|          if (row_begin + (consumer_row + row) * 16 >= pipeline_rows) break;
    $|          // Every accumulator element has one owner. Store that value
    $|          // directly instead of materializing and rereading a result tile.
    $|          #pragma unroll
    $|          for (int item = 0; item < 8; ++item) {
    $|            const int logical_row = row_begin + (consumer_row + row) * 16 + lane / 4 + (item % 4 / 2) * 8;
    $|            const int token_row = \{input_row};
    $|            const int column = column_begin + consumer_column * 16 + lane % 4 * 2 + item % 2 + item / 4 * 8;
    $|            if (logical_row < pipeline_rows && token_row >= 0 && token_row < token_count && column < \{output_width}) output[(unsigned long long)token_row * \{output_width}ULL + column] = __float2bfloat16_rn(c[row][item]);
    $|          }
    $|        }
    $|      }
    $|    }
    $|    return;
    $|  }
    $|
  source
}
