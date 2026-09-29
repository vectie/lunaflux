///|
/// Stage immutable matrix blocks while consuming the preceding block. CUDA
/// owns copy width, workgroup cooperation, and shared-address conversion.
/// Empty row microtiles are unobserved members of the parallel result map,
/// not reduction terms. Their workgroup-uniform guards leave every live dot's
/// ordered updates and all inter-stage barriers unchanged.
fn projection_fold_pipeline_source(
  pipeline : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  input_base : String,
  weight_base : String,
  stride : Int,
  groups : Int,
  output_width : Int,
) -> String {
  let width = pipeline.transfer_width()
  let rows = pipeline.row_width()
  let layout : CudaProjectionTransportLayout = { columns: width, }
  let operands = @physical_tile_ir.OperandStorage::new(
    layout.plan(),
    [rows, groups * 16],
    owners=groups * 32,
    stages=pipeline.stage_count(),
  ).unwrap()
  let input_region = operands.region(0).unwrap()
  let weight_region = operands.region(1).unwrap()
  let (input_elements, _, _) = weight_region.geometry()
  let slot = operands.slot_elements()
  let body = projection_packed_fold_body(
    pipeline,
    operands,
    rows=rows / 16,
    storage="pipeline_storage",
    row_origin="row_start",
    row_owner="0",
    row_limit="token_count",
    column_owner="warp",
  )
  // Distribute the immutable operand product into two affine producer maps.
  // This removes the per-vector operand tag and pointer selection without
  // changing the ordered reduction or moving reads across materialization.
  // Keep the next slot's global reads independent of the current matrix fold.
  // A synchronous load/store pair blocks this producer at STS on the load's
  // scoreboard even though its destination is the inactive shared slot.
  let input_copy = projection_fold_operand_copy_source(
    "pipeline_a", "destination", input_region, stride, true, true,
  )
  let weight_copy = projection_fold_operand_copy_source(
    "pipeline_b",
    "destination + \{input_elements}",
    weight_region,
    stride,
    false,
    true,
  )
  let source =
    $|  {
    $|    \{projection_operand_address_source(operands.layout())}
    $|    float c[\{rows / 16}][8] = {};
    $|    __shared__ __align__(32) __nv_bfloat16 pipeline_storage[\{operands.allocation_elements()}];
    $|    const __nv_bfloat16 *pipeline_a = \{input_base};
    $|    const __nv_bfloat16 *pipeline_b = \{weight_base};
    $|    auto stage_operands = [&](int offset, int stage) {
    $|      __nv_bfloat16 *destination = pipeline_storage + stage * \{slot};
    $|\{input_copy}
    $|\{weight_copy}
    $|      \{if projection_operand_copy_is_async(width) { "asm volatile(\"cp.async.commit_group;\" ::: \"memory\");" } else { "" }}
    $|    };
    $|\{projection_fold_lifetime_source(pipeline, body, asynchronous=true, completed_on_issue=!projection_operand_copy_is_async(width))}
    $|    #pragma unroll
    $|    for (int row = 0; row < \{rows / 16}; ++row) {
    $|      if (row_start + row * 16 >= token_count) break;
    $|\{projection_register_exchange_source(output_width)}
    $|    }
    $|  }
    $|
  source
}

///|
/// Specialize one copy domain before CUDA rendering. Invalid rows are
/// zero-filled; producer and matrix consumer use one shared permutation.
fn projection_fold_operand_copy_source(
  base : String,
  destination : String,
  region : @physical_tile_ir.OperandRegion,
  stride : Int,
  guard_rows : Bool,
  asynchronous : Bool,
) -> String {
  let (_, _, width) = region.geometry()
  let validity = if guard_rows {
    "const bool valid = row_start + row < token_count;"
  } else {
    "const bool valid = true;"
  }
  let source =
    $|      \{projection_physical_transfer_map_loop(region.ownership())} {
    $|        const int row = vector / \{width / 8}, k = (vector % \{width / 8}) * 8;
    $|        \{validity}
    $|        const __nv_bfloat16 *source = valid ? \{base} + (unsigned long long)row * \{stride}ULL + offset + k : \{base};
    $|        const unsigned address = (unsigned)__cvta_generic_to_shared(\{destination} + operand_index(row, k));
    $|        \{projection_operand_transfer_source(width, asynchronous~)}
    $|      }
  source
}
