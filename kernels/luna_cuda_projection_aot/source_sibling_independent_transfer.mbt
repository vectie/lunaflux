///|
/// Diagnostic full-worker operand traversal, retained for orthogonal transport
/// comparisons. Ownership/fragment geometry is provided by the same immutable
/// physical plan as the segmented producer; neither renderer chooses it.
fn projection_sibling_independent_transfer_source(
  physical : @luna_projection_tile_compiler.ProjectionPhysicalPipeline,
  input_width : Int,
) -> String {
  let operands = physical.operands
  let copies = StringBuilder()
  for
    domain in [
      ("input + (unsigned long long)row_begin * \{input_width}ULL", 0, true),
      (
        "gate_weight + (unsigned long long)column_begin * \{input_width}ULL",
        1,
        false,
      ),
      (
        "up_weight + (unsigned long long)column_begin * \{input_width}ULL",
        2,
        false,
      ),
    ] {
    let (base, operand, guard_rows) = domain
    let region = operands.region(operand).unwrap()
    let (offset, _, _) = region.geometry()
    let destination = if offset == 0 {
      "destination"
    } else {
      "destination + \{offset}"
    }
    if guard_rows {
      copies.write_string("if (!overlap) {\n")
      copies.write_string(
        projection_fold_operand_copy_source(
          "(" + base + ")",
          destination,
          region,
          input_width,
          guard_rows,
          false,
        ),
      )
      copies.write_string("\n} else {\n")
    }
    copies.write_string(
      projection_fold_operand_copy_source(
        "(" + base + ")",
        destination,
        region,
        input_width,
        guard_rows,
        physical.effects.has_pending_transfers(),
      ),
    )
    if guard_rows {
      copies.write_string("\n}\n")
    }
    copies.write_string("\n")
  }
  (
    $|auto stage_operands = [&](int k_base, int stage, bool overlap) {
    $|        __nv_bfloat16 *destination = sibling_storage + stage * \{operands.slot_elements()};
    $|        const int row_start = row_begin, offset = k_base;
    $|\{copies.to_string()}
    $|        \{if physical.effects.has_pending_transfers() { "asm volatile(\"cp.async.commit_group;\" ::: \"memory\");" } else { "" }}
    $|      };
  )
}

///|
fn projection_sibling_transport_source(
  physical : @luna_projection_tile_compiler.ProjectionPhysicalPipeline,
  input_width : Int,
) -> String {
  match physical.sibling_transport {
    SegmentedSiblingTransport =>
      projection_sibling_phased_transfer_source(
        physical.operands,
        physical.transfer_segments,
        input_width,
        asynchronous=physical.effects.has_pending_transfers(),
      )
    IndependentSiblingTransport =>
      projection_sibling_independent_transfer_source(physical, input_width)
    BaselineSiblingTransport =>
      abort("sibling transport must be resolved before device lowering")
  }
}
