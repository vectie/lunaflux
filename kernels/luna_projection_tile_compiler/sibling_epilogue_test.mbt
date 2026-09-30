///|
test "sibling epilogue ownership is pure and preserves ordered reductions" {
  let scheduled = physical_test_compilation(GatedMlp)
  let layout = width => {
    @physical_tile_ir.OperandLayout::new(
      columns=width,
      vector=8,
      rotations=1,
      row_group=1,
    )
  }
  let baseline = scheduled
    .refine_physical(layout, _ => CompletedPrefetch, owners_per_group=32)
    .unwrap()
  for
    ownership in [
      @luna_projection_tile_compiler.SplitSiblingPlanes,
      RetainSiblingProducer,
      CoownedSiblingValues,
    ] {
    let first = scheduled
      .refine_physical(
        layout,
        _ => CompletedPrefetch,
        owners_per_group=32,
        sibling_ownership=ownership,
      )
      .unwrap()
    let second = scheduled
      .refine_physical(
        layout,
        _ => CompletedPrefetch,
        owners_per_group=32,
        sibling_ownership=ownership,
      )
      .unwrap()
    assert_eq(first, second)
    assert_eq(first.scheduled, baseline.scheduled)
    assert_eq(first.sibling_scalar, baseline.sibling_scalar)
    assert_eq(first.down_scalar, baseline.down_scalar)
    let epilogue = first.sibling_epilogue.unwrap()
    assert_eq(
      epilogue.plane_elements(),
      first.sibling.unwrap().schedule.row_width() *
      first.sibling.unwrap().schedule.output_tiles() *
      16,
    )
    let expected_planes = if first.sibling.unwrap().sibling_reuse {
      if ownership == SplitSiblingPlanes {
        2
      } else {
        1
      }
    } else {
      0
    }
    assert_eq(epilogue.result_planes(), expected_planes)
    assert_eq(epilogue.effects() is Some(_), expected_planes != 0)
  }
}

///|
test "dense projection cannot manufacture a sibling materialization" {
  let scheduled = physical_test_compilation(Dense)
  let physical = scheduled
    .refine_physical(
      width => {
        @physical_tile_ir.OperandLayout::new(
          columns=width,
          vector=8,
          rotations=1,
          row_group=1,
        )
      },
      _ => CompletedPrefetch,
      owners_per_group=32,
    )
    .unwrap()
  assert_eq(physical.sibling_epilogue, None)
  assert_eq(
    physical.matrix.unwrap().sibling_epilogue(RetainSiblingProducer),
    None,
  )
}
