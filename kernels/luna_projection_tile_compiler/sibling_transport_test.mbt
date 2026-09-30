///|
test "sibling producer transport is orthogonal to immutable ownership" {
  let scheduled = physical_test_compilation(GatedMlp)
  let layout = width => {
    @physical_tile_ir.OperandLayout::new(
      columns=width,
      vector=8,
      rotations=1,
      row_group=1,
    )
  }
  for
    ownership in [
      @luna_projection_tile_compiler.SplitSiblingPlanes,
      RetainSiblingProducer,
      CoownedSiblingValues,
    ] {
    let ordinary = scheduled
      .refine_physical(
        layout,
        _ => CompletedPrefetch,
        owners_per_group=32,
        sibling_ownership=ownership,
      )
      .unwrap()
    for
      transport in [
        @luna_projection_tile_compiler.SegmentedSiblingTransport,
        IndependentSiblingTransport,
      ] {
      let physical = scheduled
        .refine_physical(
          layout,
          _ => CompletedPrefetch,
          owners_per_group=32,
          sibling_ownership=ownership,
          sibling_transport=transport,
        )
        .unwrap()
      assert_eq(physical.scheduled, ordinary.scheduled)
      assert_eq(physical.sibling_epilogue, ordinary.sibling_epilogue)
      assert_eq(physical.intermediate, ordinary.intermediate)
      let sibling = physical.sibling.unwrap()
      assert_eq(sibling.sibling_transport, transport)
      assert_eq(sibling.distribution, ordinary.sibling.unwrap().distribution)
      assert_eq(sibling.fragments, ordinary.sibling.unwrap().fragments)
      assert_eq(sibling.operands, ordinary.sibling.unwrap().operands)
      assert_eq(sibling.effects.has_pending_transfers(), false)
      assert_eq(
        sibling.effects.overlaps(),
        transport == IndependentSiblingTransport,
      )
    }
  }
}
