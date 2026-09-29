///|
fn physical_test_compilation(
  family : @luna_projection_strategy.ProjectionSemanticFamily,
) -> @luna_projection_tile_compiler.ScheduledProjectionTileCompilation raise {
  let dispatch = @luna_projection_strategy.plan_projection_dispatch(
    @luna_projection_strategy.ProjectionShape::new(
      family,
      input_width=1024,
      output_width=1024,
      intermediate_width=if family == GatedMlp { 3072 } else { 0 },
    ),
    256,
    @luna_projection_strategy.ProjectionCapabilities::new(
      subgroup_width=32,
      matrix_tile=Some(
        @luna_projection_strategy.ProjectionTile::new(
          rows=16,
          columns=16,
          reduction=16,
        ),
      ),
    ),
  )
  @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=32,
      maximum_tokens=256,
      demand=if family == LanguageModelHead {
        QueryRowEnds
      } else {
        AllTokenRows
      },
    ),
  )
}

///|
test "physical refinement retains all producer consumer and scalar laws" {
  let families : Array[@luna_projection_strategy.ProjectionSemanticFamily] = [
    Dense,
    Qkv,
    GatedMlp,
    LanguageModelHead,
  ]
  for family in families {
    let scheduled = physical_test_compilation(family)
    let layout = width => {
      @physical_tile_ir.OperandLayout::new(
        columns=width,
        vector=8,
        rotations=1,
        row_group=1,
      )
    }
    let first = scheduled
      .refine_physical(layout, _ => OverlappedTransfer, owners_per_group=32)
      .unwrap()
    let second = scheduled
      .refine_physical(layout, _ => OverlappedTransfer, owners_per_group=32)
      .unwrap()
    assert_eq(first, second)
    assert_eq(first.scheduled, scheduled)
    assert_eq(first.scalar.geometry().0, 1024)
    if family == GatedMlp {
      assert_eq(first.sibling_scalar.unwrap().geometry().1, 2)
      assert_eq(first.down_scalar.unwrap().geometry().1, 4)
      assert_eq(first.ordered_down.unwrap().geometry(), (3072, 1, 1))
      let sibling = first.sibling.unwrap()
      assert_eq(
        sibling.operands.region(2).unwrap().geometry().2,
        sibling.schedule.transfer_width(),
      )
      assert_true(!sibling.transfer_segments.is_empty())
    }
    for physical in [first.matrix, first.sibling, first.intermediate] {
      if physical is Some(physical) {
        assert_eq(
          physical.fragments.geometry().2,
          physical.operands.layout().geometry().0,
        )
        assert_eq(
          physical.effects.geometry().0,
          physical.schedule.stage_count(),
        )
        assert_eq(
          physical.operands.allocation_elements(),
          physical.operands.slot_elements() * physical.schedule.stage_count(),
        )
      }
    }
  }
}

///|
test "required physical plans cannot silently disappear on layout rejection" {
  let families : Array[@luna_projection_strategy.ProjectionSemanticFamily] = [
    Dense,
    GatedMlp,
    LanguageModelHead,
  ]
  for family in families {
    let scheduled = physical_test_compilation(family)
    assert_true(
      scheduled.refine_physical(
        _ => None,
        _ => OverlappedTransfer,
        owners_per_group=32,
      )
      is None,
    )
    assert_true(
      scheduled.refine_physical(
        width => {
          @physical_tile_ir.OperandLayout::new(
            columns=width + 16,
            vector=8,
            rotations=1,
            row_group=1,
          )
        },
        _ => OverlappedTransfer,
        owners_per_group=32,
      )
      is None,
    )
    assert_true(
      scheduled.refine_physical(
        width => {
          @physical_tile_ir.OperandLayout::new(
            columns=width,
            vector=8,
            rotations=1,
            row_group=1,
          )
        },
        _ => OverlappedTransfer,
        owners_per_group=0,
      )
      is None,
    )
  }
}
