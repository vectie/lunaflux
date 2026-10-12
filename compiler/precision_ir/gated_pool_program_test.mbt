///|
fn pool_program() -> @precision_ir.LearnedGatedPoolPrecision raise {
  @precision_ir.LearnedGatedPoolPrecision::new(
    input_width=8,
    feature_width=16,
    ratio=4,
    preceding_window_overlap=true,
    maximum_rows=5,
    maximum_positions=1024,
    norm_epsilon=1.0e-6,
    maximum_state_bytes=1048576L,
    maximum_workspace_bytes=1048576L,
  )
}

///|
test "pool has explicit typed arithmetic and immutable alternatives" {
  let plan = pool_program()
  let changed = plan.with_projection_multiply_add(FusedMultiplyAdd)
  let before = plan.expression(ProjectionUpdate)
  assert_true(
    before.relation_to(changed.expression(ProjectionUpdate)) == DifferentProgram,
  )
  for
    op in [
      @precision_ir.ScoreBias,
      Maximum,
      DenominatorUpdate,
      WeightedUpdate,
      Publish,
      SquareUpdate,
      Inverse,
      Normalize,
    ] {
    assert_true(
      plan.expression(op).relation_to(changed.expression(op)) ==
      IdenticalProgram,
    )
  }
  assert_eq(plan.reduction_extent(ProjectionUpdate), 8)
  assert_eq(plan.reduction_extent(WeightedUpdate), 8)
  assert_eq(plan.reduction_extent(SquareUpdate), 16)
}

///|
test "pool effect graph orders all publication and in place state dependencies" {
  let plan = pool_program()
  let effects = plan.effect_program()
  for i in 0..<4 {
    assert_true(effects.happens_before(i, i + 1))
  }
  assert_true(
    effects.steps()[0]
    .accesses()
    .any(a => a.buffer() == 2 && a.mode() == ReadWrite),
  )
  assert_true(
    effects.steps()[2]
    .accesses()
    .any(a => a.buffer() == 0 && a.mode() == ReadWrite),
  )
  assert_true(
    effects.steps()[2]
    .accesses()
    .any(a => a.buffer() == 1 && a.mode() == ReadWrite),
  )
  assert_true(effects.steps()[4].dependencies().contains(3))
  let sizes = [..plan.state_bytes(), ..plan.frame_bytes()]
  for boundary in plan.replay_boundaries() {
    assert_eq(boundary.capacity_bytes, sizes[boundary.buffer])
    guard boundary.after_stage is Some(stage) else { fail("missing producer") }
    assert_true(
      effects.steps()[stage]
      .accesses()
      .any(a => a.buffer() == boundary.buffer && a.mode() != Read),
    )
  }
}
