///|
test "materialization distinguishes publication from reusable lifetime release" {
  let scopes : Array[
    @luna_projection_tile_compiler.ProjectionMaterializationScope,
  ] = [ConsumerGroupStorage, WorkgroupStorage]
  for scope in scopes {
    for reused in [false, true] {
      let plan = scope.materialization_effects(reused~)
      if scope == ConsumerGroupStorage {
        assert_true(plan.publication() is ConsumerGroupFence)
      } else {
        assert_true(plan.publication() is WorkgroupFence)
      }
      assert_eq(
        plan.release(),
        if reused {
          Some(plan.publication())
        } else {
          None
        },
      )
    }
  }
}
