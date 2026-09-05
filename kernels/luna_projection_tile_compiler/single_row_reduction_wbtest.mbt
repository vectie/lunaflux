///|
fn reduction_test_compile(
  family : @luna_projection_strategy.ProjectionSemanticFamily,
  lanes : Int,
  input_width : Int,
  numerics : ProjectionSingleRowNumerics,
) -> ScheduledProjectionTileCompilation raise {
  compile_projection_tile(
    ProjectionTileCompilerRequest::new(
      dispatch=@luna_projection_strategy.plan_projection_dispatch(
        @luna_projection_strategy.ProjectionShape::new(
          family,
          input_width~,
          output_width=128,
          intermediate_width=if family == GatedMlp { 256 } else { 0 },
        ),
        32,
        @luna_projection_strategy.ProjectionCapabilities::new(
          subgroup_width=lanes,
          matrix_tile=Some(
            @luna_projection_strategy.ProjectionTile::new(
              rows=16,
              columns=16,
              reduction=16,
            ),
          ),
        ),
      ),
      maximum_rows=4,
      maximum_tokens=32,
      demand=AllTokenRows,
      single_row_numerics=numerics,
    ),
  )
}

///|
test "single-row reassociation is explicit and bound before scheduling" {
  let ordered = reduction_test_compile(Dense, 32, 128, PreserveDotOrder)
  let tree = reduction_test_compile(Dense, 32, 128, AllowDeterministicDotTree)
  assert_eq(ordered.schedule().single_row_reduction(), OrderedDot)
  assert_eq(tree.schedule().single_row_reduction(), StridedPairwiseDot(32))
  assert_not_eq(ordered.program().digest(), tree.program().digest())
  assert_not_eq(ordered.schedule().digest(), tree.schedule().digest())
  assert_not_eq(ordered.digest(), tree.digest())
  assert_eq(
    tree,
    reduction_test_compile(Dense, 32, 128, AllowDeterministicDotTree),
  )
  // The default API must not silently grant reassociation.
  let default_plan = compile_projection_tile(
    ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(Dense),
      maximum_rows=32,
      maximum_tokens=256,
      demand=AllTokenRows,
    ),
  )
  assert_eq(default_plan.schedule().single_row_reduction(), OrderedDot)
}

///|
test "dot-tree schedules depend on capabilities not model or vendor names" {
  let families : Array[@luna_projection_strategy.ProjectionSemanticFamily] = [
    Dense,
    Qkv,
    GatedMlp,
    LanguageModelHead,
  ]
  for family in families {
    for lanes in [16, 32, 64] {
      let plan = reduction_test_compile(
        family,
        lanes,
        128,
        AllowDeterministicDotTree,
      )
      assert_eq(
        plan.schedule().single_row_reduction(),
        StridedPairwiseDot(lanes),
      )
    }
  }
  let narrow = reduction_test_compile(Dense, 16, 128, AllowDeterministicDotTree)
  let wide = reduction_test_compile(Dense, 64, 128, AllowDeterministicDotTree)
  assert_eq(narrow.program().digest(), wide.program().digest())
  assert_not_eq(narrow.schedule().digest(), wide.schedule().digest())
}

///|
test "unavailable subgroup strategy retains an ordered fold" {
  let plan = reduction_test_compile(Dense, 32, 127, AllowDeterministicDotTree)
  assert_eq(plan.schedule().single_row_reduction(), OrderedDot)
}
