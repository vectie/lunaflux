///|
fn projection_compiler_dispatch(
  family : @luna_projection_strategy.ProjectionSemanticFamily,
) -> @luna_projection_strategy.ProjectionDispatchPlan raise {
  @luna_projection_strategy.plan_projection_dispatch(
    @luna_projection_strategy.ProjectionShape::new(
      family,
      input_width=1024,
      output_width=if family == LanguageModelHead { 151936 } else { 1024 },
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
}

///|
test "functional projection compiler prunes unobserved prompt logits" {
  let compiled = compile_projection_tile(
    ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(LanguageModelHead),
      maximum_rows=32,
      maximum_tokens=256,
      demand=QueryRowEnds,
    ),
  )
  assert_eq(compiled.program().demand(), QueryRowEnds)
  assert_eq(compiled.program().bindings().length(), 5)
  assert_eq(
    compiled.program().bindings()[2].expression(),
    ProjectionSelectQueryRowEnds(ProjectionTileValueId(0)),
  )
  let trace = compiled.optimization().trace()
  assert_eq(trace.length(), 3)
  assert_true(trace[0].changed())
  assert_true(trace[1].changed())
  assert_true(trace[2].changed())
  assert_true(compiled.schedule().reuse_input_tile())
  assert_eq(
    compiled.schedule().loops()[0].components(),
    (ProjectionParallelMap, ProjectionRowAxis, 32, 16),
  )
}

///|
test "functional projection compiler preserves all-row dense semantics" {
  let compiled = compile_projection_tile(
    ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(Dense),
      maximum_rows=32,
      maximum_tokens=256,
      demand=AllTokenRows,
    ),
  )
  assert_eq(compiled.program().bindings().length(), 4)
  assert_false(compiled.optimization().trace()[0].changed())
  assert_eq(
    compiled.schedule().loops()[0].components(),
    (ProjectionParallelMap, ProjectionRowAxis, 256, 16),
  )
}

///|
test "row-end demand is rejected for a non-head projection" {
  ignore(
    ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(Dense),
      maximum_rows=32,
      maximum_tokens=256,
      demand=QueryRowEnds,
    ),
  ) catch {
    InvalidProjectionTileCompilation(InvalidDemand) => return
    _ => fail("unexpected projection compiler error")
  }
  fail("non-head row-end demand reached semantic elaboration")
}

///|
test "functional projection compilation is referentially transparent" {
  let request = ProjectionTileCompilerRequest::new(
    dispatch=projection_compiler_dispatch(LanguageModelHead),
    maximum_rows=32,
    maximum_tokens=256,
    demand=QueryRowEnds,
  )
  let first = compile_projection_tile(request)
  let second = compile_projection_tile(request)
  assert_eq(first, second)
  assert_eq(first.digest().as_hex().length(), 64)
}
