///|
fn projection_compiler_dispatch(
  family : @luna_projection_strategy.ProjectionSemanticFamily,
) -> @luna_projection_strategy.ProjectionDispatchPlan raise {
  @luna_projection_strategy.plan_projection_dispatch(
    @luna_projection_strategy.ProjectionShape::new(
      family,
      input_width=1024,
      output_width=if family == LanguageModelHead { 151936 } else { 1024 },
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
  assert_eq(trace.length(), 5)
  assert_true(trace[0].changed())
  assert_true(trace[1].changed())
  assert_false(trace[2].changed())
  assert_false(compiled.schedule().reuse_input_tile())
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
  assert_false(compiled.optimization().trace()[2].changed())
  assert_false(compiled.schedule().reuse_input_tile())
  assert_eq(
    compiled.schedule().loops()[0].components(),
    (ProjectionParallelMap, ProjectionRowAxis, 256, 16),
  )
}

///|
test "functional MLP graph shares pure input and intermediate tile values" {
  let compiled = compile_projection_tile(
    ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(GatedMlp),
      maximum_rows=32,
      maximum_tokens=256,
      demand=AllTokenRows,
    ),
  )
  let bindings = compiled.program().bindings()
  assert_eq(bindings.length(), 9)
  assert_eq(
    bindings[4].expression(),
    ProjectionDot(ProjectionTileValueId(0), ProjectionTileValueId(1)),
  )
  assert_eq(
    bindings[5].expression(),
    ProjectionDot(ProjectionTileValueId(0), ProjectionTileValueId(2)),
  )
  assert_eq(
    bindings[6].expression(),
    ProjectionSiluGate(ProjectionTileValueId(4), ProjectionTileValueId(5)),
  )
  assert_eq(
    bindings[7].expression(),
    ProjectionDot(ProjectionTileValueId(6), ProjectionTileValueId(3)),
  )
  let trace = compiled.optimization().trace()
  assert_eq(trace[3].rule(), FuseSiblingInputTraversals)
  assert_true(trace[3].changed())
  assert_eq(trace[4].rule(), ReuseIntermediateTileAcrossOutputTiles)
  assert_true(trace[4].changed())
  assert_true(compiled.schedule().reuse_input_tile())
  assert_true(compiled.schedule().reuse_intermediate_tile())
  let loops = compiled.schedule().loops()
  assert_eq(loops.length(), 5)
  assert_eq(
    loops[1].components(),
    (ProjectionParallelMap, ProjectionIntermediateColumnAxis, 3072, 16),
  )
  assert_eq(
    loops[4].components(),
    (ProjectionSequentialFold, ProjectionIntermediateReductionAxis, 3072, 16),
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
