///|
test "operand lookahead is deterministic and preserves the intermediate ordered fold" {
  let request = ProjectionTileCompilerRequest::new(
    dispatch=projection_compiler_dispatch(GatedMlp),
    maximum_rows=32,
    maximum_tokens=256,
    demand=AllTokenRows,
  )
  let compiled = compile_projection_tile(request)
  assert_eq(compiled, compile_projection_tile(request))
  let pipeline = compiled.schedule().intermediate_pipeline().unwrap()
  assert_eq(pipeline.stage_count(), 2)
  assert_eq(pipeline.tile_width(), 16)
  assert_eq(pipeline.tile_count(), 192)
  assert_eq(pipeline.transfer_width(), 32)
  assert_eq(
    compiled.schedule().loops()[4].components(),
    (ProjectionSequentialFold, ProjectionIntermediateReductionAxis, 3072, 16),
  )
  assert_true(
    @utf8.decode(compiled.schedule.canonical).contains(
      "intermediate_operand_pipeline=2,16,192,32,64,4,256\n",
    ),
  )
  let dense = compile_projection_tile(
    ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(Dense),
      maximum_rows=32,
      maximum_tokens=256,
      demand=AllTokenRows,
    ),
  )
  assert_eq(dense.schedule().intermediate_pipeline(), None)
}
