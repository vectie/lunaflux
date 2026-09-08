///|
test "matrix pipelines preserve sparse demand and ordered reduction geometry" {
  let families : Array[@luna_projection_strategy.ProjectionSemanticFamily] = [
    Qkv,
    Dense,
    LanguageModelHead,
    GatedMlp,
  ]
  for family in families {
    let selected = family == LanguageModelHead
    let request = ProjectionTileCompilerRequest::new(
      dispatch=projection_compiler_dispatch(family),
      maximum_rows=32,
      maximum_tokens=256,
      demand=if selected { QueryRowEnds } else { AllTokenRows },
    )
    let compiled = compile_projection_tile(request)
    assert_eq(compiled, compile_projection_tile(request))
    if family == GatedMlp {
      assert_eq(compiled.schedule().matrix_pipeline(), None)
    } else {
      let pipeline = compiled.schedule().matrix_pipeline().unwrap()
      assert_eq(pipeline.stage_count(), 2)
      assert_eq(pipeline.tile_width(), 16)
      assert_eq(pipeline.tile_count(), 64)
      assert_eq(pipeline.row_width(), if selected { 16 } else { 32 })
      assert_eq(pipeline.transfer_width(), if selected { 32 } else { 64 })
      assert_eq(pipeline.minimum_rows(), if selected { 2 } else { 256 })
      assert_true(
        @utf8.decode(compiled.schedule.canonical).contains(
          "matrix_operand_pipeline=",
        ),
      )
    }
  }
}

///|
test "matrix pipeline eligibility excludes scalar and reduction tails" {
  for width in [33, 496, 512, 528, 1024, 1088] {
    let dispatch = @luna_projection_strategy.plan_projection_dispatch(
      @luna_projection_strategy.ProjectionShape::new(
        Dense,
        input_width=width,
        output_width=1024,
        intermediate_width=0,
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
    let compiled = compile_projection_tile(
      ProjectionTileCompilerRequest::new(
        dispatch~,
        maximum_rows=32,
        maximum_tokens=256,
        demand=AllTokenRows,
      ),
    )
    assert_eq(
      compiled.schedule().matrix_pipeline() is Some(_),
      width >= 512 && width % 64 == 0,
    )
  }
}
