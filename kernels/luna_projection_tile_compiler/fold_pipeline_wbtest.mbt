///|
test "transfer grouping tiles short and irregular reductions exactly" {
  for extent in [16, 32, 48, 64, 80, 128, 256, 320, 384, 512, 1024] {
    for maximum_tiles in [2, 4, 8] {
      let width = projection_transfer_width(extent, 16, maximum_tiles)
      assert_true(width >= 16 && width <= 16 * maximum_tiles)
      assert_eq(extent % width, 0)
      assert_eq(width % 16, 0)
    }
  }
  assert_eq(projection_transfer_width(48, 16, 4), 16)
  assert_eq(projection_transfer_width(32, 16, 4), 32)
  assert_eq(projection_transfer_width(320, 16, 8), 64)
  assert_eq(projection_transfer_width(384, 16, 8), 128)
}

///|
test "lookahead profitability retains short folds and wide products" {
  for extent in [1024, 2048, 2112, 2176, 3072] {
    for rows in [8, 16, 32, 1024] {
      for outputs in [1, 2, 4, 8] {
        let tiles = projection_operand_window_tiles(extent, rows, outputs)
        assert_eq(
          tiles,
          if extent >= 2048 && rows <= 16 && outputs <= 2 {
            8
          } else {
            4
          },
        )
        let width = projection_transfer_width(extent, 16, tiles)
        assert_eq(extent % width, 0)
        assert_eq(width % 16, 0)
        assert_true(width <= tiles * 16)
      }
    }
  }
  assert_eq(projection_transfer_width(2112, 16, 8), 64)
  assert_eq(projection_transfer_width(2176, 16, 8), 128)
}

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
  assert_eq(pipeline.transfer_width(), 64)
  assert_eq(pipeline.minimum_rows(), 2)
  let siblings = compiled.schedule().sibling_pipeline().unwrap()
  assert_eq(siblings.tile_count(), 64)
  assert_eq(siblings.row_width(), 64)
  assert_eq(siblings.transfer_width(), 32)
  assert_eq(siblings.minimum_rows(), 2)
  assert_eq(
    siblings.output_tiles(),
    compiled.schedule().output_tiles_per_workgroup() / 4,
  )
  assert_true(
    @utf8.decode(compiled.schedule.canonical).contains(
      "sibling_operand_pipeline=",
    ),
  )
  assert_true(
    @utf8.decode(compiled.schedule.canonical).contains(
      "sibling_product_distribution=2,4,2\n",
    ),
  )
  assert_eq(
    compiled.schedule().loops()[4].components(),
    (ProjectionSequentialFold, ProjectionIntermediateReductionAxis, 3072, 16),
  )
  assert_true(
    @utf8.decode(compiled.schedule.canonical).contains(
      "intermediate_operand_pipeline=2,16,192,64,32,4,2\n",
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
  assert_eq(dense.schedule().sibling_pipeline(), None)
}
