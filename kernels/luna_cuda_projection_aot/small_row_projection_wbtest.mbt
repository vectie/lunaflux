///|
test "bounded dense and segmented maps do not stage a dead second row tile" {
  for family in [QkvProjection, DenseProjection] {
    let parameters : ProjectionParameters = {
      family,
      input_width: 128,
      output_width: 256,
      query_width: if family == QkvProjection {
        128
      } else {
        0
      },
      intermediate_width: if family == QkvProjection {
        64
      } else {
        0
      },
    }
    let runtime = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [],
    )
    let dispatch = runtime.dispatch_plan(1024)
    for bound in [1, 2, 4, 8, 16, 17, 32, 1024] {
      let compilation = @luna_projection_tile_compiler.compile_projection_tile(
        @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
          dispatch~,
          maximum_rows=1,
          maximum_tokens=bound,
          demand=AllTokenRows,
          single_row_numerics=AllowDeterministicDotTree,
        ),
      )
      let source = projection_matrix_pipeline_source(parameters, compilation, 1)
      let row_width = if bound <= 16 { 16 } else { 32 }
      assert_true(
        source.contains("pipeline_rows + \{row_width - 1}) / \{row_width}"),
      )
      assert_true(
        source.contains("const bool is_input = element < \{row_width * 64};"),
      )
      assert_true(source.contains("k += 16"))
      assert_true(source.contains("mma.sync.aligned.m16n8k16"))
      assert_true(
        source.contains(
          "if (row_begin + (consumer_row + row) * 16 >= pipeline_rows) break;",
        ),
      )
      // The transport permutation and output map stay authoritative: changing
      // a tile domain must not revert to conflicting packed shared addresses.
      assert_true(source.contains("c ^ ((r & 7) * 8)"))
      assert_true(source.contains("temporary[result_index("))
      assert_false(source.contains("__fadd_rn"))
      assert_false(source.contains("atomicAdd"))
    }
  }
}
