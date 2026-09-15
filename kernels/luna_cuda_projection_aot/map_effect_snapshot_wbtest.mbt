///|
test "matrix map effect boundaries preserve full source" {
  let digests : Array[String] = []
  for family in [DenseProjection, LanguageModelHead] {
    let parameters : ProjectionParameters = {
      family,
      input_width: 1024,
      output_width: 80,
      query_width: 0,
      intermediate_width: 0,
    }
    let runtime = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [],
    )
    for
      candidate in runtime
      .dispatch_plan(256)
      .prefill()
      .output_distribution_candidates() {
      if candidate.output_tiles_per_workgroup() != 1 &&
        candidate.output_tiles_per_workgroup() != 8 {
        continue
      }
      for resident in [false, true] {
        if resident && family != LanguageModelHead {
          continue
        }
        let strategy = if resident {
          candidate.with_selected_row_residency(
            stable_id=3000 + candidate.output_tiles_per_workgroup(),
          )
        } else {
          candidate
        }
        let selected = projection_build_runtime_dispatch(
          parameters,
          1024,
          projection_fixture_target(),
          [row_variant_test_record(parameters, strategy)],
        )
        let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
          dispatch=selected.dispatch_plan(256),
          maximum_rows=32,
          maximum_tokens=256,
          demand=if family == LanguageModelHead {
            QueryRowEnds
          } else {
            AllTokenRows
          },
          single_row_numerics=AllowDeterministicDotTree,
        )
        let compilation = @luna_projection_tile_compiler.compile_projection_tile(
          projection_bind_tile_weights(request, parameters),
        )
        let source = projection_matrix_map_source(parameters, compilation)
        assert_false(source.is_empty())
        if resident {
          assert_true(compilation.selected_row_gather() is None)
        }
        digests.push(
          @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(source))),
        )
      }
    }
  }
  debug_inspect(
    digests,
    content=(
      #|[
      #|  "1124ad8fd03e858efcc3348e4fb03d5f66031809a31729ccdbb4d655cd25ff7b",
      #|  "1124ad8fd03e858efcc3348e4fb03d5f66031809a31729ccdbb4d655cd25ff7b",
      #|  "167c2e2fc687c9efcaedfb628f15dac3e677c6c5b5c7a87bc59ee0f6b8add77b",
      #|  "5bf071a8b395553f2d618998ca4abd8fb545c06512a182862b20d7a4c90c79fe",
      #|  "57d13f27fd72e1d55c2fcc97a0f8ffe50428cc9f244320ed4a9995d7590b97c8",
      #|  "c73503a02643f3a319843bda83ef5421a89302eba94ac0769aa81a0d4662a570",
      #|]
    ),
  )
}
