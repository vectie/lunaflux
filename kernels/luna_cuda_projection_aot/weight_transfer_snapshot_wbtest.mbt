///|
test "composed weight transfer preserves complete matrix pipeline source" {
  let digests : Array[String] = []
  for
    (family, input_width, output_width, query_width) in [
      (QkvProjection, 1024, 4096, 2048),
      (DenseProjection, 2048, 1024, 0),
      (LanguageModelHead, 1024, 151936, 0),
    ] {
    let parameters : ProjectionParameters = {
      family,
      input_width,
      output_width,
      query_width,
      intermediate_width: if family == QkvProjection {
        1024
      } else {
        0
      },
    }
    for maximum_tokens in [32, 128, 1024, 2048] {
      let dispatch = projection_build_runtime_dispatch(
        parameters,
        maximum_tokens,
        projection_fixture_target(),
        [],
      ).dispatch_plan(maximum_tokens)
      let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch~,
        maximum_rows=32,
        maximum_tokens~,
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
      let source = projection_matrix_pipeline_source(
        parameters,
        projection_refine_physical(compilation),
        32,
      )
      assert_true(!source.is_empty())
      digests.push(
        @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(source))),
      )
    }
  }
  debug_inspect(
    digests,
    content=(
      #|[
      #|  "af7f0268d8c35f9fa73c71e171ac8c70e3521ca1d2c85a4b1aa330af165d7cc9",
      #|  "af7f0268d8c35f9fa73c71e171ac8c70e3521ca1d2c85a4b1aa330af165d7cc9",
      #|  "af7f0268d8c35f9fa73c71e171ac8c70e3521ca1d2c85a4b1aa330af165d7cc9",
      #|  "af7f0268d8c35f9fa73c71e171ac8c70e3521ca1d2c85a4b1aa330af165d7cc9",
      #|  "7d137ea3b1f87b7ecc261a300fe8b85661365ad753ea0bbc6d38400c4d2cb426",
      #|  "7d137ea3b1f87b7ecc261a300fe8b85661365ad753ea0bbc6d38400c4d2cb426",
      #|  "7d137ea3b1f87b7ecc261a300fe8b85661365ad753ea0bbc6d38400c4d2cb426",
      #|  "7d137ea3b1f87b7ecc261a300fe8b85661365ad753ea0bbc6d38400c4d2cb426",
      #|  "9a25af0754dff0c41e25428e8746780e1e19a38c44d0b5d54b9193f5572655cb",
      #|  "9a25af0754dff0c41e25428e8746780e1e19a38c44d0b5d54b9193f5572655cb",
      #|  "9a25af0754dff0c41e25428e8746780e1e19a38c44d0b5d54b9193f5572655cb",
      #|  "9a25af0754dff0c41e25428e8746780e1e19a38c44d0b5d54b9193f5572655cb",
      #|]
    ),
  )
}
