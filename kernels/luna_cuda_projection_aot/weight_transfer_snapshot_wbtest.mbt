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
        parameters, compilation, 32,
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
      #|  "85259d5093abcadd168e6a66f8dfd84c7eb97690aa1ffbd02771706545766bbc",
      #|  "85259d5093abcadd168e6a66f8dfd84c7eb97690aa1ffbd02771706545766bbc",
      #|  "85259d5093abcadd168e6a66f8dfd84c7eb97690aa1ffbd02771706545766bbc",
      #|  "85259d5093abcadd168e6a66f8dfd84c7eb97690aa1ffbd02771706545766bbc",
      #|  "e2ca7f23433dffb5f05ddc249336ce6b0ecbe868a0422c601ea24621afbaf916",
      #|  "e2ca7f23433dffb5f05ddc249336ce6b0ecbe868a0422c601ea24621afbaf916",
      #|  "e2ca7f23433dffb5f05ddc249336ce6b0ecbe868a0422c601ea24621afbaf916",
      #|  "e2ca7f23433dffb5f05ddc249336ce6b0ecbe868a0422c601ea24621afbaf916",
      #|  "8dd52fbf38a6314832d406f0e659705e7711ba626e71b689f3fc5f3211865367",
      #|  "8dd52fbf38a6314832d406f0e659705e7711ba626e71b689f3fc5f3211865367",
      #|  "8dd52fbf38a6314832d406f0e659705e7711ba626e71b689f3fc5f3211865367",
      #|  "8dd52fbf38a6314832d406f0e659705e7711ba626e71b689f3fc5f3211865367",
      #|]
    ),
  )
}
