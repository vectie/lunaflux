///|
test "matrix pipeline family export preserves projection ABIs" {
  for family in [QkvProjection, DenseProjection, LanguageModelHead] {
    let parameters : ProjectionParameters = {
      family,
      input_width: if family == DenseProjection {
        2048
      } else {
        1024
      },
      output_width: match family {
        QkvProjection => 4096
        DenseProjection => 1024
        _ => 151936
      },
      query_width: if family == QkvProjection {
        2048
      } else {
        0
      },
      intermediate_width: if family == QkvProjection {
        1024
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
    let source = row_variant_primary_test_source(parameters, dispatch)
    assert_true(source.contains("void baseline("))
    assert_eq(source.contains("row_offsets"), family == LanguageModelHead)
    if @env.get_env_var("LUNA_TEST_EXPORT_MATRIX_PIPELINE") == Some("1") {
      let name = match family {
        QkvProjection => "qkv"
        DenseProjection => "output"
        _ => "head"
      }
      println("LUNA_SOURCE_BEGIN \{name}")
      println(source)
      println("LUNA_SOURCE_END")
    }
  }
}
