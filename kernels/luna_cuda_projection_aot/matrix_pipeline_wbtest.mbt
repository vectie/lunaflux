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
    if family != LanguageModelHead {
      // Producer and consumer share a row-preserving permutation. Existing
      // arena capacity and the eight-consumer ordered fold remain unchanged.
      assert_true(source.contains("matrix_storage[15360]"))
      assert_true(source.contains("vector < 768"))
      assert_true(source.contains("operand_index(local_row, k)"))
      assert_true(source.contains("consumer_row = (warp / 4) * 1"))
      assert_true(source.contains("float c[1][8]"))
      assert_true(source.contains("if (warp < 8)"))
      assert_true(source.contains("reinterpret_cast<const unsigned *>"))
    } else {
      // Sparse selected-row demand retains the measured non-regressing
      // matrix consumer until its transport replacement is competitive.
      assert_true(source.contains("matrix_storage[13824]"))
      assert_false(source.contains("operand_index(local_row, k)"))
      assert_true(source.contains("local_row * 48 + k"))
      assert_true(source.contains("consumer_row = (warp / 8) * 1"))
      assert_true(source.contains("wmma::fill_fragment(c[row], 0.0f)"))
    }
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
