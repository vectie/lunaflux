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
    assert_false(source.contains("wmma::"))
    assert_eq(source.contains("row_offsets"), family == LanguageModelHead)
    if family != LanguageModelHead {
      // Producer and consumer share a row-preserving permutation. Existing
      // Compact storage follows that same map; no padded allocation remains.
      assert_true(source.contains("matrix_storage[12288]"))
      assert_true(source.contains("vector < 256"))
      assert_true(source.contains("vector < 512"))
      assert_true(source.contains("operand_index(local_row, k)"))
      assert_true(source.contains("consumer_row = (warp / 4) * 1"))
      assert_true(source.contains("float c[1][8]"))
      assert_true(source.contains("if (warp < 8)"))
      assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
    } else {
      // Sparse demand shares the producer permutation with matrix loads;
      // wider transfers preserve the ordered microtile fold.
      assert_true(source.contains("matrix_storage[18432]"))
      assert_true(source.contains("operand_index(local_row, k)"))
      assert_true(source.contains("inner += 64"))
      assert_true(source.contains("consumer_row = (warp / 8) * 1"))
      assert_true(source.contains("float c[1][8]"))
      assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
    }
    assert_false(source.contains("temporary[result_index("))
    assert_true(source.contains("__float2bfloat16_rn(c[row][item])"))
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
