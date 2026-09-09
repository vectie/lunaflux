///|
test "matrix fold covers short reduction extents without legacy generators" {
  for width in [16, 32, 48, 64, 128, 256, 384, 512, 1024] {
    for family in [QkvProjection, DenseProjection, LanguageModelHead] {
      let parameters : ProjectionParameters = {
        family,
        input_width: width,
        output_width: 80,
        query_width: if family == QkvProjection {
          48
        } else {
          0
        },
        intermediate_width: if family == QkvProjection {
          16
        } else {
          0
        },
      }
      let dispatch = projection_build_runtime_dispatch(
        parameters,
        1024,
        projection_fixture_target(),
        [],
      ).dispatch_plan(1024)
      assert_eq(dispatch.prefill().kernel_class(), MatrixTile)
      let source = row_variant_primary_test_source(parameters, dispatch)
      assert_true(source.contains("mma.sync.aligned.m16n8k16"))
      assert_true(source.contains("operand_index(local_row, k)"))
      assert_false(source.contains("wmma::"))
      // Both row and output-column tails are masked in the common transport.
      assert_true(source.contains("logical_row < pipeline_rows"))
      assert_true(source.contains("logical_column < 80"))
      if @env.get_env_var("LUNA_TEST_EXPORT_UNIFIED_FOLD") == Some("1") {
        let label = match family {
          QkvProjection => "qkv"
          DenseProjection => "dense"
          _ => "head"
        }
        println("LUNA_SOURCE_BEGIN \{label}-\{width}")
        println(source)
        println("LUNA_SOURCE_END")
      }
    }
  }
}
