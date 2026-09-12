///|
test "down fold lowers masked operands through shared transport layout" {
  let parameters : ProjectionParameters = {
    family: GatedMlp,
    input_width: 1024,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 3072,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let compiled = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch=runtime.dispatch_plan(1024),
      maximum_rows=32,
      maximum_tokens=1024,
      demand=AllTokenRows,
    ),
  )
  let pipeline = compiled.schedule().intermediate_pipeline().unwrap()
  let source = projection_fold_pipeline_source(
    pipeline, "input_base", "weight_base", 3072, 4, 1024,
  )
  assert_eq(
    source,
    projection_fold_pipeline_source(
      pipeline, "input_base", "weight_base", 3072, 4, 1024,
    ),
  )
  // Producer and consumer share the same compact K64 permutation.
  assert_eq(pipeline.row_width(), 32)
  assert_true(source.contains("pipeline_storage[12288]"))
  assert_true(source.contains("vector < 256"))
  assert_true(source.contains("destination + operand_index(row, k)"))
  assert_true(source.contains("destination + 2048 + operand_index(row, k)"))
  assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
  assert_false(source.contains("nvcuda::wmma"))
  assert_false(source.contains("is_a"))
  assert_true(source.contains("inner += 64"))
  assert_true(source.contains("k += 16"))
  assert_true(source.contains("row_start + row < token_count"))
  assert_true(source.contains("stage_operands(inner + 64, stage ^ 1)"))
  let complete = row_variant_primary_test_source(
    parameters,
    runtime.dispatch_plan(1024),
  )
  assert_true(
    complete.contains(
      "const int live_groups = ((token_count + 31) / 32) * group_count;",
    ),
  )
  assert_true(
    complete.contains("const int row_start = (work_group / group_count) * 32;"),
  )
  assert_false(complete.contains("wmma::"))
  if @env.get_env_var("LUNA_TEST_EXPORT_DOWN_OPERAND_LAYOUT") == Some("1") {
    println("LUNA_SOURCE_BEGIN mlp")
    println(
      row_variant_primary_test_source(parameters, runtime.dispatch_plan(1024)),
    )
    println("LUNA_SOURCE_END")
  }
}

///|
test "down producer maps cover logical operands exactly once without padding reads" {
  for width in [16, 32, 64, 128] {
    let layout : CudaProjectionTransportLayout = { columns: width }
    let rows = 64
    for threads in [32, 64, 128, 256, 512] {
      let logical = FixedArray::make(rows * width, 0)
      let physical = FixedArray::make(rows * width, 0)
      for thread in 0..<threads {
        for vector = thread
            vector < rows * width / 8
            vector = vector + threads {
          let row = vector / (width / 8)
          let column = vector % (width / 8) * 8
          let address = layout.offset(row, column)
          assert_eq(address % 8, 0)
          for element in 0..<8 {
            logical[row * width + column + element] += 1
            physical[address + element] += 1
          }
        }
      }
      for count in logical {
        assert_eq(count, 1)
      }
      for count in physical {
        assert_eq(count, 1)
      }
    }
  }
}
