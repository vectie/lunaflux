///|
test "down small-row lowering keeps only demanded matrix row tiles" {
  let parameters : ProjectionParameters = {
    family: GatedMlp,
    input_width: 256,
    output_width: 256,
    query_width: 0,
    intermediate_width: 512,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  for row_case in [(8, 16), (16, 16), (32, 32), (1024, 32)] {
    let (maximum_tokens, rows) = row_case
    let transfer_width = 64
    let compilation = @luna_projection_tile_compiler.compile_projection_tile(
      @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch=runtime.dispatch_plan(1024),
        maximum_rows=Int::min(maximum_tokens, 8),
        maximum_tokens~,
        demand=AllTokenRows,
      ),
    )
    let pipeline = compilation.schedule().intermediate_pipeline().unwrap()
    let source = projection_fold_pipeline_source(
      pipeline, "input_base", "weight_base", 512, 2, 256,
    )
    assert_true(source.contains("float c[\{rows / 16}][8] = {};"))
    assert_eq(pipeline.transfer_width(), transfer_width)
    assert_true(
      source.contains("pipeline_storage[\{(rows + 32) * transfer_width * 2}]"),
    )
    assert_true(source.contains("vector < \{rows * transfer_width / 8}"))
    assert_true(
      source.contains(
        "destination + \{rows * transfer_width} + operand_index(row, k)",
      ),
    )
    assert_eq(
      source.split("if (row_start + row * 16 >= token_count) break;").count(),
      3,
    )
    assert_true(source.contains("inner < 512; inner += \{transfer_width}"))
    assert_true(source.contains("k < \{transfer_width}; k += 16"))
    assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
    assert_true(
      source.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"),
    )
    assert_true(source.contains("row_start + row < token_count"))
    assert_false(source.contains("__shfl_down_sync"))
    // This changes the row domain, not a CUDA-specific replacement for the
    // compiler's ordered matrix fold or its producer/consumer permutation.
    let layout : CudaProjectionTransportLayout = { columns: transfer_width }
    assert_true(source.contains(layout.address_function()))
    assert_true(source.contains(projection_result_address_source()))
  }
}

///|
test "bounded down operand bases preserve producer and matrix load bank maps" {
  for shape in [(16, 128), (32, 64), (48, 64), (64, 64)] {
    let (rows, width) = shape
    let layout : CudaProjectionTransportLayout = { columns: width }
    for groups in [1, 2, 4, 8] {
      let slot = (rows + groups * 16) * width
      for stage in 0..<2 {
        for base in [stage * slot, stage * slot + rows * width] {
          assert_eq(base % 64, 0)
          // An ldmatrix eight-lane row-address group loads eight contiguous
          // 16-byte vectors. The same map also covers producer vector writes.
          for row_start in [0, 8] {
            for k = 0; k < width; k = k + 8 {
              let banks = FixedArray::make(32, false)
              for lane in 0..<8 {
                let address = base + layout.offset(row_start + lane, k)
                for word in 0..<4 {
                  let bank = (address / 2 + word) % 32
                  assert_false(banks[bank])
                  banks[bank] = true
                }
              }
            }
          }
        }
      }
    }
  }
}
