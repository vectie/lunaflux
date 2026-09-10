///|
test "selected-row prefetch lowering emits bounded constant slots and guards partial reduction windows" {
  for width in [16, 32, 48, 64, 80, 144, 1024] {
    let parameters : ProjectionParameters = {
      family: LanguageModelHead,
      input_width: width,
      output_width: 32,
      query_width: 0,
      intermediate_width: 0,
    }
    let target = projection_fixture_target()
    let baseline = projection_build_runtime_dispatch(parameters, 256, target, [])
    let strategy = for
      candidate in baseline
      .dispatch_plan(256)
      .prefill()
      .output_distribution_candidates() {
      if candidate.output_tiles_per_workgroup() == 1 {
        break candidate
      }
    } nobreak {
      fail("missing selected-row matrix strategy")
    }
    let runtime = projection_build_runtime_dispatch(parameters, 256, target, [
      row_variant_test_record(parameters, strategy),
    ])
    let gather = @luna_projection_tile_compiler.projection_selected_row_gather(
      runtime.dispatch_plan(256),
      QueryRowEnds,
    ).unwrap()
    let depth = gather.prefetch_fragments()
    let source = projection_selected_row_prefetch_fold_source(
      parameters, gather,
    )
    let advance = depth * 16
    assert_true(source.contains("inner < \{width}; inner += \{advance}"))
    assert_false(source.contains("operand\{depth}_a"))
    assert_false(source.contains("[slot]"))
    assert_false(source.contains("__shared__"))
    assert_false(source.contains("__syncthreads"))
    assert_false(source.contains("__shfl"))
    for slot in 0..<depth {
      let offset = slot * 16
      assert_true(
        source.contains("unsigned operand\{slot}_a[4], operand\{slot}_b[4]"),
      )
      assert_true(source.contains("const int k = \{offset} + (lane % 4) * 2"))
      assert_true(source.contains("if (inner + \{offset} < \{width})"))
      let future_guard = "if (inner + \{advance + offset} < \{width})"
      let fold = "\"r\"(operand\{slot}_a[0])"
      let before_fold = source.split(fold).to_array()[0]
      assert_true(before_fold.contains(future_guard))
      assert_true(
        before_fold.contains(
          "const int k = inner + \{advance + offset} + (lane % 4) * 2",
        ),
      )
      for part in 0..<4 {
        assert_true(
          source.contains("operand\{slot}_a[\{part}] = next_a[\{part}]"),
        )
        assert_true(
          source.contains("operand\{slot}_b[\{part}] = next_b[\{part}]"),
        )
      }
    }
    assert_eq(
      source
      .split("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
      .count(),
      depth + 1,
    )
  }
}
