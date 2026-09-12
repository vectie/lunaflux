///|
test "materialized consumer keeps asynchronous loads ahead of ordered matrix consumption" {
  let parameters = sibling_resource_test_parameters()
  let baseline = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  for groups in [1, 2, 4, 8, 16] {
    let strategy = sibling_resource_test_strategy(baseline, 8, groups)
    let runtime = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [row_variant_test_record(parameters, strategy)],
    )
    let compilation = @luna_projection_tile_compiler.compile_projection_tile(
      @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
        dispatch=runtime.dispatch_plan(8),
        maximum_rows=8,
        maximum_tokens=8,
        demand=AllTokenRows,
      ),
    )
    let pipeline = compilation.schedule().intermediate_pipeline().unwrap()
    let source = projection_fold_pipeline_source(
      pipeline,
      "input",
      "weight",
      3072,
      Int::min(4, groups),
      1024,
    )
    assert_true(source.contains("cp.async.cg.shared.global"))
    assert_false(source.contains("ld.global.v4.u32"))
    let prefix = source.split("const __nv_bfloat16 *current").to_array()[0]
    let width = if groups <= 2 { 128 } else { 64 }
    assert_eq(pipeline.transfer_width(), width)
    assert_true(prefix.contains("stage_operands(inner + \{width}, stage ^ 1)"))
    assert_true(source.contains("k < \{width}; k += 16"))
    assert_true(source.contains("cp.async.wait_group 0;"))
  }
}

///|
test "sibling pure operand product lowers to independent affine copy domains" {
  let source = sibling_bounded_source_for_test(8, 2)
  assert_false(source.contains("is_input"))
  assert_false(source.contains("is_gate"))
  assert_eq(source.split("ld.global.v4.u32").count(), 2)
  assert_true(source.contains("if (!overlap)"))
  assert_true(source.contains("stage_operands(0, 0, false)"))
  assert_true(source.contains("stage_operands(inner + 64, stage ^ 1, true)"))
  assert_true(source.contains("gate_weight + (unsigned long long)column_begin"))
  assert_true(source.contains("up_weight + (unsigned long long)column_begin"))
  assert_true(
    source.contains("const bool valid = row_start + row < token_count"),
  )
  assert_eq(source.split("ldmatrix.sync.aligned.m8n8.x4.shared.b16").count(), 4)
  assert_eq(
    source.split("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32").count(),
    3,
  )
  assert_true(
    source.contains("__float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up)"),
  )
}
