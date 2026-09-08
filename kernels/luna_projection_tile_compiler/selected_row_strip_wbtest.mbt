///|
fn strip_test_dispatch(
  input_width : Int,
) -> @luna_projection_strategy.ProjectionDispatchPlan raise {
  let shape = @luna_projection_strategy.ProjectionShape::new(
    LanguageModelHead,
    input_width~,
    output_width=151936,
  )
  let capabilities = @luna_projection_strategy.ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=Some(
      @luna_projection_strategy.ProjectionTile::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
  let baseline = @luna_projection_strategy.plan_projection_dispatch(
    shape, 256, capabilities,
  )
  let strategy = for
    candidate in baseline.prefill().output_distribution_candidates() {
    if candidate.output_tiles_per_workgroup() == 1 {
      break candidate
    }
  } nobreak {
    fail("single output tile is absent")
  }
  let backend = @luna_projection_strategy.ProjectionBackendId::new("test")
  let target = @luna_projection_strategy.ProjectionTargetId::new("test")
  let record = @luna_projection_strategy.ProjectionAutotuneRecord::new(
    key=@luna_projection_strategy.ProjectionAutotuneKey::new(
      backend~,
      target~,
      numeric_type=ProjectionBFloat16,
      shape~,
      strategy~,
    ),
    median_latency_ns=1L,
    sample_count=3,
  )
  @luna_projection_strategy.build_projection_runtime_dispatch(
    [record],
    backend~,
    target~,
    numeric_type=ProjectionBFloat16,
    shape~,
    maximum_rows=256,
    capabilities~,
  ).dispatch_plan(256)
}

///|
test "selected-row strip lifetime is pure canonical and keeps ordered inner fold" {
  let dispatch = strip_test_dispatch(1024)
  let request = ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=32,
    maximum_tokens=256,
    demand=QueryRowEnds,
  )
  let compiled = compile_projection_tile(request)
  assert_true(
    @utf8.decode(compiled.schedule.canonical).contains(
      "selected_row_reduction_strip=128\n",
    ),
  )
  assert_eq(compiled, compile_projection_tile(request))
  assert_eq(compiled.schedule().single_row_reduction(), OrderedDot)
  assert_eq(
    compiled.schedule().loops()[2].components(),
    (ProjectionSequentialFold, ProjectionReductionAxis, 1024, 16),
  )
  for width in [64, 144, 240] {
    assert_true(
      projection_selected_row_strip(strip_test_dispatch(width), QueryRowEnds)
      is None,
    )
  }
  for width in [128, 256, 896, 2048] {
    let strip = projection_selected_row_strip(
      strip_test_dispatch(width),
      QueryRowEnds,
    ).unwrap()
    assert_eq(strip.reduction_elements(), 128)
    assert_eq(strip.local_storage_bytes(), 9216)
  }
  assert_true(projection_selected_row_strip(dispatch, AllTokenRows) is None)
  assert_true(
    projection_selected_row_strip(
      projection_compiler_dispatch(LanguageModelHead),
      QueryRowEnds,
    )
    is None,
  )
}
