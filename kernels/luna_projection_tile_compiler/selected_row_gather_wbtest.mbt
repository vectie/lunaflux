///|
fn gather_test_dispatch(
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
  if baseline.prefill().kernel_class() != MatrixTile {
    return baseline
  }
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
test "selected-row gather scope is pure canonical and keeps ordered inner fold" {
  let dispatch = gather_test_dispatch(1024)
  let request = ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=32,
    maximum_tokens=256,
    demand=QueryRowEnds,
  )
  let compiled = compile_projection_tile(request)
  assert_true(
    @utf8.decode(compiled.schedule.canonical).contains(
      "selected_row_operand_scope=register-fragment,16,16,16\n",
    ),
  )
  assert_true(
    @utf8.decode(compiled.schedule.canonical).contains(
      "selected_row_ordered_prefetch=8\n",
    ),
  )
  assert_eq(compiled, compile_projection_tile(request))
  assert_eq(compiled.schedule().single_row_reduction(), OrderedDot)
  assert_eq(
    compiled.schedule().loops()[2].components(),
    (ProjectionSequentialFold, ProjectionReductionAxis, 1024, 16),
  )
  for width in [15, 145, 241] {
    assert_true(
      projection_selected_row_gather(gather_test_dispatch(width), QueryRowEnds)
      is None,
    )
  }
  for width in [16, 32, 48, 64, 80, 128, 144, 240, 256, 896, 2048] {
    let gather = projection_selected_row_gather(
      gather_test_dispatch(width),
      QueryRowEnds,
    ).unwrap()
    assert_eq(gather.dimensions(), (16, 16, 16))
    assert_eq(
      gather.prefetch_fragments(),
      if width < 128 {
        width / 16
      } else {
        8
      },
    )
  }
  assert_true(projection_selected_row_gather(dispatch, AllTokenRows) is None)
  assert_true(
    projection_selected_row_gather(
      projection_compiler_dispatch(LanguageModelHead),
      QueryRowEnds,
    )
    is None,
  )
}
