///|
test "output distribution enumerates immutable independent map partitions" {
  let shape = ProjectionShape::new(Dense, input_width=1024, output_width=2048)
  let capabilities = ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=Some(ProjectionTile::new(rows=16, columns=16, reduction=16)),
  )
  let dispatch = plan_projection_dispatch(shape, 8, capabilities)
  let baseline = dispatch.prefill()
  let candidates = baseline.output_distribution_candidates()
  assert_eq(candidates.length(), 5)
  assert_eq(candidates[0], baseline)
  for candidate in candidates {
    assert_eq(candidate.tile(), baseline.tile())
    assert_eq(candidate.row_bucket(), 8)
    assert_true(
      compatible_projection_candidate(shape, 8, candidate, capabilities),
    )
    assert_false(
      compatible_projection_candidate(
        shape,
        8,
        candidate,
        ProjectionCapabilities::new(subgroup_width=32),
      ),
    )
  }
  assert_eq(dispatch.decode().output_distribution_candidates().length(), 1)
  assert_eq(baseline.output_tiles_per_workgroup(), 8)
}

///|
test "offline measurements select distribution without changing the reduction" {
  let shape = ProjectionShape::new(Dense, input_width=1024, output_width=2048)
  let caps = ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=Some(ProjectionTile::new(rows=16, columns=16, reduction=16)),
  )
  let baseline = plan_projection_dispatch(shape, 8, caps).prefill()
  let narrow = baseline.output_distribution_candidates()[1]
  let backend = ProjectionBackendId::new("test-backend")
  let target = ProjectionTargetId::new("test-device")
  let records = [
    ProjectionAutotuneRecord::new(
      key=ProjectionAutotuneKey::new(
        backend~,
        target~,
        numeric_type=ProjectionBFloat16,
        shape~,
        strategy=narrow,
      ),
      median_latency_ns=1000L,
      sample_count=3,
    ),
  ]
  let selected = build_projection_runtime_dispatch(
    records,
    backend~,
    target~,
    numeric_type=ProjectionBFloat16,
    shape~,
    maximum_rows=8,
    capabilities=caps,
  )
  assert_eq(selected.dispatch_plan(8).prefill(), narrow)
  assert_eq(narrow.output_tiles_per_workgroup(), 1)
  assert_eq(narrow.tile(), baseline.tile())
}
