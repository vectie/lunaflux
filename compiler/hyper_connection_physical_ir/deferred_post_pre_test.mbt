///|
test "deferred post pre partitions controls without changing numerical ownership" {
  let plan = @precision.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=64,
    sinkhorn_iterations=20,
    control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5,
    mix_law=RoundedBf16ControlsAndProducts,
    control_square_law=VectorizedTreeFloat32(partitions=256, vector_width=1),
    control_dot_law=StripedTreeFloat32(partitions=256),
  )
  for group in [1, 2, 4] {
    let fused = @hyper_connection_physical_ir.DeferredPostPrePlan::new(
      plan,
      plan,
      controls_per_group=group,
    )
    assert_eq(fused.geometry(), (4, 4096, 64, 256, group))
    assert_eq(fused.workgroups(), 64 * 24 / group)
    assert_eq(fused.retained_values_per_worker(), 64)
  }
  for group in [0, 5] {
    assert_true(
      (try? @hyper_connection_physical_ir.DeferredPostPrePlan::new(
        plan,
        plan,
        controls_per_group=group,
      ))
      is Err(_),
    )
  }
  let dispatch = @hyper_connection_physical_ir.DeferredPostPreDispatch::new(
    plan,
    plan,
    [(1, 1), (32, 2), (64, 4)],
    workgroup_budget=128,
  )
  assert_eq(dispatch.workgroups(), 128)
  assert_eq(dispatch.buckets().length(), 3)
  assert_true(
    (try? @hyper_connection_physical_ir.DeferredPostPreDispatch::new(
      plan,
      plan,
      [(1, 1), (32, 2)],
      workgroup_budget=128,
    ))
    is Err(_),
  )
  assert_true(
    (try? @hyper_connection_physical_ir.DeferredPostPreDispatch::new(
      plan,
      plan,
      [(32, 2), (1, 1), (64, 4)],
      workgroup_budget=128,
    ))
    is Err(_),
  )
  let ordered = @precision.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=64,
    sinkhorn_iterations=20,
    control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5,
    mix_law=RoundedBf16ControlsAndProducts,
  )
  assert_true(
    (try? @hyper_connection_physical_ir.DeferredPostPrePlan::new(
      plan,
      ordered,
      controls_per_group=1,
    ))
    is Err(_),
  )
}
