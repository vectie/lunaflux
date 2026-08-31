///|
test "device sampling workspace is fixed and row-major" {
  let plan = @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=8,
    vocabulary_size=151936,
    maximum_top_k=64,
    maximum_workspace_bytes=6144L,
  )
  assert_eq(plan.parameter_row_bytes(), 40L)
  assert_eq(plan.result_row_bytes(), 8L)
  assert_eq(plan.candidate_token_offset_bytes(), 0L)
  assert_eq(plan.candidate_weight_offset_bytes(), 2048L)
  assert_eq(plan.workspace_bytes(), 6144L)
}

///|
test "route keeps greedy and bounded top-k on device" {
  let plan = @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=4,
    vocabulary_size=32000,
    maximum_top_k=128,
    maximum_workspace_bytes=6144L,
  )
  assert_eq(
    @luna_sampling_strategy.select_sampling_route(
      plan,
      mode=0,
      temperature=1.0,
      has_top_k=false,
      top_k=0,
      has_top_p=false,
      top_p=0.0,
    ),
    DeviceGreedy,
  )
  assert_eq(
    @luna_sampling_strategy.select_sampling_route(
      plan,
      mode=1,
      temperature=0.7,
      has_top_k=true,
      top_k=40,
      has_top_p=true,
      top_p=0.9,
    ),
    DeviceRestrictiveTopK,
  )
}

///|
test "unbounded or oversized stochastic rows retain canonical host fallback" {
  let plan = @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=2,
    vocabulary_size=32000,
    maximum_top_k=64,
    maximum_workspace_bytes=1536L,
  )
  assert_eq(
    @luna_sampling_strategy.select_sampling_route(
      plan,
      mode=1,
      temperature=1.0,
      has_top_k=false,
      top_k=0,
      has_top_p=true,
      top_p=0.95,
    ),
    HostCanonical,
  )
  assert_eq(
    @luna_sampling_strategy.select_sampling_route(
      plan,
      mode=1,
      temperature=1.0,
      has_top_k=true,
      top_k=128,
      has_top_p=false,
      top_p=0.0,
    ),
    HostCanonical,
  )
}

///|
test "workspace and parameter failures are fail closed" {
  try
    @luna_sampling_strategy.plan_device_sampling(
      maximum_rows=8,
      vocabulary_size=32000,
      maximum_top_k=64,
      maximum_workspace_bytes=6143L,
    )
  catch {
    InvalidSamplingStrategy(Workspace) => ()
    _ => fail("unexpected device sampling workspace error")
  } noraise {
    _ => fail("undersized device sampling workspace was admitted")
  }
}

///|
test "F64 candidate weights remain aligned for odd candidate counts" {
  let plan = @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=1,
    vocabulary_size=7,
    maximum_top_k=3,
    maximum_workspace_bytes=40L,
  )
  assert_eq(plan.candidate_weight_offset_bytes(), 16L)
  assert_eq(plan.workspace_bytes(), 40L)
}
