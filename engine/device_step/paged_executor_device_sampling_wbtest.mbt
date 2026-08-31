///|
test "embedded stochastic sampling v2 binds exact LM-head manifest module" {
  let metadata = blueprint_metadata()
  let model = blueprint_model_plan(metadata, final_language_model_head=true)
  let layout = blueprint_layout(model)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, model, layout, target)
  let memory = blueprint_memory_plan(metadata, model, layout, target)
  let contracts = full_contract_set(model, static_plan, memory, layout, target)
  let artifacts = blueprint_artifacts(contracts)
  let blueprint = admit_paged_graph_execution_blueprint(
    model,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    artifacts,
  )
  let policy = admit_paged_embedded_cuda_device_sampling_v2(
    model,
    blueprint,
    artifacts,
    maximum_top_k=4,
    maximum_workspace_bytes=1024L,
  )
  guard policy is PagedCudaRestrictiveTopKV2(admission) else {
    fail("embedded v2 reducer did not select device stochastic mode")
  }
  assert_eq(admission.plan.maximum_rows(), blueprint_limits().max_rows())
  assert_eq(admission.plan.vocabulary_size(), 8)
  assert_eq(admission.plan.maximum_top_k(), 4)
  assert_eq(
    admission.function_symbol,
    @luna_cuda_sampling_aot.device_sampling_cuda_function_symbol(),
  )
  validate_paged_sampling_execution_policy(model, blueprint, artifacts, policy)
}

///|
test "device sampling parameter staging is deterministic and fallback explicit" {
  let plan = @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=2,
    vocabulary_size=8,
    maximum_top_k=4,
    maximum_workspace_bytes=1024L,
  )
  let first : PagedDeviceSamplingScratch = {
    plan,
    parameters: FixedArray::make(80, b'\x7f'),
    results: FixedArray::make(16, b'\x00'),
  }
  let second : PagedDeviceSamplingScratch = {
    plan,
    parameters: FixedArray::make(80, b'\x35'),
    results: FixedArray::make(16, b'\x00'),
  }
  stage_paged_device_sampling_row(
    first, 0, true, 1, 0.75, true, 3, true, 0.9, 0x1234UL, 7,
  )
  stage_paged_device_sampling_row(
    second, 0, true, 1, 0.75, true, 3, true, 0.9, 0x1234UL, 7,
  )
  stage_paged_device_sampling_row(
    first, 1, true, 1, 1.0, false, 0, true, 0.95, 0x5678UL, 9,
  )
  stage_paged_device_sampling_row(
    second, 1, true, 1, 1.0, false, 0, true, 0.95, 0x5678UL, 9,
  )
  assert_eq(first.parameters, second.parameters)
  assert_eq(read_paged_i32(first.parameters, 0), 1)
  assert_eq(read_paged_i32(first.parameters, 1), 3)
  assert_eq(read_paged_i32(first.parameters, 8), 7)
  assert_eq(read_paged_i32(first.parameters, 9), 3)
  assert_eq(read_paged_i32(first.parameters, 10), -1)
  assert_eq(read_paged_i32(first.parameters, 11), 0)
  assert_eq(read_paged_i32(first.parameters, 18), 0)
  assert_eq(read_paged_i32(first.parameters, 19), 0)
}

///|
test "device sampling admission rejects a workspace below the fixed row plan" {
  let metadata = blueprint_metadata()
  let model = blueprint_model_plan(metadata, final_language_model_head=true)
  let layout = blueprint_layout(model)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, model, layout, target)
  let memory = blueprint_memory_plan(metadata, model, layout, target)
  let contracts = full_contract_set(model, static_plan, memory, layout, target)
  let artifacts = blueprint_artifacts(contracts)
  let blueprint = admit_paged_graph_execution_blueprint(
    model,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    artifacts,
  )
  try
    admit_paged_embedded_cuda_device_sampling_v2(
      model,
      blueprint,
      artifacts,
      maximum_top_k=4,
      maximum_workspace_bytes=8L,
    )
  catch {
    InvalidExecutor(SamplingArtifact) => ()
    _ => fail("unexpected undersized v2 workspace failure")
  } noraise {
    _ => fail("undersized device-sampling workspace was admitted")
  }
}
