///|
test "segmented greedy version is selected by the admitted head only" {
  let metadata = blueprint_metadata()
  let model = blueprint_model_plan(metadata, final_language_model_head=true)
  let layout = blueprint_layout(model)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, model, layout, target)
  let memory = blueprint_memory_plan(metadata, model, layout, target)
  let contracts = full_contract_set(model, static_plan, memory, layout, target)
  let artifacts = blueprint_artifacts(contracts)
  let original = admit_paged_graph_execution_blueprint(
    model,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    artifacts,
  )
  let steps = original
    .steps()
    .map(fn(step) {
      if step.operation_kind() == LanguageModelHead {
        {
          ..step,
          function_symbol: step.function_symbol() +
          @luna_cuda_sampling_aot.segmented_greedy_head_suffix(),
        }
      } else {
        step
      }
    })
  let blueprint = { ..original, steps, }
  let policy = admit_paged_embedded_cuda_greedy_sampling(
    model, blueprint, artifacts,
  )
  guard policy is PagedCudaGreedySampling(admission) else {
    fail("missing segmented admission")
  }
  assert_eq(
    admission.function_symbol,
    @luna_cuda_sampling_aot.segmented_greedy_partial_symbol(),
  )
  validate_paged_sampling_execution_policy(model, blueprint, artifacts, policy)
  try
    validate_paged_sampling_execution_policy(model, original, artifacts, policy)
  catch {
    InvalidExecutor(SamplingArtifact) => ()
    _ => fail("unexpected compatibility failure")
  } noraise {
    _ => fail("v2 admitted against a v1 head")
  }
}
