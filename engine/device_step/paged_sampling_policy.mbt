///|
const PAGED_GREEDY_RESULT_ROW_BYTES : Int64 = 8L

///|
/// Joins an inert offline sampler to one exact authenticated graph bundle.
/// The executor reauthenticates this opaque join before loading the symbol.
pub fn admit_paged_cuda_greedy_sampling(
  model_plan : @plan.ModelPlan,
  artifacts : @artifact.KernelArtifactBundle,
  artifact : @luna_cuda_sampling_aot.GreedySamplingCudaArtifact,
) -> PagedSamplingExecutionPolicy raise DeviceStepError {
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(
    model_plan,
  ) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  if artifacts.model_identity() != model_plan.identity() ||
    artifacts.device_target() != artifact.target() ||
    boundary.vocabulary_size() != artifact.vocabulary_size() {
    raise InvalidExecutor(SamplingArtifact)
  }
  let mut module_index = -1
  for index, module_ in artifacts.modules() {
    if module_.digest() == artifact.module_digest() &&
      module_.module_bytes() == artifact.module_bytes() {
      if module_index >= 0 {
        raise InvalidExecutor(SamplingArtifact)
      }
      module_index = index
    }
  }
  if module_index < 0 {
    raise InvalidExecutor(SamplingArtifact)
  }
  PagedCudaGreedySampling({
    model_identity: artifacts.model_identity(),
    target: artifacts.device_target(),
    catalog_version: artifacts.catalog_version(),
    module_index,
    module_digest: artifact.module_digest(),
    module_bytes: artifact.module_bytes(),
    function_symbol: artifact.function_symbol(),
    vocabulary_size: artifact.vocabulary_size(),
    maximum_rows: artifact.maximum_rows(),
    result_row_bytes: artifact.result_row_bytes(),
  })
}

///|
/// Selects the product-owned greedy reducer co-compiled into the exact
/// language-model-head module by the BF16 release producer. The module is
/// already content-authenticated by the full-graph artifact bundle; this join
/// adds no caller-selected bytes or symbol and opens no runtime resource.
pub fn admit_paged_embedded_cuda_greedy_sampling(
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
) -> PagedSamplingExecutionPolicy raise DeviceStepError {
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(
    model_plan,
  ) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  if blueprint.identity() != model_plan.identity() ||
    artifacts.model_identity() != model_plan.identity() ||
    blueprint.target() != artifacts.device_target() ||
    blueprint.catalog_version() != artifacts.catalog_version() ||
    blueprint.limits().max_rows() <= 0 {
    raise InvalidExecutor(SamplingArtifact)
  }
  let mut module_index = -1
  let mut segmented = false
  for step in blueprint.steps() {
    if step.operation_kind() == LanguageModelHead {
      if module_index >= 0 || step.operation_id() != boundary.producer() {
        raise InvalidExecutor(SamplingArtifact)
      }
      module_index = step.artifact_module_index()
      segmented = step
        .function_symbol()
        .has_suffix(@luna_cuda_sampling_aot.segmented_greedy_head_suffix())
    }
  }
  guard module_index >= 0 &&
    artifacts.modules().get(module_index) is Some(module_) else {
    raise InvalidExecutor(SamplingArtifact)
  }
  PagedCudaGreedySampling({
    model_identity: model_plan.identity(),
    target: artifacts.device_target(),
    catalog_version: artifacts.catalog_version(),
    module_index,
    module_digest: module_.digest(),
    module_bytes: module_.module_bytes(),
    function_symbol: if segmented {
      @luna_cuda_sampling_aot.segmented_greedy_partial_symbol()
    } else {
      @luna_cuda_sampling_aot.greedy_sampling_cuda_function_symbol()
    },
    vocabulary_size: boundary.vocabulary_size(),
    maximum_rows: blueprint.limits().max_rows(),
    result_row_bytes: PAGED_GREEDY_RESULT_ROW_BYTES,
  })
}

///|
/// Replays the startup-only sampling join against the exact worker graph.
/// No hash, filesystem operation, allocation, module load, or device call is
/// performed here.
pub fn validate_paged_sampling_execution_policy(
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
  policy : PagedSamplingExecutionPolicy,
) -> Unit raise DeviceStepError {
  match policy {
    PagedHostSampling => ()
    PagedCudaGreedySampling(admission) => {
      let expected = admit_paged_embedded_cuda_greedy_sampling(
        model_plan, blueprint, artifacts,
      )
      if expected != policy ||
        (
          admission.function_symbol !=
          @luna_cuda_sampling_aot.greedy_sampling_cuda_function_symbol() &&
          admission.function_symbol !=
          @luna_cuda_sampling_aot.segmented_greedy_partial_symbol()
        ) {
        raise InvalidExecutor(SamplingArtifact)
      }
    }
    PagedCudaRestrictiveTopKV2(admission) =>
      validate_paged_cuda_device_sampling_admission(
        model_plan, blueprint, artifacts, admission,
      )
  }
}
