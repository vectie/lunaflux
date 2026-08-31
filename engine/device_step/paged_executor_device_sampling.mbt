///|
const PAGED_DEVICE_SAMPLING_PARAMETER_ROW_BYTES : Int64 = 40L

///|
const PAGED_DEVICE_SAMPLING_RESULT_ROW_BYTES : Int64 = 8L

///|
const PAGED_DEVICE_SAMPLING_FALLBACK_MODE : Int = -1

///|
/// Binds the restrictive-top-k v2 ABI to the exact content-addressed module
/// that owns the terminal language-model head. The release producer must
/// co-compile the v2 symbol into that module. Missing symbols fail during
/// startup preparation; no runtime JIT or caller-selected module is allowed.
pub fn admit_paged_embedded_cuda_device_sampling_v2(
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
  maximum_top_k~ : Int,
  maximum_workspace_bytes~ : Int64,
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
  let plan = @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=blueprint.limits().max_rows(),
    vocabulary_size=boundary.vocabulary_size(),
    maximum_top_k~,
    maximum_workspace_bytes~,
  ) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  let abi = @luna_cuda_sampling_aot.device_sampling_cuda_abi_v2(plan) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  if !abi.manifest_bindable() ||
    abi.function_symbol() !=
    @luna_cuda_sampling_aot.device_sampling_cuda_function_symbol() ||
    abi.grid_x() != blueprint.limits().max_rows() ||
    abi.block_x() != 256 ||
    abi.parameter_bytes() !=
    plan.maximum_rows().to_int64() * plan.parameter_row_bytes() ||
    abi.workspace_bytes() != plan.workspace_bytes() ||
    abi.result_bytes() !=
    plan.maximum_rows().to_int64() * plan.result_row_bytes() {
    raise InvalidExecutor(SamplingArtifact)
  }
  let mut module_index = -1
  for step in blueprint.steps() {
    if step.operation_kind() == LanguageModelHead {
      if module_index >= 0 || step.operation_id() != boundary.producer() {
        raise InvalidExecutor(SamplingArtifact)
      }
      module_index = step.artifact_module_index()
    }
  }
  guard module_index >= 0 &&
    artifacts.modules().get(module_index) is Some(module_) else {
    raise InvalidExecutor(SamplingArtifact)
  }
  PagedCudaRestrictiveTopKV2({
    model_identity: model_plan.identity(),
    target: artifacts.device_target(),
    catalog_version: artifacts.catalog_version(),
    module_index,
    module_digest: module_.digest(),
    module_bytes: module_.module_bytes(),
    function_symbol: abi.function_symbol(),
    plan,
  })
}

///|
fn validate_paged_cuda_device_sampling_admission(
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
  admission : PagedCudaDeviceSamplingAdmission,
) -> Unit raise DeviceStepError {
  guard artifacts.modules().get(admission.module_index) is Some(module_) else {
    raise InvalidExecutor(SamplingArtifact)
  }
  let plan = admission.plan
  let abi = @luna_cuda_sampling_aot.device_sampling_cuda_abi_v2(plan) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  if admission.model_identity != model_plan.identity() ||
    admission.model_identity != blueprint.identity() ||
    admission.target != blueprint.target() ||
    admission.target != artifacts.device_target() ||
    admission.catalog_version != blueprint.catalog_version() ||
    admission.catalog_version != artifacts.catalog_version() ||
    module_.digest() != admission.module_digest ||
    module_.module_bytes() != admission.module_bytes ||
    admission.function_symbol != abi.function_symbol() ||
    plan.maximum_rows() != blueprint.limits().max_rows() ||
    plan.vocabulary_size() != model_plan.shape_constraints().vocabulary_size() ||
    plan.parameter_row_bytes() != PAGED_DEVICE_SAMPLING_PARAMETER_ROW_BYTES ||
    plan.result_row_bytes() != PAGED_DEVICE_SAMPLING_RESULT_ROW_BYTES {
    raise InvalidExecutor(SamplingArtifact)
  }
}

///|
fn allocate_paged_sampling_resource(
  context : @device.Context,
  bytes : Int64,
  stage : PagedGraphResource,
) -> PagedOwnedAllocation raise DeviceStepError {
  if bytes <= 0L {
    raise InvalidExecutor(SamplingArtifact)
  }
  let allocation = context.allocate(bytes) catch {
    error => raise ExecutorDevice(stage, error)
  }
  if allocation.byte_count() != bytes {
    try allocation.close() catch {
      _ => ()
    } noraise {
      _ => ()
    }
    raise InvalidExecutor(AllocationSize)
  }
  { resource: allocation, open: true }
}

///|
fn prepare_paged_cuda_device_sampling_step(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  admission : PagedCudaDeviceSamplingAdmission,
  artifacts : @artifact.KernelArtifactBundle,
  completion : PagedCompletionScratch,
  descriptor : DeviceStepResources,
  activation : @device.Allocation,
  resources : PagedGraphResources,
) -> PreparedPagedGraphStep raise DeviceStepError {
  validate_paged_cuda_device_sampling_admission(
    model_plan, blueprint, artifacts, admission,
  )
  guard resources.modules.get(admission.module_index) is Some(module_owner) &&
    module_owner.open else {
    raise InvalidExecutor(SamplingArtifact)
  }
  if !admission.target.matches_capability(context.capability()) ||
    admission.plan.vocabulary_size() != completion.vocabulary_size {
    raise InvalidExecutor(SamplingArtifact)
  }
  let abi = @luna_cuda_sampling_aot.device_sampling_cuda_abi_v2(admission.plan) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  let function_resource = module_owner.resource.load_function(
    @utf8.encode(admission.function_symbol),
  ) catch {
    error => raise ExecutorDevice(Function, error)
  }
  let function_index = resources.functions.length()
  resources.functions.push({
    resource: function_resource,
    module_index: admission.module_index,
    open: true,
  })
  let parameters = allocate_paged_sampling_resource(
    context,
    abi.parameter_bytes(),
    SamplingParameters,
  )
  resources.device_sampling_parameters = Some(parameters)
  let workspace = allocate_paged_sampling_resource(
    context,
    abi.workspace_bytes(),
    SamplingWorkspace,
  )
  resources.device_sampling_workspace = Some(workspace)
  let results = allocate_paged_sampling_resource(
    context,
    abi.result_bytes(),
    SamplingResult,
  )
  resources.device_sampling_results = Some(results)
  let terminal_bytes = completion.row_bytes *
    blueprint.limits().max_tokens().to_int64()
  if terminal_bytes <= 0L ||
    completion.final_output_offset > activation.byte_count() ||
    terminal_bytes > activation.byte_count() - completion.final_output_offset {
    raise InvalidExecutor(SamplingOutput)
  }
  let arguments = @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      descriptor.counts,
      offset=0L,
      byte_count=16L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      descriptor.query_row_offsets,
      offset=0L,
      byte_count=(blueprint.limits().max_rows() + 1).to_int64() * 4L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      activation,
      offset=completion.final_output_offset,
      byte_count=terminal_bytes,
      alignment=2L,
    ),
    @device.KernelArgument::device_region(
      parameters.resource,
      offset=0L,
      byte_count=abi.parameter_bytes(),
      alignment=8L,
    ),
    @device.KernelArgument::device_region(
      workspace.resource,
      offset=0L,
      byte_count=abi.workspace_bytes(),
      alignment=8L,
    ),
    @device.KernelArgument::device_region(
      results.resource,
      offset=0L,
      byte_count=abi.result_bytes(),
      alignment=4L,
    ),
  ]) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=abi.grid_x(),
    grid_y=1,
    grid_z=1,
    block_x=abi.block_x(),
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise InvalidExecutor(LaunchDimensions)
  }
  completion.mode = CudaRestrictiveTopKCompletion({
    plan: admission.plan,
    parameters: FixedArray::make(abi.parameter_bytes().to_int(), b'\x00'),
    results: FixedArray::make(abi.result_bytes().to_int(), b'\x00'),
  })
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(
    model_plan,
  ) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  { operation_id: boundary.producer(), function_index, dimensions, arguments }
}

///|
fn write_paged_device_sampling_row(
  host : FixedArray[Byte],
  row : Int,
  mode : Int,
  top_k : Int,
  temperature : Double,
  top_p : Double,
  seed : UInt64,
  sample_index : Int,
  flags : Int,
) -> Unit {
  let offset = row * PAGED_DEVICE_SAMPLING_PARAMETER_ROW_BYTES.to_int()
  host.unsafe_write_uint32_le(offset, mode.reinterpret_as_uint())
  host.unsafe_write_uint32_le(offset + 4, top_k.reinterpret_as_uint())
  host.unsafe_write_uint64_le(offset + 8, temperature.reinterpret_as_uint64())
  host.unsafe_write_uint64_le(offset + 16, top_p.reinterpret_as_uint64())
  host.unsafe_write_uint64_le(offset + 24, seed)
  host.unsafe_write_uint32_le(offset + 32, sample_index.reinterpret_as_uint())
  host.unsafe_write_uint32_le(offset + 36, flags.reinterpret_as_uint())
}

///|
fn write_paged_device_sampling_fallback(
  host : FixedArray[Byte],
  row : Int,
) -> Unit {
  write_paged_device_sampling_row(
    host,
    row,
    PAGED_DEVICE_SAMPLING_FALLBACK_MODE,
    0,
    0.0,
    0.0,
    0UL,
    0,
    0,
  )
}

///|
fn paged_device_sampling_route(
  plan : @luna_sampling_strategy.SamplingDevicePlan,
  mode : Int,
  temperature : Double,
  has_top_k : Bool,
  top_k : Int,
  has_top_p : Bool,
  top_p : Double,
) -> @luna_sampling_strategy.SamplingRoute raise DeviceStepError {
  @luna_sampling_strategy.select_sampling_route(
    plan,
    mode~,
    temperature~,
    has_top_k~,
    top_k~,
    has_top_p~,
    top_p~,
  ) catch {
    _ => raise InvalidExecutor(SamplingMode)
  }
}

///|
fn stage_paged_device_sampling_row(
  scratch : PagedDeviceSamplingScratch,
  row : Int,
  produces_token : Bool,
  mode : Int,
  temperature : Double,
  has_top_k : Bool,
  top_k : Int,
  has_top_p : Bool,
  top_p : Double,
  seed : UInt64,
  sample_index : Int,
) -> Unit raise DeviceStepError {
  if !produces_token ||
    paged_device_sampling_route(
      scratch.plan,
      mode,
      temperature,
      has_top_k,
      top_k,
      has_top_p,
      top_p,
    ) !=
    DeviceRestrictiveTopK {
    write_paged_device_sampling_fallback(scratch.parameters, row)
    return
  }
  write_paged_device_sampling_row(
    scratch.parameters,
    row,
    1,
    top_k,
    temperature,
    if has_top_p {
      top_p
    } else {
      0.0
    },
    seed,
    sample_index,
    1 | (if has_top_p { 2 } else { 0 }),
  )
}
