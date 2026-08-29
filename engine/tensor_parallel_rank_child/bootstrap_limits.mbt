///|
const RANK_CHILD_MAX_CONFIG_BYTES : Int64 = 16777216L

///|
const RANK_CHILD_MAX_WEIGHT_HEADER_BYTES : Int = 16777216

///|
const RANK_CHILD_JSON_DEPTH : Int = 16

///|
const RANK_CHILD_MAX_OPERATIONS : Int = 4096

///|
const RANK_CHILD_MAX_OPERANDS : Int = 32

///|
const RANK_CHILD_MAX_DESCRIPTORS : Int = 32

///|
const RANK_CHILD_MAX_SYMBOL_BYTES : Int = 1024

///|
const RANK_CHILD_TRANSFER_CHUNK_BYTES : Int = 4194304

///|
const RANK_CHILD_ALIGNMENT_BYTES : Int64 = 256L

///|
fn rank_child_artifact_limits(
  source : @worker_wire.EncodedBootstrapSource,
  operation_count : Int,
) -> @artifact.KernelArtifactLimits raise TensorParallelRankChildError {
  let ceilings = source.ceilings()
  @artifact.KernelArtifactLimits::new(
    max_modules=operation_count,
    max_entry_points=operation_count,
    max_module_bytes=ceilings.max_module_bytes(),
    max_total_module_bytes=ceilings.max_total_module_bytes(),
    max_function_symbol_bytes=RANK_CHILD_MAX_SYMBOL_BYTES,
  ) catch {
    _ => raise Invalid(Limit)
  }
}

///|
fn rank_child_weight_limits(
  source : @worker_wire.EncodedBootstrapSource,
  operation_count : Int,
  rank_weight_ceiling : Int64,
) -> @device_materialize.DeviceWeightFileLimits raise TensorParallelRankChildError {
  let ceilings = source.ceilings()
  let materialized_limit = ceilings
    .max_weight_file_bytes()
    .min(ceilings.max_weight_arena_bytes())
    .min(rank_weight_ceiling)
  let header = @safetensors.HeaderLimits::new(
    max_header_bytes=RANK_CHILD_MAX_WEIGHT_HEADER_BYTES,
    max_tensor_count=operation_count,
    max_rank=2,
    max_name_length=256,
    max_payload_bytes=ceilings.max_weight_file_bytes(),
    max_materialized_tensor_bytes=materialized_limit,
  ) catch {
    _ => raise Invalid(Limit)
  }
  @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    transfer_chunk_bytes=RANK_CHILD_TRANSFER_CHUNK_BYTES,
    safetensors=header,
  ) catch {
    error => raise Weights(error)
  }
}

///|
fn rank_child_execution_limits(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  operation_count : Int,
  activation_ceiling : Int64,
) -> (
  @device_step.DeviceStepLimits,
  @execution_manifest_file.ExecutionManifestFileLimits,
  @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits,
  @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits,
) raise TensorParallelRankChildError {
  if operation_count <= 0 ||
    operation_count > RANK_CHILD_MAX_OPERATIONS ||
    activation_ceiling <= 0L {
    raise Invalid(Limit)
  }
  let artifact = rank_child_artifact_limits(source, operation_count)
  let worker = startup.worker_limits()
  let step = @device_step.DeviceStepLimits::new(
    max_rows=worker.max_plan_rows(),
    max_tokens=worker.max_plan_tokens(),
    max_pages=worker.max_plan_pages(),
    max_sequence_tokens=worker.max_sequence_tokens(),
    max_token_id=worker.max_token_id(),
  ) catch {
    _ => raise Invalid(Limit)
  }
  let memory = @device_memory.DeviceMemoryLimits::new(
    activation_alignment=RANK_CHILD_ALIGNMENT_BYTES,
    max_arena_bytes=activation_ceiling,
  ) catch {
    _ => raise Invalid(Limit)
  }
  let contract = @launch_contract.LaunchContractLimits::new(
    max_profiles=1,
    max_contracts=operation_count,
    max_operands_per_contract=RANK_CHILD_MAX_OPERANDS,
  ) catch {
    _ => raise Invalid(Limit)
  }
  let artifact_file = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=source.ceilings().max_execution_manifest_bytes(),
    max_json_depth=RANK_CHILD_JSON_DEPTH,
    max_path_bytes=4096,
    artifact~,
  ) catch {
    _ => raise Invalid(Limit)
  }
  let manifest = @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=source.ceilings().max_execution_manifest_bytes(),
    max_json_depth=RANK_CHILD_JSON_DEPTH,
    max_modules=operation_count,
    max_operations=operation_count,
    device_memory=memory,
    launch_contract=contract,
    artifact_file~,
  ) catch {
    error => raise Manifest(error)
  }
  let launch = @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits::new(
    max_profiles=1,
    max_contracts=operation_count,
    max_operands_per_contract=RANK_CHILD_MAX_OPERANDS,
  ) catch {
    _ => raise Invalid(Limit)
  }
  let execution = @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits::new(
    max_operations=operation_count,
    max_operands=operation_count * RANK_CHILD_MAX_OPERANDS,
    max_descriptors=RANK_CHILD_MAX_DESCRIPTORS,
    max_collectives=operation_count,
    max_descriptor_bytes=activation_ceiling,
    max_activation_workspace_bytes=activation_ceiling,
    artifact_limits=artifact,
  ) catch {
    _ => raise Invalid(Limit)
  }
  (step, manifest, launch, execution)
}
