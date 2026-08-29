///|
/// Independent lowercase SHA-256 identity of the exact descriptor bytes.
pub struct RuntimeDescriptorDigest(String) derive(Debug, Eq)

///|
pub fn RuntimeDescriptorDigest::from_sha256(
  value : StringView,
) -> RuntimeDescriptorDigest raise RuntimeDescriptorFileError {
  if !is_lower_sha256(value) {
    raise InvalidExpectedDigest
  }
  RuntimeDescriptorDigest(value.to_owned())
}

///|
pub fn RuntimeDescriptorDigest::as_hex(
  self : RuntimeDescriptorDigest,
) -> String {
  self.0
}

///|
/// Exact startup ceilings for the descriptor snapshot and structure scan.
pub struct RuntimeDescriptorFileLimits {
  priv max_descriptor_bytes : Int64
  priv max_json_depth : Int
}

///|
/// Opaque inert evidence reconstructed from one independently digest-pinned
/// descriptor and two independently bound approved roots.
///
/// It owns no root/file alias, native handle, device context, module, or
/// readiness authority.
pub struct RuntimeDescriptorAdmission {
  priv descriptor_digest : RuntimeDescriptorDigest
  priv target : @catalog.DeviceTarget
  priv device_ordinal : Int
  priv execution : @execution_manifest_file.PagedExecutionAdmission
  priv source : @worker_wire.EncodedBootstrapSource
  priv startup : @worker_wire.WorkerStartupContract
  priv _worker_plan : @device_worker.DeviceWorkerPlan
  priv graph_memory : @runtime_resolved.ResolvedGraphMemoryCapacity
}

///|
/// Opaque inert host admission for the strict production-I8 runtime recipe.
///
/// It retains only immutable inspected/planned evidence. It owns no root,
/// file, allocation, device context, executor, or readiness authority.
pub struct I8RuntimeDescriptorAdmission {
  priv descriptor_digest : RuntimeDescriptorDigest
  priv target : @catalog.DeviceTarget
  priv device_ordinal : Int
  priv weights : @numeric_weight_file.NumericWeightFileInspection
  priv execution : @execution_manifest_file.I8PagedExecutionAdmissionV4
  priv source : @worker_wire.EncodedBootstrapSource
  priv startup : @worker_wire.WorkerStartupContract
}

///|
/// Opaque, pre-hardware reusable FP8 runtime descriptor. Its execution value
/// retains the file-inspected externally approved v3 release authority.
pub struct Fp8RuntimeDescriptorAdmission {
  priv descriptor_digest : RuntimeDescriptorDigest
  priv target : @catalog.DeviceTarget
  priv device_ordinal : Int
  priv execution : @execution_manifest_file.Fp8PagedExecutionAdmissionV5
  priv source : @worker_wire.EncodedBootstrapSource
  priv startup : @worker_wire.WorkerStartupContract
}

///|
priv struct Fp8RuntimeDescriptorClaims {
  model : I8RuntimeDescriptorModelClaims
  kernels : Fp8RuntimeDescriptorKernelClaims
  execution : I8RuntimeDescriptorExecutionClaims
  worker : RuntimeDescriptorWorkerClaims
  inference : RuntimeDescriptorInferenceClaims
  ceilings : RuntimeDescriptorCeilingClaims
}

///|
priv struct Fp8RuntimeDescriptorKernelClaims {
  manifest_locator : String
  manifest_digest : String
  policy : String
}

///|
priv struct TensorParallelRankDescriptorClaims {
  rank : Int
  device_ordinal : Int
  memory_ceiling_bytes : Int64
  weight_reserved_bytes : Int64
  activation_workspace_reserved_bytes : Int64
  kv_reserved_bytes : Int64
  collective_reserved_bytes : Int64
}

///|
priv struct TensorParallelDescriptorClaims {
  ranks : ReadOnlyArray[TensorParallelRankDescriptorClaims]
  minimum_collective_runtime_version_code : Int
  maximum_collective_runtime_version_code : Int
  collective_exchange_timeout_millis : Int
}

///|
priv struct TensorParallelRuntimeDescriptorClaims {
  model : RuntimeDescriptorModelClaims
  kernels : TensorParallelRuntimeDescriptorKernelClaims
  execution : RuntimeDescriptorExecutionClaims
  worker : RuntimeDescriptorWorkerClaims
  inference : RuntimeDescriptorInferenceClaims
  ceilings : RuntimeDescriptorCeilingClaims
  tensor_parallel : TensorParallelDescriptorClaims
}

///|
/// Tensor-parallel rank bootstrap bytes bind the fresh rendezvous and group
/// generation at attempt construction time, so the static descriptor cannot
/// truthfully carry a final bootstrap digest.
priv struct TensorParallelRuntimeDescriptorKernelClaims {
  manifest_locator : String
  manifest_digest : String
  policy : String
}

///|
/// Device-free, owner-free foundation for one exact local tensor-parallel
/// startup. The topology is admitted from physical inventory, but no context,
/// collective, process, allocation, module, or readiness authority is owned.
pub struct TensorParallelRuntimeDescriptorAdmission {
  priv descriptor_digest : RuntimeDescriptorDigest
  priv model : @spec.LlamaModelMetadata
  priv plan : @plan.ModelPlan
  priv kv_layout : @device_layout.DeviceKvLayout
  priv source : @worker_wire.EncodedBootstrapSource
  priv model_generation : @worker_protocol.ModelPlanGeneration
  priv topology_selection : @device_topology.AdmittedRankOrderedLocalTopology
  priv inspections : ReadOnlyArray[
    @device_materialize.DeviceShardedWeightFileInspection,
  ]
  priv worker : @worker_protocol.WorkerProtocolLimits
  priv inference : @inference.InferenceLimits
  priv device_step_limits : @device_step.DeviceStepLimits
  priv manifest_limits : @execution_manifest_file.ExecutionManifestFileLimits
  priv launch_limits : @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits
  priv execution_limits : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits
  priv collective_exchange_timeout_millis : Int
}

///|
priv struct I8RuntimeDescriptorClaims {
  model : I8RuntimeDescriptorModelClaims
  kernels : I8RuntimeDescriptorKernelClaims
  execution : I8RuntimeDescriptorExecutionClaims
  worker : RuntimeDescriptorWorkerClaims
  inference : RuntimeDescriptorInferenceClaims
  ceilings : RuntimeDescriptorCeilingClaims
}

///|
priv struct I8RuntimeDescriptorModelClaims {
  config_locator : String
  config_digest : String
  content_digest : String
  numeric_weights_locator : String
  numeric_weight_artifact_digest : String
  max_batch_rows : Int
}

///|
priv struct I8RuntimeDescriptorKernelClaims {
  manifest_locator : String
  manifest_digest : String
  policy : String
  bootstrap_v3_digest : String
}

///|
priv struct I8RuntimeDescriptorExecutionClaims {
  device_ordinal : Int
  compute_major : Int
  compute_minor : Int
  total_memory_bytes : Int64
  supports_bf16 : Bool
  supports_cublas_lt : Bool
  tokens_per_page : Int
  total_page_count : Int
  model_generation : UInt64
}

///|
priv struct RuntimeDescriptorClaims {
  model : RuntimeDescriptorModelClaims
  kernels : RuntimeDescriptorKernelClaims
  execution : RuntimeDescriptorExecutionClaims
  worker : RuntimeDescriptorWorkerClaims
  inference : RuntimeDescriptorInferenceClaims
  ceilings : RuntimeDescriptorCeilingClaims
}

///|
priv struct MistralRuntimeDescriptorClaims {
  model : MistralRuntimeDescriptorModelClaims
  kernels : RuntimeDescriptorKernelClaims
  execution : RuntimeDescriptorExecutionClaims
  worker : RuntimeDescriptorWorkerClaims
  inference : RuntimeDescriptorInferenceClaims
  ceilings : RuntimeDescriptorCeilingClaims
}

///|
priv struct MistralRuntimeDescriptorModelClaims {
  family : String
  config_locator : String
  config_digest : String
  content_digest : String
  numeric_weights_locator : String
  numeric_weight_artifact_digest : String
  weight_manifest_digest : String
  sliding_window_tokens : Int
  max_batch_rows : Int
}

///|
priv struct RuntimeDescriptorModelClaims {
  config_locator : String
  config_digest : String
  weights_locator : String
  content_digest : String
  max_batch_rows : Int
}

///|
priv struct RuntimeDescriptorKernelClaims {
  manifest_locator : String
  manifest_digest : String
  policy : String
  bootstrap_digest : String
  fused_runtime : RuntimeDescriptorFusedRuntimeClaims?
  sampling_runtime : @worker_wire.WorkerBootstrapSamplingRuntime
}

///|
priv struct RuntimeDescriptorFusedRuntimeClaims {
  locator : String
  digest : String
}

///|
priv struct RuntimeDescriptorExecutionClaims {
  device_ordinal : Int
  compute_major : Int
  compute_minor : Int
  supports_bf16 : Bool
  supports_cublas_lt : Bool
  tokens_per_page : Int
  total_page_count : Int
  model_generation : UInt64
}

///|
priv struct RuntimeDescriptorWorkerClaims {
  max_prefill_rows : Int
  max_decode_rows : Int
  max_plan_rows : Int
  max_plan_tokens : Int
  max_plan_pages : Int
  max_capabilities : Int
  max_completion_slots : Int
  max_sequence_tokens : Int
  max_token_id : Int
}

///|
priv struct RuntimeDescriptorInferenceClaims {
  max_text_bytes : Int
  max_input_tokens : Int
  max_new_tokens : Int
  max_context_tokens : Int
  max_token_id : Int
  max_stop_token_ids : Int
  max_stop_strings : Int
  max_stop_string_bytes : Int
  max_trace_bytes : Int
  max_cache_scope_bytes : Int
  max_decoded_delta_bytes : Int
  max_deadline_millis : UInt64
  max_top_k : Int
  max_temperature : Double
}

///|
priv struct RuntimeDescriptorCeilingClaims {
  max_model_config_bytes : Int
  max_weight_file_bytes : Int64
  max_weight_arena_bytes : Int64
  max_activation_arena_bytes : Int64
  max_kv_arena_bytes : Int64
  max_execution_manifest_bytes : Int64
  max_module_bytes : Int64
  max_total_module_bytes : Int64
  max_graph_capture_bytes : Int64?
}

///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}
