///|
priv struct PagedOwnedStream {
  resource : @device.Stream
  mut open : Bool
}

///|
priv struct PagedOwnedAllocation {
  resource : @device.Allocation
  mut open : Bool
}

///|
priv struct PagedOwnedLease {
  resource : @device.AllocationLease
  mut open : Bool
}

///|
priv struct PagedOwnedModule {
  resource : @device.Module
  mut open : Bool
}

///|
priv struct PagedOwnedFunction {
  resource : @device.Function
  module_index : Int
  mut open : Bool
}

///|
priv struct PagedOwnedOrderedExecutor {
  resource : @device.OrderedKernelExecutor
  operation_ids : ReadOnlyArray[@plan.OperationId]
  mut open : Bool
}

///|
priv enum PagedBucketStepLaunch {
  FixedBucketLaunch
  QueryTokenCappedGridX(@launch_contract.AotLaunchDimensions)
  QueryTileCappedGridX(@launch_contract.AotLaunchDimensions, Int)
  ProjectionTensorCoreGridX(@launch_contract.AotLaunchDimensions, Int)
  ProjectionScalarGridX(@launch_contract.AotLaunchDimensions, Int)
  ProjectionMlpUpGridX(@launch_contract.AotLaunchDimensions, Int)
  ProjectionMlpDownGridX(@launch_contract.AotLaunchDimensions, Int)
}

///|
/// Startup-only proof that selected Qwen and projection steps admit smaller
/// launch geometry. Unlisted steps retain their exact manifest geometry in
/// every bucket, including kernels that authenticate the full launch grid.
priv struct PagedGraphLaunchContract {
  step_launches : FixedArray[PagedBucketStepLaunch]
  decode_step_launches : FixedArray[PagedBucketStepLaunch]
  prefill_split_step_launches : FixedArray[PagedBucketStepLaunch]?
}

///|
/// Allocation-free slot dispatch for startup-materialized ordered executors.
/// A negative slot selects only the explicit fallback owner. Captured-required
/// mode may use its maximum-envelope capture as that fallback; the policy that
/// admits eager fallback owns a distinct eager executor instead.
priv struct PagedOrderedExecutorDispatch {
  slot_owner_indices : FixedArray[Int]
  prefill_split_slot_owner_indices : FixedArray[Int]
  fallback_owner_index : Int
  telemetry_owner_index : Int
  prefill_split_owner_index : Int
  prefill_split_minimum_context_tokens : Int
  prefill_split_query_tile : Int
  decode_owner_index : Int
  decode_split_owner_index : Int
  decode_split_minimum_context_tokens : Int
}

///|
priv enum PagedDescriptorOwner {
  Complete(DeviceStepOwner)
  Failed(FailedDeviceStepPreparation)
}

///|
priv enum PagedNumericWeightOwner {
  I8NumericWeights(I8LoadedExecutionBlueprint)
  Bf16NumericWeights(@numeric_weight_file.AuthenticatedNumericWeightAuthority)
  Fp8NumericWeights(@numeric_weight_file.AuthenticatedNumericWeightAuthority)
}

///|
priv struct PagedGraphResources {
  mut descriptor : PagedDescriptorOwner?
  mut stream : PagedOwnedStream?
  mut weight_lease : PagedOwnedLease?
  mut activation_workspace : PagedOwnedAllocation?
  mut qwen_fused_attention_sidecar : PagedOwnedAllocation?
  mut prefill_partition_workspace : PagedOwnedAllocation?
  mut decode_split_workspace : PagedOwnedAllocation?
  mut persistent_kv : PagedOwnedAllocation?
  mut fused_dispatch_canary : PagedOwnedAllocation?
  mut greedy_sampling_results : PagedOwnedAllocation?
  mut device_sampling_parameters : PagedOwnedAllocation?
  mut device_sampling_workspace : PagedOwnedAllocation?
  mut device_sampling_results : PagedOwnedAllocation?
  modules : Array[PagedOwnedModule]
  functions : Array[PagedOwnedFunction]
  ordered_executors : Array[PagedOwnedOrderedExecutor]
  mut graph_launch_contract : PagedGraphLaunchContract?
  mut numeric_weights : PagedNumericWeightOwner?
}

///|
priv struct PagedFunctionBlueprint {
  identity : @catalog.AotKernelEntryPoint
  module_index : Int
  symbol : String
}

///|
priv struct PreparedPagedGraphStep {
  operation_id : @plan.OperationId
  function_index : Int
  dimensions : @device.KernelLaunchDimensions
  arguments : @device.KernelArguments
}

///|
priv struct PagedCompletionScratch {
  final_output_offset : Int64
  row_bytes : Int64
  vocabulary_size : Int
  limits : @sampling.SamplingLimits
  sampler : @sampling.SamplerScratch
  encoded_logits : FixedArray[Byte]
  logits : FixedArray[Double]
  selected_tokens : FixedArray[Int]
  mut mode : PagedSamplingCompletionMode
}

///|
priv struct PagedDeviceSamplingScratch {
  plan : @luna_sampling_strategy.SamplingDevicePlan
  parameters : FixedArray[Byte]
  results : FixedArray[Byte]
}

///|
priv enum PagedSamplingCompletionMode {
  HostLogitsCompletion
  CudaGreedyCompletion(FixedArray[Byte])
  CudaRestrictiveTopKCompletion(PagedDeviceSamplingScratch)
}

///|
/// Startup-only greedy sampling placement. Stochastic modes are rejected while
/// this policy is active.
pub struct PagedCudaGreedySamplingAdmission {
  priv model_identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv module_index : Int
  priv module_digest : @catalog.AotArtifactDigest
  priv module_bytes : Bytes
  priv function_symbol : String
  priv vocabulary_size : Int
  priv maximum_rows : Int
  priv result_row_bytes : Int64
} derive(Debug, Eq)

///|
/// Startup-only admission for the restrictive top-k v2 reducer co-compiled
/// into the exact manifest-owned language-model-head module.
pub struct PagedCudaDeviceSamplingAdmission {
  priv model_identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv module_index : Int
  priv module_digest : @catalog.AotArtifactDigest
  priv module_bytes : Bytes
  priv function_symbol : String
  priv plan : @luna_sampling_strategy.SamplingDevicePlan
} derive(Debug, Eq)

///|
pub(all) enum PagedSamplingExecutionPolicy {
  PagedHostSampling
  PagedCudaGreedySampling(PagedCudaGreedySamplingAdmission)
  PagedCudaRestrictiveTopKV2(PagedCudaDeviceSamplingAdmission)
} derive(Debug, Eq)

///|
priv struct Fp8PagedFrameEnvelopeV3 {
  profile : @launch_contract.PagedKernelProfile
  model_generation_value : UInt64
  kv_layout_version : Int
  tokens_per_page : Int
  total_page_count : Int
  release_digest : String
}

///|
priv struct Fp8FrameAdmissionStateV3 {
  mut armed : Bool
  mut descriptor_epoch : UInt64
  mut wire_owner : @worker_wire.PlanFrameOwnerIdentity?
  mut wire_epoch : UInt64
  mut sequence_value : UInt64
  mut row_count : Int
  mut token_count : Int
  mut page_count : Int
}

///|
priv struct Fp8ScaleEvidenceV3 {
  workspace_base : Int64
  workspace_bytes : Int64
  host : FixedArray[Byte]
}

///|
priv struct FusedDispatchCanaryState {
  per_token : UInt
  host : FixedArray[Byte]
}

///|
/// Single-owner synchronous executor for one admitted full paged graph.
///
/// The owner never exposes allocations, modules, functions, streams, kernel
/// arguments, or close authority for its private native resources.
pub struct PagedGraphExecutor {
  priv resources : PagedGraphResources
  priv completion : PagedCompletionScratch
  priv fp8_frame_envelope : Fp8PagedFrameEnvelopeV3?
  priv fp8_frame_admission : Fp8FrameAdmissionStateV3?
  priv fp8_scale_evidence : Fp8ScaleEvidenceV3?
  priv graph_telemetry : PagedGraphTelemetryOwner
  priv graph_bucket_table : @luna_execution_graph_strategy.ExecutionGraphBucketTable
  priv ordered_dispatch : PagedOrderedExecutorDispatch
  priv fused_residual_spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan]
  priv fused_dispatch_canary : FusedDispatchCanaryState?
  priv mut lifecycle : PagedGraphLifecycle
}

///|
/// Exact staged-step capability. Its constructor and fields remain opaque.
#valtype
pub struct StagedPagedGraphExecution {
  priv executor : PagedGraphExecutor
  priv epoch : UInt64
  priv graph_bucket : @luna_execution_graph_strategy.ExecutionGraphBucket
  priv ordered_executor_index : Int
}

///|
pub fn StagedPagedGraphExecution::graph_bucket(
  self : StagedPagedGraphExecution,
) -> @luna_execution_graph_strategy.ExecutionGraphBucket {
  self.graph_bucket
}

///|
/// Exact proof that every synchronous graph launch completed successfully.
#valtype
pub struct ExecutedPagedGraphExecution {
  priv executor : PagedGraphExecutor
  priv epoch : UInt64
  priv ordered_executor_index : Int
}

///|
pub(all) enum PagedGraphPreparation {
  PagedReady(PagedGraphExecutor)
  PagedCleanupRequired(FailedPagedGraphPreparation)
}

///|
/// Cleanup authority retained when preparation and its first cleanup both
/// fail. No executable owner is returned in this state.
pub struct FailedPagedGraphPreparation {
  priv resources : PagedGraphResources
  priv cause : DeviceStepError
  priv mut cleanup_failure : DeviceStepError
  priv mut lifecycle : PagedGraphLifecycle
}

///|
pub fn PagedGraphExecutor::lifecycle(
  self : PagedGraphExecutor,
) -> PagedGraphLifecycle {
  self.lifecycle
}

///|
pub fn PagedGraphExecutor::uses_fused_residual_rmsnorm(
  self : PagedGraphExecutor,
) -> Bool {
  !self.fused_residual_spans.is_empty()
}

///|
pub fn FailedPagedGraphPreparation::cause(
  self : FailedPagedGraphPreparation,
) -> DeviceStepError {
  self.cause
}

///|
pub fn FailedPagedGraphPreparation::cleanup_failure(
  self : FailedPagedGraphPreparation,
) -> DeviceStepError {
  self.cleanup_failure
}

///|
pub fn FailedPagedGraphPreparation::is_closed(
  self : FailedPagedGraphPreparation,
) -> Bool {
  self.lifecycle == GraphClosed
}
