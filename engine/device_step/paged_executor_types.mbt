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
  mut open : Bool
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
  mut persistent_kv : PagedOwnedAllocation?
  mut fused_dispatch_canary : PagedOwnedAllocation?
  mut greedy_sampling_results : PagedOwnedAllocation?
  modules : Array[PagedOwnedModule]
  functions : Array[PagedOwnedFunction]
  mut ordered_executor : PagedOwnedOrderedExecutor?
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
priv enum PagedSamplingCompletionMode {
  HostLogitsCompletion
  CudaGreedyCompletion(FixedArray[Byte])
}

///|
/// Startup-only sampling placement. `PagedCudaGreedySampling` admits only
/// greedy requests; stochastic modes are rejected while staging.
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
pub(all) enum PagedSamplingExecutionPolicy {
  PagedHostSampling
  PagedCudaGreedySampling(PagedCudaGreedySamplingAdmission)
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
  priv steps : ReadOnlyArray[PreparedPagedGraphStep]
  priv completion : PagedCompletionScratch
  priv fp8_frame_envelope : Fp8PagedFrameEnvelopeV3?
  priv fp8_frame_admission : Fp8FrameAdmissionStateV3?
  priv fp8_scale_evidence : Fp8ScaleEvidenceV3?
  priv graph_telemetry : PagedGraphTelemetryOwner
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
}

///|
/// Exact proof that every synchronous graph launch completed successfully.
#valtype
pub struct ExecutedPagedGraphExecution {
  priv executor : PagedGraphExecutor
  priv epoch : UInt64
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
