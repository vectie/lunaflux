///|
/// Opaque lowering candidates emitted only by the three owning BF16 family
/// packages. No source text, symbol, flags, or module identity is accepted
/// independently at this boundary.
pub(all) enum LunaBf16KernelCandidate {
  Pointwise(@luna_cuda_pointwise_aot.PointwiseCudaLowering)
  Projection(@luna_cuda_projection_aot.ProjectionCudaLowering)
  PagedAttention(@luna_cuda_paged_attention_aot.PagedAttentionCudaLowering)
} derive(Debug, Eq)

///|
/// One candidate paired with the two immutable outputs and receipt from an
/// approved offline compiler driver. Construction is inert; the aggregate
/// producer authenticates every identity before publication.
pub struct LunaBf16OfflineCompilation {
  priv candidate : LunaBf16KernelCandidate
  priv first_module : Bytes
  priv second_module : Bytes
  priv receipt : @luna_kernel_bundle.LunaDeterministicCompileReceipt
}

///|
pub fn LunaBf16OfflineCompilation::new(
  candidate~ : LunaBf16KernelCandidate,
  first_module~ : Bytes,
  second_module~ : Bytes,
  receipt~ : @luna_kernel_bundle.LunaDeterministicCompileReceipt,
) -> LunaBf16OfflineCompilation {
  { candidate, first_module, second_module, receipt }
}

///|
pub struct LunaBf16KernelProducerLimits {
  priv max_operations : Int
  priv launch_contracts : @launch_contract.LaunchContractLimits
  priv bundle : @luna_kernel_bundle.LunaKernelBundleLimits
}

///|
pub fn LunaBf16KernelProducerLimits::new(
  max_operations~ : Int,
  launch_contracts~ : @launch_contract.LaunchContractLimits,
  bundle~ : @luna_kernel_bundle.LunaKernelBundleLimits,
) -> LunaBf16KernelProducerLimits raise LunaBf16KernelProducerError {
  if max_operations < 9 {
    raise Invalid(Limits)
  }
  { max_operations, launch_contracts, bundle }
}

///|
/// Final inert result of actual-digest catalog construction, full-graph launch
/// admission, strict family binding, and exact bundle production.
pub struct LunaBf16KernelReleaseInput {
  priv catalog : @catalog.ResolvedKernelCatalog
  priv contracts : @launch_contract.PagedKvLaunchContractSet
  priv bundle : @luna_kernel_bundle.LunaKernelBundleInput
  priv embedded_greedy_sampling : Bool
}

///|
/// True only after the final language-model-head candidate, deterministic
/// compile receipt, actual module digest, and launch binding have all joined.
pub fn LunaBf16KernelReleaseInput::has_authenticated_embedded_greedy_sampling(
  self : LunaBf16KernelReleaseInput,
) -> Bool {
  self.embedded_greedy_sampling
}

///|
pub fn LunaBf16KernelReleaseInput::catalog(
  self : LunaBf16KernelReleaseInput,
) -> @catalog.ResolvedKernelCatalog {
  self.catalog
}

///|
pub fn LunaBf16KernelReleaseInput::contracts(
  self : LunaBf16KernelReleaseInput,
) -> @launch_contract.PagedKvLaunchContractSet {
  self.contracts
}

///|
pub fn LunaBf16KernelReleaseInput::bundle(
  self : LunaBf16KernelReleaseInput,
) -> @luna_kernel_bundle.LunaKernelBundleInput {
  self.bundle
}

///|
priv struct ReusableFamilyBinding {
  operation_kind : @plan.OperationKind
  operation_shape : @plan.OperationShape
  capability : @plan.KernelCapabilityId
  semantics : @catalog.KernelExecutionSemantics
  workspace_bytes : Int64
  workspace_alignment : Int64
  family : @catalog.AotKernelFamily
  entry_point_id : @catalog.AotKernelEntryPointId
  function_symbol : String
  source_digest_sha256 : String
}

///|
priv struct PreparedCompilation {
  compilation : LunaBf16OfflineCompilation
  operation : @plan.PlanOperation
}
