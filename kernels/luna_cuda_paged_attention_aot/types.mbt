///|
pub struct PagedAttentionAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
} derive(Debug, Eq)

///|
pub fn PagedAttentionAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
) -> PagedAttentionAotLimits raise PagedAttentionAotError {
  if max_source_bytes <= 0 || max_recipe_bytes <= 0 {
    raise Invalid(Limits)
  }
  { max_source_bytes, max_recipe_bytes }
}

///|
pub struct PagedAttentionCudaLowering {
  priv target : @catalog.DeviceTarget
  priv operation_id : @plan.OperationId
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv profile : @launch_contract.PagedKernelProfile
  priv query_heads : Int
  priv kv_heads : Int
  priv head_dimension : Int
  priv tokens_per_page : Int
  priv page_stride_bytes : Int64
  priv physical_page_capacity : Int64
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : PagedAttentionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : PagedAttentionCudaRecipeDigest
} derive(Debug, Eq)

///|
/// Exact post-compile join to the final content-addressed full-graph contract.
pub struct PagedAttentionManifestCudaLowering {
  priv candidate : PagedAttentionCudaLowering
  priv entry_point : @catalog.AotKernelEntryPoint
  priv binding_record : Bytes
  priv binding_digest : PagedAttentionManifestBindingDigest
} derive(Debug, Eq)

///|
pub fn PagedAttentionCudaLowering::target(
  self : PagedAttentionCudaLowering,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn PagedAttentionCudaLowering::operation_id(
  self : PagedAttentionCudaLowering,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn PagedAttentionCudaLowering::entry_point_id(
  self : PagedAttentionCudaLowering,
) -> @catalog.AotKernelEntryPointId {
  self.entry_point_id
}

///|
pub fn PagedAttentionCudaLowering::profile(
  self : PagedAttentionCudaLowering,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn PagedAttentionCudaLowering::query_heads(
  self : PagedAttentionCudaLowering,
) -> Int {
  self.query_heads
}

///|
pub fn PagedAttentionCudaLowering::kv_heads(
  self : PagedAttentionCudaLowering,
) -> Int {
  self.kv_heads
}

///|
pub fn PagedAttentionCudaLowering::head_dimension(
  self : PagedAttentionCudaLowering,
) -> Int {
  self.head_dimension
}

///|
pub fn PagedAttentionCudaLowering::tokens_per_page(
  self : PagedAttentionCudaLowering,
) -> Int {
  self.tokens_per_page
}

///|
pub fn PagedAttentionCudaLowering::page_stride_bytes(
  self : PagedAttentionCudaLowering,
) -> Int64 {
  self.page_stride_bytes
}

///|
pub fn PagedAttentionCudaLowering::physical_page_capacity(
  self : PagedAttentionCudaLowering,
) -> Int64 {
  self.physical_page_capacity
}

///|
pub fn PagedAttentionCudaLowering::operands(
  self : PagedAttentionCudaLowering,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn PagedAttentionCudaLowering::launch_dimensions(
  self : PagedAttentionCudaLowering,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn PagedAttentionCudaLowering::compiler_policy(
  self : PagedAttentionCudaLowering,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn PagedAttentionCudaLowering::function_symbol(
  self : PagedAttentionCudaLowering,
) -> String {
  self.function_symbol
}

///|
pub fn PagedAttentionCudaLowering::source_bytes(
  self : PagedAttentionCudaLowering,
) -> Bytes {
  self.source
}

///|
pub fn PagedAttentionCudaLowering::source_digest(
  self : PagedAttentionCudaLowering,
) -> PagedAttentionCudaSourceDigest {
  self.source_digest
}

///|
pub fn PagedAttentionCudaLowering::recipe_bytes(
  self : PagedAttentionCudaLowering,
) -> Bytes {
  self.recipe
}

///|
pub fn PagedAttentionCudaLowering::recipe_digest(
  self : PagedAttentionCudaLowering,
) -> PagedAttentionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn PagedAttentionManifestCudaLowering::candidate(
  self : PagedAttentionManifestCudaLowering,
) -> PagedAttentionCudaLowering {
  self.candidate
}

///|
pub fn PagedAttentionManifestCudaLowering::entry_point(
  self : PagedAttentionManifestCudaLowering,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn PagedAttentionManifestCudaLowering::manifest_bindable(
  _self : PagedAttentionManifestCudaLowering,
) -> Bool {
  true
}

///|
pub fn PagedAttentionManifestCudaLowering::binding_record_bytes(
  self : PagedAttentionManifestCudaLowering,
) -> Bytes {
  self.binding_record
}

///|
pub fn PagedAttentionManifestCudaLowering::binding_digest(
  self : PagedAttentionManifestCudaLowering,
) -> PagedAttentionManifestBindingDigest {
  self.binding_digest
}
