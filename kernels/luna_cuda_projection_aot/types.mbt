///|
pub(all) enum ProjectionAotFamily {
  QkvProjection
  DenseProjection
  GatedMlp
  LanguageModelHead
} derive(Debug, Eq)

///|
pub(all) enum ProjectionNumericalContract {
  OrderedF32AccumulationBf16RoundV1
  OrderedF32SiluGatedBf16RoundV1
  TensorCoreF32AccumulationBf16RoundV1
  TensorCoreF32SiluGatedBf16RoundV1
} derive(Debug, Eq)

///|
pub struct ProjectionAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
} derive(Debug, Eq)

///|
pub fn ProjectionAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
) -> ProjectionAotLimits raise ProjectionAotError {
  if max_source_bytes <= 0 || max_recipe_bytes <= 0 {
    raise Invalid(Limits)
  }
  { max_source_bytes, max_recipe_bytes }
}

///|
pub struct ProjectionCudaLowering {
  priv family : ProjectionAotFamily
  priv operation_id : @plan.OperationId
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv input_width : Int
  priv output_width : Int
  priv intermediate_width : Int
  priv dispatch_plan : @luna_projection_strategy.ProjectionDispatchPlan
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv numerical_contract : ProjectionNumericalContract
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : ProjectionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : ProjectionCudaRecipeDigest
} derive(Debug, Eq)

///|
pub fn ProjectionCudaLowering::family(
  self : ProjectionCudaLowering,
) -> ProjectionAotFamily {
  self.family
}

///|
/// Exact post-compile join between a deterministic source candidate and the
/// admitted full-graph contract that owns its content-addressed module.
pub struct ProjectionManifestCudaLowering {
  priv candidate : ProjectionCudaLowering
  priv entry_point : @catalog.AotKernelEntryPoint
  priv binding_record : Bytes
  priv binding_digest : ProjectionCudaManifestBindingDigest
} derive(Debug, Eq)

///|
pub fn ProjectionCudaLowering::operation_id(
  self : ProjectionCudaLowering,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn ProjectionCudaLowering::entry_point_id(
  self : ProjectionCudaLowering,
) -> @catalog.AotKernelEntryPointId {
  self.entry_point_id
}

///|
pub fn ProjectionCudaLowering::target(
  self : ProjectionCudaLowering,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn ProjectionCudaLowering::profile(
  self : ProjectionCudaLowering,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn ProjectionCudaLowering::input_width(
  self : ProjectionCudaLowering,
) -> Int {
  self.input_width
}

///|
pub fn ProjectionCudaLowering::output_width(
  self : ProjectionCudaLowering,
) -> Int {
  self.output_width
}

///|
pub fn ProjectionCudaLowering::intermediate_width(
  self : ProjectionCudaLowering,
) -> Int {
  self.intermediate_width
}

///|
/// Backend-neutral decode/prefill strategy selected before CUDA lowering.
pub fn ProjectionCudaLowering::dispatch_plan(
  self : ProjectionCudaLowering,
) -> @luna_projection_strategy.ProjectionDispatchPlan {
  self.dispatch_plan
}

///|
pub fn ProjectionCudaLowering::operands(
  self : ProjectionCudaLowering,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn ProjectionCudaLowering::launch_dimensions(
  self : ProjectionCudaLowering,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn ProjectionCudaLowering::numerical_contract(
  self : ProjectionCudaLowering,
) -> ProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn ProjectionCudaLowering::compiler_policy(
  self : ProjectionCudaLowering,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn ProjectionCudaLowering::function_symbol(
  self : ProjectionCudaLowering,
) -> String {
  self.function_symbol
}

///|
pub fn ProjectionCudaLowering::source_bytes(
  self : ProjectionCudaLowering,
) -> Bytes {
  self.source
}

///|
pub fn ProjectionCudaLowering::source_digest(
  self : ProjectionCudaLowering,
) -> ProjectionCudaSourceDigest {
  self.source_digest
}

///|
pub fn ProjectionCudaLowering::recipe_bytes(
  self : ProjectionCudaLowering,
) -> Bytes {
  self.recipe
}

///|
pub fn ProjectionCudaLowering::recipe_digest(
  self : ProjectionCudaLowering,
) -> ProjectionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn ProjectionManifestCudaLowering::candidate(
  self : ProjectionManifestCudaLowering,
) -> ProjectionCudaLowering {
  self.candidate
}

///|
pub fn ProjectionManifestCudaLowering::entry_point(
  self : ProjectionManifestCudaLowering,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn ProjectionManifestCudaLowering::manifest_bindable(
  _self : ProjectionManifestCudaLowering,
) -> Bool {
  true
}

///|
pub fn ProjectionManifestCudaLowering::binding_record_bytes(
  self : ProjectionManifestCudaLowering,
) -> Bytes {
  self.binding_record
}

///|
pub fn ProjectionManifestCudaLowering::binding_digest(
  self : ProjectionManifestCudaLowering,
) -> ProjectionCudaManifestBindingDigest {
  self.binding_digest
}
