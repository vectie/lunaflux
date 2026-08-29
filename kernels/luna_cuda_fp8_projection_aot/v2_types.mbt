///|
pub(all) enum Fp8ProjectionV2NumericalContract {
  ExternalDynamicPerTensorFiniteE4M3F32AccumulateBf16V2
  GatedExternalAndPostSiluBoundExpfFiniteE4M3F32AccumulateBf16V2
} derive(Debug, Eq)

///|
pub struct Fp8ProjectionCudaLoweringV2 {
  priv model_identity : @spec.ModelIdentity
  priv family : Fp8ProjectionFamily
  priv operation_id : @plan.OperationId
  priv execution_digest : @numeric_contract.OperationExecutionDigest
  priv activation_scale_policy : @numeric_contract.ActivationScalePolicy
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv numerical_contract : Fp8ProjectionV2NumericalContract
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : Fp8ProjectionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : Fp8ProjectionCudaRecipeDigest
  priv workspace_bytes : Int64
} derive(Debug, Eq)

///|
pub struct Fp8ProjectionManifestCudaLoweringV2 {
  priv candidate : Fp8ProjectionCudaLoweringV2
  priv entry_point : @catalog.AotKernelEntryPoint
  priv binding_digest : Fp8ProjectionManifestBindingDigest
} derive(Debug, Eq)

///|
pub fn Fp8ProjectionCudaLoweringV2::model_identity(
  self : Fp8ProjectionCudaLoweringV2,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn Fp8ProjectionCudaLoweringV2::family(
  self : Fp8ProjectionCudaLoweringV2,
) -> Fp8ProjectionFamily {
  self.family
}

///|
pub fn Fp8ProjectionCudaLoweringV2::operation_id(
  self : Fp8ProjectionCudaLoweringV2,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn Fp8ProjectionCudaLoweringV2::execution_digest(
  self : Fp8ProjectionCudaLoweringV2,
) -> @numeric_contract.OperationExecutionDigest {
  self.execution_digest
}

///|
pub fn Fp8ProjectionCudaLoweringV2::activation_scale_policy(
  self : Fp8ProjectionCudaLoweringV2,
) -> @numeric_contract.ActivationScalePolicy {
  self.activation_scale_policy
}

///|
pub fn Fp8ProjectionCudaLoweringV2::entry_point_id(
  self : Fp8ProjectionCudaLoweringV2,
) -> @catalog.AotKernelEntryPointId {
  self.entry_point_id
}

///|
pub fn Fp8ProjectionCudaLoweringV2::target(
  self : Fp8ProjectionCudaLoweringV2,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn Fp8ProjectionCudaLoweringV2::profile(
  self : Fp8ProjectionCudaLoweringV2,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn Fp8ProjectionCudaLoweringV2::operands(
  self : Fp8ProjectionCudaLoweringV2,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn Fp8ProjectionCudaLoweringV2::launch_dimensions(
  self : Fp8ProjectionCudaLoweringV2,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn Fp8ProjectionCudaLoweringV2::numerical_contract(
  self : Fp8ProjectionCudaLoweringV2,
) -> Fp8ProjectionV2NumericalContract {
  self.numerical_contract
}

///|
pub fn Fp8ProjectionCudaLoweringV2::compiler_policy(
  self : Fp8ProjectionCudaLoweringV2,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn Fp8ProjectionCudaLoweringV2::function_symbol(
  self : Fp8ProjectionCudaLoweringV2,
) -> String {
  self.function_symbol
}

///|
pub fn Fp8ProjectionCudaLoweringV2::source_bytes(
  self : Fp8ProjectionCudaLoweringV2,
) -> Bytes {
  self.source
}

///|
pub fn Fp8ProjectionCudaLoweringV2::source_digest(
  self : Fp8ProjectionCudaLoweringV2,
) -> Fp8ProjectionCudaSourceDigest {
  self.source_digest
}

///|
pub fn Fp8ProjectionCudaLoweringV2::recipe_bytes(
  self : Fp8ProjectionCudaLoweringV2,
) -> Bytes {
  self.recipe
}

///|
pub fn Fp8ProjectionCudaLoweringV2::recipe_digest(
  self : Fp8ProjectionCudaLoweringV2,
) -> Fp8ProjectionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn Fp8ProjectionCudaLoweringV2::workspace_bytes(
  self : Fp8ProjectionCudaLoweringV2,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn Fp8ProjectionManifestCudaLoweringV2::candidate(
  self : Fp8ProjectionManifestCudaLoweringV2,
) -> Fp8ProjectionCudaLoweringV2 {
  self.candidate
}

///|
pub fn Fp8ProjectionManifestCudaLoweringV2::entry_point(
  self : Fp8ProjectionManifestCudaLoweringV2,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8ProjectionManifestCudaLoweringV2::binding_digest(
  self : Fp8ProjectionManifestCudaLoweringV2,
) -> Fp8ProjectionManifestBindingDigest {
  self.binding_digest
}
