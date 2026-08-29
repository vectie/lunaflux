///|
pub fn Fp8ProjectionCudaLowering::model_identity(
  self : Fp8ProjectionCudaLowering,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn Fp8ProjectionCudaLowering::family(
  self : Fp8ProjectionCudaLowering,
) -> Fp8ProjectionFamily {
  self.family
}

///|
pub fn Fp8ProjectionCudaLowering::operation_id(
  self : Fp8ProjectionCudaLowering,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn Fp8ProjectionCudaLowering::execution_digest(
  self : Fp8ProjectionCudaLowering,
) -> @numeric_contract.OperationExecutionDigest {
  self.execution_digest
}

///|
pub fn Fp8ProjectionCudaLowering::entry_point_id(
  self : Fp8ProjectionCudaLowering,
) -> @catalog.AotKernelEntryPointId {
  self.entry_point_id
}

///|
pub fn Fp8ProjectionCudaLowering::target(
  self : Fp8ProjectionCudaLowering,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn Fp8ProjectionCudaLowering::profile(
  self : Fp8ProjectionCudaLowering,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn Fp8ProjectionCudaLowering::operands(
  self : Fp8ProjectionCudaLowering,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn Fp8ProjectionCudaLowering::launch_dimensions(
  self : Fp8ProjectionCudaLowering,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn Fp8ProjectionCudaLowering::numerical_contract(
  self : Fp8ProjectionCudaLowering,
) -> Fp8ProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn Fp8ProjectionCudaLowering::compiler_policy(
  self : Fp8ProjectionCudaLowering,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn Fp8ProjectionCudaLowering::function_symbol(
  self : Fp8ProjectionCudaLowering,
) -> String {
  self.function_symbol
}

///|
pub fn Fp8ProjectionCudaLowering::source_bytes(
  self : Fp8ProjectionCudaLowering,
) -> Bytes {
  self.source
}

///|
pub fn Fp8ProjectionCudaLowering::source_digest(
  self : Fp8ProjectionCudaLowering,
) -> Fp8ProjectionCudaSourceDigest {
  self.source_digest
}

///|
pub fn Fp8ProjectionCudaLowering::recipe_bytes(
  self : Fp8ProjectionCudaLowering,
) -> Bytes {
  self.recipe
}

///|
pub fn Fp8ProjectionCudaLowering::recipe_digest(
  self : Fp8ProjectionCudaLowering,
) -> Fp8ProjectionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn Fp8ProjectionManifestCudaLowering::candidate(
  self : Fp8ProjectionManifestCudaLowering,
) -> Fp8ProjectionCudaLowering {
  self.candidate
}

///|
pub fn Fp8ProjectionManifestCudaLowering::entry_point(
  self : Fp8ProjectionManifestCudaLowering,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8ProjectionManifestCudaLowering::binding_digest(
  self : Fp8ProjectionManifestCudaLowering,
) -> Fp8ProjectionManifestBindingDigest {
  self.binding_digest
}
