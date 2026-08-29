///|
pub(all) enum Fp8ProjectionFamily {
  QkvProjection
  DenseProjection
  GatedMlp
  LanguageModelHead
} derive(Debug, Eq)

///|
pub(all) enum Fp8ProjectionNumericalContract {
  DynamicPerTensorFiniteE4M3F32AccumulateBf16V1
  DynamicPerTensorFiniteE4M3F32SiluGatedBf16V1
} derive(Debug, Eq)

///|
pub struct Fp8ProjectionAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
}

///|
pub fn Fp8ProjectionAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
) -> Fp8ProjectionAotLimits raise Fp8ProjectionAotError {
  if max_source_bytes <= 0 ||
    max_source_bytes > 16 * 1024 * 1024 ||
    max_recipe_bytes <= 0 ||
    max_recipe_bytes > 16 * 1024 * 1024 {
    raise Invalid(Limits)
  }
  { max_source_bytes, max_recipe_bytes }
}

///|
pub struct Fp8ProjectionCudaSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8ProjectionCudaSourceDigest::as_hex(
  self : Fp8ProjectionCudaSourceDigest,
) -> String {
  self.0
}

///|
pub struct Fp8ProjectionCudaRecipeDigest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8ProjectionCudaRecipeDigest::as_hex(
  self : Fp8ProjectionCudaRecipeDigest,
) -> String {
  self.0
}

///|
pub struct Fp8ProjectionManifestBindingDigest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8ProjectionManifestBindingDigest::as_hex(
  self : Fp8ProjectionManifestBindingDigest,
) -> String {
  self.0
}

///|
pub struct Fp8ProjectionCudaLowering {
  priv model_identity : @spec.ModelIdentity
  priv family : Fp8ProjectionFamily
  priv operation_id : @plan.OperationId
  priv execution_digest : @numeric_contract.OperationExecutionDigest
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv input_width : Int
  priv output_width : Int
  priv intermediate_width : Int
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv numerical_contract : Fp8ProjectionNumericalContract
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : Fp8ProjectionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : Fp8ProjectionCudaRecipeDigest
} derive(Debug, Eq)

///|
pub struct Fp8ProjectionManifestCudaLowering {
  priv candidate : Fp8ProjectionCudaLowering
  priv entry_point : @catalog.AotKernelEntryPoint
  priv binding_digest : Fp8ProjectionManifestBindingDigest
} derive(Debug, Eq)
