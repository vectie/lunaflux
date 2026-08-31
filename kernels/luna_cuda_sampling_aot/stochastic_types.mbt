///|
/// Offline and startup bounds for one deterministic device-sampling artifact.
pub struct DeviceSamplingAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
  priv max_module_bytes : Int64
} derive(Debug, Eq)

///|
pub fn DeviceSamplingAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
  max_module_bytes~ : Int64,
) -> DeviceSamplingAotLimits raise SamplingBoundaryError {
  if max_source_bytes <= 0 || max_recipe_bytes <= 0 || max_module_bytes <= 0L {
    raise Invalid(ArtifactLimits)
  }
  { max_source_bytes, max_recipe_bytes, max_module_bytes }
}

///|
/// Exact CUDA lowering for restrictive-top-k stochastic sampling.
pub struct DeviceSamplingCudaLowering {
  priv target : @catalog.DeviceTarget
  priv plan : @luna_sampling_strategy.SamplingDevicePlan
  priv function_symbol : String
  priv source : Bytes
  priv source_sha256 : String
  priv recipe : Bytes
  priv recipe_sha256 : String
  priv compiler : @luna_specialization.LunaCompilerPolicy
} derive(Debug, Eq)

///|
pub fn DeviceSamplingCudaLowering::target(
  self : DeviceSamplingCudaLowering,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn DeviceSamplingCudaLowering::plan(
  self : DeviceSamplingCudaLowering,
) -> @luna_sampling_strategy.SamplingDevicePlan {
  self.plan
}

///|
pub fn DeviceSamplingCudaLowering::function_symbol(
  self : DeviceSamplingCudaLowering,
) -> String {
  self.function_symbol
}

///|
pub fn DeviceSamplingCudaLowering::source_bytes(
  self : DeviceSamplingCudaLowering,
) -> Bytes {
  self.source
}

///|
pub fn DeviceSamplingCudaLowering::source_sha256(
  self : DeviceSamplingCudaLowering,
) -> String {
  self.source_sha256
}

///|
pub fn DeviceSamplingCudaLowering::recipe_bytes(
  self : DeviceSamplingCudaLowering,
) -> Bytes {
  self.recipe
}

///|
pub fn DeviceSamplingCudaLowering::recipe_sha256(
  self : DeviceSamplingCudaLowering,
) -> String {
  self.recipe_sha256
}

///|
pub struct DeviceSamplingCompileReceipt {
  priv source_sha256 : String
  priv recipe_sha256 : String
  priv toolchain_sha256 : String
  priv driver_sha256 : String
  priv first_module_sha256 : String
  priv second_module_sha256 : String
} derive(Debug, Eq)

///|
pub fn DeviceSamplingCompileReceipt::from_offline_builder(
  source_sha256~ : String,
  recipe_sha256~ : String,
  toolchain_sha256~ : String,
  driver_sha256~ : String,
  first_module_sha256~ : String,
  second_module_sha256~ : String,
) -> DeviceSamplingCompileReceipt raise SamplingBoundaryError {
  if !sampling_is_sha256(source_sha256) ||
    !sampling_is_sha256(recipe_sha256) ||
    !sampling_is_sha256(toolchain_sha256) ||
    !sampling_is_sha256(driver_sha256) ||
    !sampling_is_sha256(first_module_sha256) ||
    !sampling_is_sha256(second_module_sha256) {
    raise Invalid(ArtifactDigest)
  }
  {
    source_sha256,
    recipe_sha256,
    toolchain_sha256,
    driver_sha256,
    first_module_sha256,
    second_module_sha256,
  }
}

///|
/// Deterministic standalone module candidate. It is intentionally not
/// manifest-bindable until the paged graph schema carries parameter and
/// workspace operands for this versioned ABI.
pub struct DeviceSamplingCudaArtifact {
  priv target : @catalog.DeviceTarget
  priv plan : @luna_sampling_strategy.SamplingDevicePlan
  priv function_symbol : String
  priv module_bytes : Bytes
  priv module_digest : @catalog.AotArtifactDigest
  priv source_sha256 : String
  priv recipe_sha256 : String
  priv driver_sha256 : String
} derive(Debug, Eq)

///|
pub fn DeviceSamplingCudaArtifact::target(
  self : DeviceSamplingCudaArtifact,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn DeviceSamplingCudaArtifact::plan(
  self : DeviceSamplingCudaArtifact,
) -> @luna_sampling_strategy.SamplingDevicePlan {
  self.plan
}

///|
pub fn DeviceSamplingCudaArtifact::function_symbol(
  self : DeviceSamplingCudaArtifact,
) -> String {
  self.function_symbol
}

///|
pub fn DeviceSamplingCudaArtifact::module_bytes(
  self : DeviceSamplingCudaArtifact,
) -> Bytes {
  self.module_bytes
}

///|
pub fn DeviceSamplingCudaArtifact::module_digest(
  self : DeviceSamplingCudaArtifact,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn DeviceSamplingCudaArtifact::manifest_bindable(
  _self : DeviceSamplingCudaArtifact,
) -> Bool {
  false
}

///|
/// Exact ordered operands for the standalone v2 launch ABI.
pub(all) enum DeviceSamplingCudaOperandV2 {
  SamplingCounts
  SamplingRowOffsets
  SamplingBf16Logits
  SamplingParametersV2
  SamplingWorkspace
  SamplingResults
} derive(Debug, Eq)

///|
/// Startup-sized launch seam for a future paged-manifest revision.
pub struct DeviceSamplingCudaAbiV2 {
  priv function_symbol : String
  priv operands : ReadOnlyArray[DeviceSamplingCudaOperandV2]
  priv grid_x : Int
  priv block_x : Int
  priv parameter_bytes : Int64
  priv workspace_bytes : Int64
  priv result_bytes : Int64
} derive(Debug, Eq)

///|
pub fn DeviceSamplingCudaAbiV2::function_symbol(
  self : DeviceSamplingCudaAbiV2,
) -> String {
  self.function_symbol
}

///|
pub fn DeviceSamplingCudaAbiV2::operands(
  self : DeviceSamplingCudaAbiV2,
) -> ReadOnlyArray[DeviceSamplingCudaOperandV2] {
  self.operands
}

///|
pub fn DeviceSamplingCudaAbiV2::grid_x(self : DeviceSamplingCudaAbiV2) -> Int {
  self.grid_x
}

///|
pub fn DeviceSamplingCudaAbiV2::block_x(self : DeviceSamplingCudaAbiV2) -> Int {
  self.block_x
}

///|
pub fn DeviceSamplingCudaAbiV2::parameter_bytes(
  self : DeviceSamplingCudaAbiV2,
) -> Int64 {
  self.parameter_bytes
}

///|
pub fn DeviceSamplingCudaAbiV2::workspace_bytes(
  self : DeviceSamplingCudaAbiV2,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn DeviceSamplingCudaAbiV2::result_bytes(
  self : DeviceSamplingCudaAbiV2,
) -> Int64 {
  self.result_bytes
}

///|
pub fn DeviceSamplingCudaAbiV2::manifest_bindable(
  _self : DeviceSamplingCudaAbiV2,
) -> Bool {
  false
}
