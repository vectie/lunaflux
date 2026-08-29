///|
/// Offline and startup bounds for one deterministic greedy-sampling artifact.
pub struct GreedySamplingAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
  priv max_module_bytes : Int64
} derive(Debug, Eq)

///|
pub fn GreedySamplingAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
  max_module_bytes~ : Int64,
) -> GreedySamplingAotLimits raise SamplingBoundaryError {
  if max_source_bytes <= 0 || max_recipe_bytes <= 0 || max_module_bytes <= 0L {
    raise Invalid(ArtifactLimits)
  }
  { max_source_bytes, max_recipe_bytes, max_module_bytes }
}

///|
/// Exact deterministic CUDA source and recipe selected before compilation.
pub struct GreedySamplingCudaLowering {
  priv target : @catalog.DeviceTarget
  priv vocabulary_size : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
  priv source_sha256 : String
  priv recipe : Bytes
  priv recipe_sha256 : String
  priv compiler : @luna_specialization.LunaCompilerPolicy
} derive(Debug, Eq)

///|
pub fn GreedySamplingCudaLowering::source_bytes(
  self : GreedySamplingCudaLowering,
) -> Bytes {
  self.source
}

///|
pub fn GreedySamplingCudaLowering::source_sha256(
  self : GreedySamplingCudaLowering,
) -> String {
  self.source_sha256
}

///|
pub fn GreedySamplingCudaLowering::recipe_bytes(
  self : GreedySamplingCudaLowering,
) -> Bytes {
  self.recipe
}

///|
pub fn GreedySamplingCudaLowering::recipe_sha256(
  self : GreedySamplingCudaLowering,
) -> String {
  self.recipe_sha256
}

///|
pub fn GreedySamplingCudaLowering::function_symbol(
  self : GreedySamplingCudaLowering,
) -> String {
  self.function_symbol
}

///|
pub struct GreedySamplingCompileReceipt {
  priv source_sha256 : String
  priv recipe_sha256 : String
  priv toolchain_sha256 : String
  priv driver_sha256 : String
  priv first_module_sha256 : String
  priv second_module_sha256 : String
} derive(Debug, Eq)

///|
pub fn GreedySamplingCompileReceipt::from_offline_builder(
  source_sha256~ : String,
  recipe_sha256~ : String,
  toolchain_sha256~ : String,
  driver_sha256~ : String,
  first_module_sha256~ : String,
  second_module_sha256~ : String,
) -> GreedySamplingCompileReceipt raise SamplingBoundaryError {
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
/// Authenticated, deterministic CUBIN for the startup-selected greedy mode.
pub struct GreedySamplingCudaArtifact {
  priv target : @catalog.DeviceTarget
  priv vocabulary_size : Int
  priv maximum_rows : Int
  priv result_row_bytes : Int64
  priv function_symbol : String
  priv module_bytes : Bytes
  priv module_digest : @catalog.AotArtifactDigest
  priv source_sha256 : String
  priv recipe_sha256 : String
  priv driver_sha256 : String
} derive(Debug, Eq)

///|
pub fn GreedySamplingCudaArtifact::target(
  self : GreedySamplingCudaArtifact,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GreedySamplingCudaArtifact::vocabulary_size(
  self : GreedySamplingCudaArtifact,
) -> Int {
  self.vocabulary_size
}

///|
pub fn GreedySamplingCudaArtifact::maximum_rows(
  self : GreedySamplingCudaArtifact,
) -> Int {
  self.maximum_rows
}

///|
pub fn GreedySamplingCudaArtifact::result_row_bytes(
  self : GreedySamplingCudaArtifact,
) -> Int64 {
  self.result_row_bytes
}

///|
pub fn GreedySamplingCudaArtifact::function_symbol(
  self : GreedySamplingCudaArtifact,
) -> String {
  self.function_symbol
}

///|
pub fn GreedySamplingCudaArtifact::module_bytes(
  self : GreedySamplingCudaArtifact,
) -> Bytes {
  self.module_bytes
}

///|
pub fn GreedySamplingCudaArtifact::module_digest(
  self : GreedySamplingCudaArtifact,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}
