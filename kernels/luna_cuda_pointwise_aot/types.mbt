///|
pub(all) enum PointwiseAotFamily {
  EmbeddingLookup
  RmsNorm
  QkRmsNorm
  PositionedRotary
  ResidualAdd
  KvCacheWrite
} derive(Debug, Eq)

///|
pub(all) enum PointwiseNumericalContract {
  BitExactBf16Copy
  OrderedBf16Add
  Float32ReductionToleranceV1
  Float32TranscendentalToleranceV1
} derive(Debug, Eq)

///|
pub struct PointwiseAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
} derive(Debug, Eq)

///|
pub fn PointwiseAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
) -> PointwiseAotLimits raise PointwiseAotError {
  if max_source_bytes <= 0 || max_recipe_bytes <= 0 {
    raise Invalid(Limits)
  }
  { max_source_bytes, max_recipe_bytes }
}

///|
pub struct PointwiseCudaLowering {
  priv family : PointwiseAotFamily
  priv operation_id : @plan.OperationId
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv numerical_contract : PointwiseNumericalContract
  priv manifest_bindable : Bool
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : PointwiseSourceDigest
  priv recipe : Bytes
  priv recipe_digest : PointwiseRecipeDigest
} derive(Debug, Eq)

///|
/// Exact post-artifact join between one deterministic lowering candidate and
/// the admitted full-graph launch contract that names its module family.
pub struct PointwiseManifestCudaLowering {
  priv candidate : PointwiseCudaLowering
  priv entry_point : @catalog.AotKernelEntryPoint
  priv binding_record : Bytes
  priv binding_digest : PointwiseManifestBindingDigest
} derive(Debug, Eq)

///|
pub fn PointwiseCudaLowering::family(
  self : PointwiseCudaLowering,
) -> PointwiseAotFamily {
  self.family
}

///|
pub fn PointwiseCudaLowering::operation_id(
  self : PointwiseCudaLowering,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn PointwiseCudaLowering::entry_point_id(
  self : PointwiseCudaLowering,
) -> @catalog.AotKernelEntryPointId {
  self.entry_point_id
}

///|
pub fn PointwiseCudaLowering::target(
  self : PointwiseCudaLowering,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn PointwiseCudaLowering::profile(
  self : PointwiseCudaLowering,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn PointwiseCudaLowering::operands(
  self : PointwiseCudaLowering,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn PointwiseCudaLowering::launch_dimensions(
  self : PointwiseCudaLowering,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn PointwiseCudaLowering::numerical_contract(
  self : PointwiseCudaLowering,
) -> PointwiseNumericalContract {
  self.numerical_contract
}

///|
pub fn PointwiseCudaLowering::manifest_bindable(
  self : PointwiseCudaLowering,
) -> Bool {
  self.manifest_bindable
}

///|
pub fn PointwiseCudaLowering::function_symbol(
  self : PointwiseCudaLowering,
) -> String {
  self.function_symbol
}

///|
pub fn PointwiseCudaLowering::source_bytes(
  self : PointwiseCudaLowering,
) -> Bytes {
  self.source
}

///|
pub fn PointwiseCudaLowering::source_digest(
  self : PointwiseCudaLowering,
) -> PointwiseSourceDigest {
  self.source_digest
}

///|
pub fn PointwiseCudaLowering::recipe_bytes(
  self : PointwiseCudaLowering,
) -> Bytes {
  self.recipe
}

///|
pub fn PointwiseCudaLowering::recipe_digest(
  self : PointwiseCudaLowering,
) -> PointwiseRecipeDigest {
  self.recipe_digest
}

///|
pub fn PointwiseManifestCudaLowering::candidate(
  self : PointwiseManifestCudaLowering,
) -> PointwiseCudaLowering {
  self.candidate
}

///|
pub fn PointwiseManifestCudaLowering::entry_point(
  self : PointwiseManifestCudaLowering,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn PointwiseManifestCudaLowering::manifest_bindable(
  _self : PointwiseManifestCudaLowering,
) -> Bool {
  true
}

///|
pub fn PointwiseManifestCudaLowering::binding_record_bytes(
  self : PointwiseManifestCudaLowering,
) -> Bytes {
  self.binding_record
}

///|
pub fn PointwiseManifestCudaLowering::binding_digest(
  self : PointwiseManifestCudaLowering,
) -> PointwiseManifestBindingDigest {
  self.binding_digest
}
