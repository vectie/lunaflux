///|
pub enum MistralWeightRule {
  ModelContent
  ModelPlan
  TensorCount
  TensorName
  TensorDType
  TensorRank
  TensorDimension
} derive(Debug, Eq)

///|
pub(all) suberror MistralWeightsError {
  Mismatch(MistralWeightRule)
  MissingTensor(reference~ : Int)
  InvalidTensor(reference~ : Int, MistralWeightRule)
  InvalidModelPlan(@plan.PlanValidationError)
  InvalidNumericWeightFile(@numeric_weight_file.NumericWeightFileError)
} derive(Debug, Eq)

///|
pub struct MistralWeightBinding {
  priv reference : @plan.TensorRef
  priv tensor : @safetensors.TensorMetadata
} derive(Debug, Eq)

///|
pub struct MistralWeightManifest {
  priv identity : @spec.ModelIdentity
  priv bindings : ReadOnlyArray[MistralWeightBinding]
  priv materialized_bytes : Int64
  priv manifest_sha256 : String
} derive(Debug, Eq)

///|
pub fn MistralWeightBinding::reference(self : Self) -> @plan.TensorRef {
  self.reference
}

///|
pub fn MistralWeightBinding::tensor(self : Self) -> @safetensors.TensorMetadata {
  self.tensor
}

///|
pub fn MistralWeightManifest::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn MistralWeightManifest::length(self : Self) -> Int {
  self.bindings.length()
}

///|
pub fn MistralWeightManifest::binding(
  self : Self,
  reference : @plan.TensorRef,
) -> MistralWeightBinding? {
  self.bindings.get(reference.as_int())
}

///|
pub fn MistralWeightManifest::materialized_bytes(self : Self) -> Int64 {
  self.materialized_bytes
}

///|
pub fn MistralWeightManifest::manifest_sha256(self : Self) -> String {
  self.manifest_sha256
}

///|
pub struct MistralWeightFileInspection {
  priv inspection : @numeric_weight_file.NumericWeightFileInspection
  priv route_manifest_sha256 : String
}

///|
pub fn MistralWeightFileInspection::route_manifest_sha256(
  self : Self,
) -> String {
  self.route_manifest_sha256
}

///|
/// Projects the already-authenticated generic inspection for the shared dense
/// numeric worker. This grants no allocation or readiness authority.
pub fn MistralWeightFileInspection::generic_inspection(
  self : Self,
) -> @numeric_weight_file.NumericWeightFileInspection {
  self.inspection
}

///|
pub fn MistralWeightFileInspection::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.inspection.identity()
}

///|
pub fn MistralWeightFileInspection::artifact_digest(
  self : Self,
) -> @numeric_weight_file.NumericWeightArtifactDigest {
  self.inspection.artifact_digest()
}

///|
pub fn MistralWeightFileInspection::layout(
  self : Self,
) -> @device_materialize.DeviceWeightLayout {
  self.inspection.layout()
}

///|
pub fn MistralWeightFileInspection::numeric_schema_digest(
  self : Self,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.inspection.numeric_schema_digest()
}
