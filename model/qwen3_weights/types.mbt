///|
/// Exact validation rule rejected while binding dense Qwen3 BF16 weights.
pub(all) enum Qwen3WeightRule {
  TensorCount
  TensorName
  TensorDType
  TensorRank
  TensorDimension
  TensorByteRange
  TiedOutput
  TiedOutputByteEquality
  RouteManifest
} derive(Debug, Eq)

///|
/// Payload-safe dense Qwen3 weight-binding failures.
pub(all) suberror Qwen3WeightsError {
  Mismatch(Qwen3WeightRule)
  UnknownTensor(index~ : Int)
  MissingTensor(reference~ : Int)
  InvalidTensor(reference~ : Int, Qwen3WeightRule)
  InvalidModelPlan(@plan.PlanValidationError)
  InvalidNumericWeightFile(@numeric_weight_file.NumericWeightFileError)
  InvalidSourceSafetensors(@safetensors_reader.SafetensorsReaderError)
  InvalidSourceFileLimit
  SourceFileUnavailable
  SourceFileCloseFailed
  SourceFileTooLarge
  SourceFileChanged
  SourceArtifactDigestMismatch
} derive(Debug, Eq)

///|
/// One immutable mapping from the future Qwen3 semantic tensor order to
/// validated safetensors metadata.
pub struct Qwen3TensorBinding {
  priv reference : @plan.TensorRef
  priv tensor : @safetensors.TensorMetadata
} derive(Debug, Eq)

///|
/// Complete unique-storage weight mapping for one dense Qwen3 model.
///
/// Qwen3's language-model head aliases reference zero, the embedding table.
/// A redundant physical `lm_head.weight` in an authenticated source artifact
/// is validated but deliberately excluded from the materialization byte count.
pub struct Qwen3WeightBindings {
  priv content_digest : @spec.ContentDigest
  priv bindings : ReadOnlyArray[Qwen3TensorBinding]
  priv source_tensor_count : Int
  priv source_bytes : Int64
  priv materialized_bytes : Int64
  priv redundant_lm_head : @safetensors.TensorMetadata?
} derive(Debug, Eq)

///|
/// Startup bounds for offline admission of an original Qwen3 safetensors file.
pub struct Qwen3SourceFileLimits {
  priv max_file_bytes : Int64
  priv compare_chunk_bytes : Int
  priv safetensors : @safetensors.HeaderLimits
} derive(Debug, Eq)

///|
/// Inert proof that one original dense-Qwen3 artifact has the exact tensor
/// vocabulary and, when present, byte-identical redundant tied-output storage.
pub struct Qwen3SourceFileInspection {
  priv bindings : Qwen3WeightBindings
  priv artifact_sha256 : String
  priv source_manifest_sha256 : String
}

///|
pub fn Qwen3SourceFileLimits::new(
  max_file_bytes~ : Int64,
  compare_chunk_bytes~ : Int,
  safetensors~ : @safetensors.HeaderLimits,
) -> Self raise Qwen3WeightsError {
  if max_file_bytes <= 8L ||
    compare_chunk_bytes <= 0 ||
    compare_chunk_bytes > 16 * 1024 * 1024 {
    raise InvalidSourceFileLimit
  }
  { max_file_bytes, compare_chunk_bytes, safetensors }
}

///|
pub fn Qwen3SourceFileInspection::bindings(self : Self) -> Qwen3WeightBindings {
  self.bindings
}

///|
pub fn Qwen3SourceFileInspection::artifact_sha256(self : Self) -> String {
  self.artifact_sha256
}

///|
pub fn Qwen3SourceFileInspection::source_manifest_sha256(self : Self) -> String {
  self.source_manifest_sha256
}

///|
pub fn Qwen3TensorBinding::reference(self : Self) -> @plan.TensorRef {
  self.reference
}

///|
/// Authenticated canonical numeric file bound to the exact Qwen3 route.
pub struct Qwen3WeightFileInspection {
  priv inspection : @numeric_weight_file.NumericWeightFileInspection
  priv route_manifest_sha256 : String
}

///|
pub fn Qwen3WeightFileInspection::route_manifest_sha256(self : Self) -> String {
  self.route_manifest_sha256
}

///|
pub fn Qwen3WeightFileInspection::generic_inspection(
  self : Self,
) -> @numeric_weight_file.NumericWeightFileInspection {
  self.inspection
}

///|
pub fn Qwen3WeightFileInspection::layout(
  self : Self,
) -> @device_materialize.DeviceWeightLayout {
  self.inspection.layout()
}

///|
pub fn Qwen3WeightFileInspection::identity(self : Self) -> @spec.ModelIdentity {
  self.inspection.identity()
}

///|
pub fn Qwen3WeightFileInspection::artifact_digest(
  self : Self,
) -> @numeric_weight_file.NumericWeightArtifactDigest {
  self.inspection.artifact_digest()
}

///|
pub fn Qwen3WeightFileInspection::numeric_schema_digest(
  self : Self,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.inspection.numeric_schema_digest()
}

///|
pub fn Qwen3TensorBinding::tensor(self : Self) -> @safetensors.TensorMetadata {
  self.tensor
}

///|
pub fn Qwen3WeightBindings::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn Qwen3WeightBindings::length(self : Self) -> Int {
  self.bindings.length()
}

///|
pub fn Qwen3WeightBindings::binding(
  self : Self,
  reference : @plan.TensorRef,
) -> Qwen3TensorBinding? {
  self.bindings.get(reference.as_int())
}

///|
/// Returns the unique embedding storage binding.
pub fn Qwen3WeightBindings::embedding_binding(
  self : Self,
) -> Qwen3TensorBinding {
  self.bindings[0]
}

///|
/// Returns the language-model-head semantic alias.
///
/// This is exactly the embedding binding, including `TensorRef(0)` and source
/// byte range; no second device region is implied.
pub fn Qwen3WeightBindings::lm_head_binding(self : Self) -> Qwen3TensorBinding {
  self.bindings[0]
}

///|
pub fn Qwen3WeightBindings::source_tensor_count(self : Self) -> Int {
  self.source_tensor_count
}

///|
pub fn Qwen3WeightBindings::source_bytes(self : Self) -> Int64 {
  self.source_bytes
}

///|
/// Bytes required by the unique semantic tensor regions.
pub fn Qwen3WeightBindings::materialized_bytes(self : Self) -> Int64 {
  self.materialized_bytes
}

///|
/// Returns a validated but redundant tied-output copy when the source file
/// carries one. The copy grants no separate semantic tensor reference.
pub fn Qwen3WeightBindings::redundant_lm_head_tensor(
  self : Self,
) -> @safetensors.TensorMetadata? {
  self.redundant_lm_head
}
