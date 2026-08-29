///|
const NUMERIC_WEIGHT_FILE_FORMAT : String = "lunaflux.numeric-safetensors.v1"

///|
const NUMERIC_WEIGHT_MAX_CHUNK_BYTES : Int = 16 * 1024 * 1024

///|
/// One F32 scale scalar is the minimum independently validated payload unit.
const NUMERIC_WEIGHT_MIN_CHUNK_BYTES : Int = 4

///|
#valtype
pub struct NumericWeightArtifactDigest {
  priv value : String
} derive(Debug, Eq, Hash)

///|
pub fn NumericWeightArtifactDigest::from_sha256(
  value : StringView,
) -> NumericWeightArtifactDigest raise NumericWeightFileError {
  if value.length() != 64 ||
    !value.all(c => c.is_ascii_digit() || (c >= 'a' && c <= 'f')) {
    raise InvalidArtifactDigest
  }
  { value: value.to_owned() }
}

///|
pub fn NumericWeightArtifactDigest::as_hex(
  self : NumericWeightArtifactDigest,
) -> String {
  self.value
}

///|
pub struct NumericWeightFileLimits {
  priv max_file_bytes : Int64
  priv max_header_bytes : Int
  priv transfer_chunk_bytes : Int
  priv max_tensor_count : Int
}

///|
pub fn NumericWeightFileLimits::new(
  max_file_bytes~ : Int64,
  max_header_bytes~ : Int,
  transfer_chunk_bytes~ : Int,
  max_tensor_count~ : Int,
) -> NumericWeightFileLimits raise NumericWeightFileError {
  if max_file_bytes <= 8L {
    raise InvalidLimit(FileBytes)
  }
  if max_header_bytes <= 0 || max_header_bytes > 16 * 1024 * 1024 {
    raise InvalidLimit(HeaderBytes)
  }
  if transfer_chunk_bytes < NUMERIC_WEIGHT_MIN_CHUNK_BYTES ||
    transfer_chunk_bytes > NUMERIC_WEIGHT_MAX_CHUNK_BYTES {
    raise InvalidLimit(TransferChunkBytes)
  }
  if max_tensor_count <= 0 || max_tensor_count > 65536 {
    raise InvalidLimit(TensorCount)
  }
  { max_file_bytes, max_header_bytes, transfer_chunk_bytes, max_tensor_count }
}

///|
pub fn NumericWeightFileLimits::max_file_bytes(
  self : NumericWeightFileLimits,
) -> Int64 {
  self.max_file_bytes
}

///|
pub fn NumericWeightFileLimits::max_header_bytes(
  self : NumericWeightFileLimits,
) -> Int {
  self.max_header_bytes
}

///|
pub fn NumericWeightFileLimits::transfer_chunk_bytes(
  self : NumericWeightFileLimits,
) -> Int {
  self.transfer_chunk_bytes
}

///|
pub fn NumericWeightFileLimits::max_tensor_count(
  self : NumericWeightFileLimits,
) -> Int {
  self.max_tensor_count
}

///|
priv struct NumericTransferRegion {
  ordinal : Int
  source_offset_bytes : Int64
  destination_offset_bytes : Int64
  byte_length : Int64
} derive(Eq)

///|
priv struct PreparedNumericWeightFile[S] {
  stamp : S
  payload_offset_bytes : Int64
  artifact_digest : NumericWeightArtifactDigest
  layout : @device_materialize.DeviceWeightLayout
  regions : ReadOnlyArray[NumericTransferRegion]
  limits : NumericWeightFileLimits
  scratch : FixedArray[Byte]
}

///|
/// Inert evidence from a complete authenticated numeric-file admission.
/// This value owns no allocation and grants no device or kernel readiness.
pub struct NumericWeightFileInspection {
  priv model_plan : @plan.ModelPlan
  priv locator : @approved_fs.ApprovedRelativeLocator
  priv expected_artifact_digest : NumericWeightArtifactDigest
  priv limits : NumericWeightFileLimits
  priv alignment_bytes : Int64
  priv max_arena_bytes : Int64
  priv payload_offset_bytes : Int64
  priv numeric_schema_digest : @numeric_contract.ModelNumericSchemaDigest
  priv layout : @device_materialize.DeviceWeightLayout
  priv regions : ReadOnlyArray[NumericTransferRegion]
}

///|
pub fn NumericWeightFileInspection::artifact_digest(
  self : NumericWeightFileInspection,
) -> NumericWeightArtifactDigest {
  self.expected_artifact_digest
}

///|
pub fn NumericWeightFileInspection::identity(
  self : NumericWeightFileInspection,
) -> @spec.ModelIdentity {
  self.layout.identity()
}

///|
pub fn NumericWeightFileInspection::numeric_schema_digest(
  self : NumericWeightFileInspection,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema_digest
}

///|
pub fn NumericWeightFileInspection::layout(
  self : NumericWeightFileInspection,
) -> @device_materialize.DeviceWeightLayout {
  self.layout
}

///|
/// Explicit owner of one fully authenticated final numeric device arena.
pub struct NumericDeviceWeights {
  priv allocation : @device.Allocation
  priv artifact_digest : NumericWeightArtifactDigest
  priv numeric_schema_digest : @numeric_contract.ModelNumericSchemaDigest
  priv layout : @device_materialize.DeviceWeightLayout
  priv mut open : Bool
}

///|
/// Exact authenticated join consumed by numeric executors and blueprints.
/// A bare allocation is never obtainable from the unjoined weight owner.
pub struct AuthenticatedNumericWeightAuthority {
  priv weights : NumericDeviceWeights
}

///|
pub(all) enum NumericDeviceWeightPreparation {
  NumericWeightsReady(NumericDeviceWeights)
  NumericWeightsCleanupRequired(FailedNumericDeviceWeightPreparation)
}

///|
pub struct FailedNumericDeviceWeightPreparation {
  priv allocation : @device.Allocation
  priv cause : NumericWeightFileError
  priv stage : NumericWeightFailureStage
  priv mut cleanup_failure : @device.DeviceError
  priv mut open : Bool
}

///|
pub fn NumericDeviceWeights::artifact_digest(
  self : NumericDeviceWeights,
) -> NumericWeightArtifactDigest {
  self.artifact_digest
}

///|
pub fn NumericDeviceWeights::identity(
  self : NumericDeviceWeights,
) -> @spec.ModelIdentity {
  self.layout.identity()
}

///|
pub fn NumericDeviceWeights::numeric_schema_digest(
  self : NumericDeviceWeights,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema_digest
}

///|
pub fn NumericDeviceWeights::layout(
  self : NumericDeviceWeights,
) -> @device_materialize.DeviceWeightLayout {
  self.layout
}

///|
fn require_numeric_weights_open(
  open : Bool,
) -> Unit raise NumericWeightFileError {
  if !open {
    raise Closed
  }
}

///|
fn validate_numeric_weight_authority(
  open : Bool,
  identity : @spec.ModelIdentity,
  schema_digest : @numeric_contract.ModelNumericSchemaDigest,
  layout : @device_materialize.DeviceWeightLayout,
  artifact_digest : NumericWeightArtifactDigest,
  expected_identity : @spec.ModelIdentity,
  expected_schema_digest : @numeric_contract.ModelNumericSchemaDigest,
  expected_layout : @device_materialize.DeviceWeightLayout,
  expected_artifact_digest : NumericWeightArtifactDigest,
) -> Unit raise NumericWeightFileError {
  require_numeric_weights_open(open)
  if identity != expected_identity {
    raise AuthorityMismatch(Identity)
  }
  if schema_digest != expected_schema_digest {
    raise AuthorityMismatch(NumericSchema)
  }
  if layout != expected_layout {
    raise AuthorityMismatch(Layout)
  }
  if artifact_digest != expected_artifact_digest {
    raise AuthorityMismatch(Artifact)
  }
}

///|
/// Mints execution-facing allocation authority only after an exact four-way
/// join of model, schema, physical layout, and independently pinned artifact.
pub fn NumericDeviceWeights::authenticate(
  self : NumericDeviceWeights,
  expected_identity~ : @spec.ModelIdentity,
  expected_schema_digest~ : @numeric_contract.ModelNumericSchemaDigest,
  expected_layout~ : @device_materialize.DeviceWeightLayout,
  expected_artifact_digest~ : NumericWeightArtifactDigest,
) -> AuthenticatedNumericWeightAuthority raise NumericWeightFileError {
  validate_numeric_weight_authority(
    self.open,
    self.identity(),
    self.numeric_schema_digest(),
    self.layout,
    self.artifact_digest,
    expected_identity,
    expected_schema_digest,
    expected_layout,
    expected_artifact_digest,
  )
  { weights: self }
}

///|
pub fn AuthenticatedNumericWeightAuthority::artifact_digest(
  self : AuthenticatedNumericWeightAuthority,
) -> NumericWeightArtifactDigest {
  self.weights.artifact_digest()
}

///|
pub fn AuthenticatedNumericWeightAuthority::identity(
  self : AuthenticatedNumericWeightAuthority,
) -> @spec.ModelIdentity {
  self.weights.identity()
}

///|
pub fn AuthenticatedNumericWeightAuthority::numeric_schema_digest(
  self : AuthenticatedNumericWeightAuthority,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.weights.numeric_schema_digest()
}

///|
pub fn AuthenticatedNumericWeightAuthority::layout(
  self : AuthenticatedNumericWeightAuthority,
) -> @device_materialize.DeviceWeightLayout {
  self.weights.layout()
}

///|
/// Returns the safe allocation only after the exact authenticated join.
pub fn AuthenticatedNumericWeightAuthority::allocation(
  self : AuthenticatedNumericWeightAuthority,
) -> @device.Allocation raise NumericWeightFileError {
  require_numeric_weights_open(self.weights.open)
  self.weights.allocation
}

///|
pub fn AuthenticatedNumericWeightAuthority::close(
  self : AuthenticatedNumericWeightAuthority,
) -> Unit raise NumericWeightFileError {
  self.weights.close()
}

///|
pub fn NumericDeviceWeights::close(
  self : NumericDeviceWeights,
) -> Unit raise NumericWeightFileError {
  if !self.open {
    return
  }
  self.allocation.close() catch {
    error => raise DeviceFailed(error)
  }
  self.open = false
}

///|
pub fn NumericDeviceWeights::is_closed(self : NumericDeviceWeights) -> Bool {
  !self.open
}

///|
pub fn FailedNumericDeviceWeightPreparation::cause(
  self : FailedNumericDeviceWeightPreparation,
) -> NumericWeightFileError {
  self.cause
}

///|
pub fn FailedNumericDeviceWeightPreparation::stage(
  self : FailedNumericDeviceWeightPreparation,
) -> NumericWeightFailureStage {
  self.stage
}

///|
pub fn FailedNumericDeviceWeightPreparation::cleanup_failure(
  self : FailedNumericDeviceWeightPreparation,
) -> @device.DeviceError {
  self.cleanup_failure
}

///|
pub fn FailedNumericDeviceWeightPreparation::is_closed(
  self : FailedNumericDeviceWeightPreparation,
) -> Bool {
  !self.open
}

///|
pub fn FailedNumericDeviceWeightPreparation::retry_close(
  self : FailedNumericDeviceWeightPreparation,
) -> Unit raise NumericWeightFileError {
  if !self.open {
    return
  }
  self.allocation.close() catch {
    error => {
      self.cleanup_failure = error
      raise CleanupFailed(stage=self.stage, error)
    }
  }
  self.open = false
}
