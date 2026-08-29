///|
/// Stable, payload-free reasons numeric weight admission or ownership failed.
pub(all) suberror NumericWeightFileError {
  InvalidLimit(NumericWeightFileLimit)
  InvalidArtifactDigest
  Closed
  FileUnavailable
  UnsupportedFileType
  FileCloseFailed
  FileTooLarge
  FileTruncated
  FileChanged
  ArtifactDigestMismatch
  InvalidSafetensorsPrefix(@safetensors_reader.SafetensorsReaderError)
  InvalidHeaderEncoding
  InvalidHeaderJson
  DuplicateHeaderKey
  InvalidHeaderEnvelope
  MissingDocumentMetadata
  UnexpectedDocumentMetadata
  FormatMismatch
  ModelContentMismatch
  ModelPlanMismatch
  NumericSchemaMismatch
  TensorCountMismatch(expected~ : Int, actual~ : Int)
  MissingTensor(ordinal~ : Int)
  InvalidTensor(ordinal~ : Int)
  UnexpectedTensorField(ordinal~ : Int)
  InvalidTensorDType(ordinal~ : Int)
  InvalidTensorShape(ordinal~ : Int, axis~ : Int)
  InvalidTensorRange(ordinal~ : Int)
  TensorSizeOverflow(ordinal~ : Int)
  PayloadNotContiguous(ordinal~ : Int)
  TrailingPayload
  NonFiniteFp8Code(ordinal~ : Int, element~ : Int)
  ReservedI8Code(ordinal~ : Int, row~ : Int)
  InvalidScale(ordinal~ : Int, row~ : Int)
  ZeroRowScaleMismatch(ordinal~ : Int, row~ : Int)
  InvalidLayout(@device_materialize.DeviceWeightError)
  AuthorityMismatch(NumericWeightAuthorityRule)
  DeviceFailed(@device.DeviceError)
  CleanupFailed(stage~ : NumericWeightFailureStage, @device.DeviceError)
} derive(Debug, Eq)

///|
pub(all) enum NumericWeightAuthorityRule {
  Identity
  NumericSchema
  Layout
  Artifact
} derive(Debug, Eq)

///|
pub(all) enum NumericWeightFileLimit {
  FileBytes
  HeaderBytes
  TransferChunkBytes
  TensorCount
} derive(Debug, Eq)

///|
pub(all) enum NumericWeightFailureStage {
  Transfer
  SourceRevalidation
  SourceClose
} derive(Debug, Eq)
