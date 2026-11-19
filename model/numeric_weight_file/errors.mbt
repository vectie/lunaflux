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
  TensorCountMismatch(expected~ : Int32, actual~ : Int32)
  MissingTensor(ordinal~ : Int32)
  InvalidTensor(ordinal~ : Int32)
  UnexpectedTensorField(ordinal~ : Int32)
  InvalidTensorDType(ordinal~ : Int32)
  InvalidTensorShape(ordinal~ : Int32, axis~ : Int32)
  InvalidTensorRange(ordinal~ : Int32)
  TensorSizeOverflow(ordinal~ : Int32)
  PayloadNotContiguous(ordinal~ : Int32)
  TrailingPayload
  NonFiniteFp8Code(ordinal~ : Int32, element~ : Int32)
  ReservedI8Code(ordinal~ : Int32, row~ : Int32)
  InvalidScale(ordinal~ : Int32, row~ : Int32)
  ZeroRowScaleMismatch(ordinal~ : Int32, row~ : Int32)
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
