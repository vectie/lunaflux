///|
/// Stable reasons a validated weight set cannot be placed in one device arena.
///
/// Errors retain no model payload, tensor name, path, device pointer, CUDA
/// result number, or vendor diagnostic. Nested materialization and device
/// errors are themselves bounded public vocabularies.
pub(all) suberror DeviceWeightError {
  InvalidAlignment
  InvalidArenaLimit
  EmptyBindings
  MissingBinding(reference~ : Int)
  BindingOrderMismatch(reference~ : Int)
  LayoutSizeOverflow(reference~ : Int)
  ArenaTooLarge
  UnexpectedTensor(reference~ : Int)
  TensorLengthMismatch(reference~ : Int)
  InvalidFileLimit(DeviceWeightFileLimit)
  FileUnavailable
  FileCloseFailed
  UnsupportedFileType
  FileTooLarge
  FileTruncated
  FileChangedDuringLoad
  ContentDigestMismatch
  InvalidSafetensors(@safetensors_reader.SafetensorsReaderError)
  InvalidLlamaWeights(@llama_weights.LlamaWeightsError)
  MaterializationFailed(@materialize.MaterializationError)
  DeviceFailed(@device.DeviceError)
  CleanupFailed(stage~ : DeviceWeightFailureStage, @device.DeviceError)
} derive(Debug, Eq)

///|
/// Bounded stage retained when cleanup failure supersedes the primary error.
/// The native allocation finalizer remains a defensive retry path, but callers
/// must treat this result as a failed load and an unhealthy resource boundary.
pub(all) enum DeviceWeightFailureStage {
  TensorValidation
  Transfer
  SourceRevalidation
  SourceClose
  MaterializationResult
} derive(Debug, Eq)

///|
/// Bounded configuration fields owned by the streaming file loader.
pub(all) enum DeviceWeightFileLimit {
  FileBytes
  TransferChunkBytes
} derive(Debug, Eq)
