///|
/// Stable reasons authenticated weights cannot be placed in one device arena.
///
/// Errors retain no model payload, tensor name, path, device pointer, CUDA
/// result number, or vendor diagnostic. Nested file-schema and device errors
/// are themselves bounded public vocabularies.
pub(all) suberror DeviceWeightError {
  InvalidAlignment
  InvalidArenaLimit
  EmptyBindings
  MissingBinding(reference~ : Int)
  BindingOrderMismatch(reference~ : Int)
  LayoutSizeOverflow(reference~ : Int)
  ArenaTooLarge
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
  InvalidTensorParallelPlan(@llama_tensor_parallel.LlamaTensorParallelError)
  TensorParallelPlanMismatch
  InvalidShardedTransfer(reference~ : Int, segment~ : Int)
  DeviceFailed(@device.DeviceError)
  CleanupFailed(stage~ : DeviceWeightFailureStage, @device.DeviceError)
} derive(Debug, Eq)

///|
/// Bounded stage retained when cleanup failure supersedes the primary error.
/// The native allocation finalizer remains a defensive retry path, but callers
/// must treat this result as a failed load and an unhealthy resource boundary.
pub(all) enum DeviceWeightFailureStage {
  Transfer
  SourceRevalidation
  SourceClose
} derive(Debug, Eq)

///|
/// Bounded configuration fields owned by the streaming file loader.
pub(all) enum DeviceWeightFileLimit {
  FileBytes
  TransferChunkBytes
} derive(Debug, Eq)
