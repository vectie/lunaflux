///|
/// Limit whose configured value cannot define a safe metadata envelope.
pub(all) enum MetadataLimit {
  HeaderBytes
  TensorCount
  TensorRank
  TensorNameLength
  PayloadBytes
  MaterializedTensorBytes
} derive(Debug, Eq)

///|
pub extend MetadataLimit with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend MetadataLimit with Eq::{not_equal, equal}

///|
/// Bounded, payload-safe safetensors metadata failures.
///
/// Tensor positions are bounded by `HeaderLimits::max_tensor_count`; raw tensor
/// names, file paths, header fragments, and numeric attacker input are omitted.
pub(all) suberror SafetensorsError {
  InvalidLimit(MetadataLimit)
  HeaderTooLarge
  TooManyTensors
  InvalidTensorName(Int32)
  InvalidDType(Int32)
  InvalidRank(Int32)
  InvalidDimension(Int32)
  TensorSizeOverflow(Int32)
  InvalidByteRange(Int32)
  RangeOutsidePayload(Int32)
  RangeSizeMismatch(Int32)
  DuplicateTensor(Int32, Int32)
  OverlappingTensors(Int32, Int32)
  MaterializedSizeExceeded
} derive(Debug, Eq)

///|
pub extend SafetensorsError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SafetensorsError with Eq::{not_equal, equal}
