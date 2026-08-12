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
/// Bounded, payload-safe safetensors metadata failures.
///
/// Tensor positions are bounded by `HeaderLimits::max_tensor_count`; raw tensor
/// names, file paths, header fragments, and numeric attacker input are omitted.
pub(all) suberror SafetensorsError {
  InvalidLimit(MetadataLimit)
  HeaderTooLarge
  TooManyTensors
  InvalidTensorName(Int)
  InvalidDType(Int)
  InvalidRank(Int)
  InvalidDimension(Int)
  TensorSizeOverflow(Int)
  InvalidByteRange(Int)
  RangeOutsidePayload(Int)
  RangeSizeMismatch(Int)
  DuplicateTensor(Int, Int)
  OverlappingTensors(Int, Int)
  MaterializedSizeExceeded
} derive(Debug, Eq)
