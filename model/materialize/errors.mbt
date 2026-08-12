///|
/// Configurable limit owned by the materialization boundary.
pub(all) enum MaterializationLimit {
  HeaderBytes
  TensorCount
  PayloadBytes
  HostBytes
} derive(Debug, Eq)

///|
/// Bounded failures emitted before or during weight materialization.
///
/// Tensor references are bounded by `MaterializationLimits::max_tensor_count`.
/// No variant retains model bytes, tensor names, file paths, digests, offsets,
/// or callback error text. Errors raised by a visitor are propagated unchanged.
pub(all) suberror MaterializationError {
  InvalidLimit(MaterializationLimit)
  MissingHeaderLength
  HeaderLengthOverflow
  HeaderTooLarge
  TruncatedHeader
  PayloadTooLarge
  EnvelopeLengthMismatch
  MaterializedSizeOverflow
  TooManyTensors
  MissingBinding(reference~ : Int)
  BindingOrderMismatch(reference~ : Int)
  InvalidTensorRange(reference~ : Int)
  TensorRangeOutsidePayload(reference~ : Int)
  TensorRangeOverflow(reference~ : Int)
  MaterializedSizeMismatch
  HostMaterializationTooLarge
} derive(Debug, Eq)
