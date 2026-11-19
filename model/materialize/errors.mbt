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
  MissingBinding(reference~ : Int32)
  BindingOrderMismatch(reference~ : Int32)
  InvalidTensorRange(reference~ : Int32)
  TensorRangeOutsidePayload(reference~ : Int32)
  TensorRangeOverflow(reference~ : Int32)
  MaterializedSizeMismatch
  HostMaterializationTooLarge
} derive(Debug, Eq)
