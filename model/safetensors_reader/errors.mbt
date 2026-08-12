///|
/// Tensor fields decoded by the bounded safetensors reader.
pub(all) enum TensorField {
  DType
  Shape
  DataOffsets
} derive(Debug, Eq)

///|
/// Bounded, payload-safe failures from a safetensors byte reader.
///
/// These errors never retain the encoded header, tensor names, paths, or
/// attacker-controlled numeric text. Tensor positions are bounded by the
/// supplied `HeaderLimits`.
pub(all) suberror SafetensorsReaderError {
  MissingHeaderLength
  HeaderLengthOverflow
  HeaderTooLarge
  TruncatedHeader
  InvalidHeaderEnvelope
  InvalidHeaderEncoding
  InvalidHeaderJson
  DuplicateHeaderKey(depth~ : Int, index~ : Int)
  InvalidRoot
  InvalidMetadata
  InvalidTensor(Int)
  MissingTensorField(Int, TensorField)
  UnexpectedTensorField(Int)
  InvalidTensorField(Int, TensorField)
  InvalidInteger(Int, TensorField)
  PayloadSizeMismatch
  Validation(@safetensors.SafetensorsError)
} derive(Debug, Eq)
