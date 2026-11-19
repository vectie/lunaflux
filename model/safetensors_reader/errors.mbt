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
  HeaderEnvelopeSizeMismatch
  InvalidPayloadLength
  InvalidHeaderEnvelope
  InvalidHeaderEncoding
  InvalidHeaderJson
  DuplicateHeaderKey(depth~ : Int32, index~ : Int32)
  InvalidRoot
  InvalidMetadata
  InvalidTensor(Int32)
  MissingTensorField(Int32, TensorField)
  UnexpectedTensorField(Int32)
  InvalidTensorField(Int32, TensorField)
  InvalidInteger(Int32, TensorField)
  PayloadSizeMismatch
  Validation(@safetensors.SafetensorsError)
} derive(Debug, Eq)
