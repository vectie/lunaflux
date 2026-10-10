///|
/// Exact quantized payload layouts admitted by the DeepSeek V4 schema.
pub(all) enum DeepSeekV4QuantizedPayloadEncoding {
  Fp8E4M3Block128Parameter
  Ue8M0Block128ScaleMetadata
  PackedFp4E2M1Block32Parameter
  Ue8M0Block32ScaleMetadata
} derive(Debug, Eq)

///|
/// Evidence gaps that this package deliberately does not guess through.
pub(all) enum DeepSeekV4QuantizedInterpretationUnavailable {
  Ue8M0ExponentBiasAndSpecialCodesUnproven
  Fp4NibbleAndCodebookOrderingUnproven
  Fp4OddColumnPaddingUnproven
  Ue8M0Block32PartialPaddingUnproven
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4ScalarInterpretationAvailability {
  FiniteFp8E4M3Available
  Unavailable(DeepSeekV4QuantizedInterpretationUnavailable)
} derive(Debug, Eq)

///|
pub(all) suberror DeepSeekV4QuantizedPayloadError {
  InvalidLimit
  UnsupportedLogicalStorage(@deepseek_v4_numeric_plan.DeepSeekV4LogicalStorage)
  InvalidLogicalShape
  InvalidBlockShape
  LayoutUnavailable(DeepSeekV4QuantizedInterpretationUnavailable)
  ElementCountExceeded
  PayloadByteCountExceeded
  PayloadByteLengthMismatch(expected~ : Int64, actual~ : Int64)
  NonFiniteFp8PayloadCode(element~ : Int64, code~ : Byte)
  ScalarIndexOutOfBounds(Int64)
  NonFiniteFp8Code(Byte)
  DecodeUnavailable(DeepSeekV4QuantizedInterpretationUnavailable)
  ScalePairMismatch
  CanonicalEncodingTooLarge
} derive(Debug, Eq)

///|
/// Startup-only absolute bounds for one host payload contract.
pub struct DeepSeekV4QuantizedPayloadLimits {
  priv maximum_elements : Int64
  priv maximum_payload_bytes : Int64
  priv maximum_canonical_bytes : Int
} derive(Debug, Eq)

///|
/// Exact scaling block geometry attached to a payload encoding.
pub struct DeepSeekV4QuantizedBlockShape {
  priv rows : Int
  priv columns : Int
} derive(Debug, Eq)

///|
/// Fingerprint of storage geometry and declared model identity, not payload bytes.
pub struct DeepSeekV4QuantizedPayloadContractDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
pub struct DeepSeekV4QuantizedPayloadContract {
  priv schema_version : Int
  priv encoding : DeepSeekV4QuantizedPayloadEncoding
  priv block_shape : DeepSeekV4QuantizedBlockShape
  priv logical_shape : ReadOnlyArray[Int64]
  priv logical_element_count : Int64
  priv payload_bytes : Bytes
  priv model_content_digest : @spec.ContentDigest
  priv fp4_codebook : @deepseek_v4_numeric_plan.DeepSeekV4Fp4CodebookBinding?
  priv contract_digest : DeepSeekV4QuantizedPayloadContractDigest
}
