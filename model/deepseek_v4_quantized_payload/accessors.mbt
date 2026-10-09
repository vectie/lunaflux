///|
pub fn DeepSeekV4QuantizedPayloadContract::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::encoding(
  self : Self,
) -> DeepSeekV4QuantizedPayloadEncoding {
  self.encoding
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::block_shape(
  self : Self,
) -> DeepSeekV4QuantizedBlockShape {
  self.block_shape
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::logical_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.logical_shape
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::logical_element_count(
  self : Self,
) -> Int64 {
  self.logical_element_count
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::payload_bytes(self : Self) -> Bytes {
  self.payload_bytes
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::payload_digest(
  self : Self,
) -> DeepSeekV4QuantizedPayloadDigest {
  self.payload_digest
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::model_content_digest(
  self : Self,
) -> @spec.ContentDigest {
  self.model_content_digest
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::fp4_codebook(
  self : Self,
) -> @deepseek_v4_numeric_plan.DeepSeekV4Fp4CodebookBinding? {
  self.fp4_codebook
}

///|
pub fn DeepSeekV4QuantizedPayloadContract::contract_digest(
  self : Self,
) -> DeepSeekV4QuantizedPayloadContractDigest {
  self.contract_digest
}
