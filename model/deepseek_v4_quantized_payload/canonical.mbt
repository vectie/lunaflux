///|
priv struct PayloadCanonicalWriter {
  hash : @canonical_sha256.CanonicalSha256
  mut bytes : Int
  maximum : Int
}

///|
fn PayloadCanonicalWriter::new(maximum : Int) -> Self {
  { hash: @canonical_sha256.CanonicalSha256::new(), bytes: 0, maximum, }
}

///|
fn PayloadCanonicalWriter::reserve(
  self : Self,
  count : Int,
) -> Unit raise DeepSeekV4QuantizedPayloadError {
  if count < 0 || self.bytes > self.maximum - count {
    raise CanonicalEncodingTooLarge
  }
  self.bytes += count
}

///|
fn PayloadCanonicalWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise DeepSeekV4QuantizedPayloadError {
  self.reserve(1)
  ignore(self.hash.update_byte(value))
}

///|
fn PayloadCanonicalWriter::int(
  self : Self,
  value : Int,
) -> Unit raise DeepSeekV4QuantizedPayloadError {
  self.reserve(8)
  ignore(
    self.hash.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()),
  )
}

///|
fn PayloadCanonicalWriter::int64(
  self : Self,
  value : Int64,
) -> Unit raise DeepSeekV4QuantizedPayloadError {
  self.reserve(8)
  ignore(self.hash.update_u64_le(value.reinterpret_as_uint64()))
}

///|
fn PayloadCanonicalWriter::digest_text(
  self : Self,
  value : String,
) -> Unit raise DeepSeekV4QuantizedPayloadError {
  self.int(value.length())
  self.reserve(value.length())
  for character in value {
    self.byte(character.to_int().to_byte())
  }
}

///|
fn encoding_tag(encoding : DeepSeekV4QuantizedPayloadEncoding) -> Byte {
  match encoding {
    Fp8E4M3Block128Parameter => 1
    Ue8M0Block128ScaleMetadata => 2
    PackedFp4E2M1Block32Parameter => 3
    Ue8M0Block32ScaleMetadata => 4
  }
}

///|
fn canonical_contract_digest(
  encoding : DeepSeekV4QuantizedPayloadEncoding,
  block_shape : DeepSeekV4QuantizedBlockShape,
  logical_shape : ArrayView[Int64],
  model_content_digest : @spec.ContentDigest,
  fp4_codebook : @deepseek_v4_numeric_plan.DeepSeekV4Fp4CodebookBinding?,
  payload_bytes : Int64,
  maximum : Int,
) -> DeepSeekV4QuantizedPayloadContractDigest raise DeepSeekV4QuantizedPayloadError {
  let writer = PayloadCanonicalWriter::new(maximum)
  writer.digest_text("lunaflux/deepseek-v4-quantized-layout/v2")
  writer.byte(encoding_tag(encoding))
  writer.int(block_shape.rows())
  writer.int(block_shape.columns())
  writer.int(logical_shape.length())
  for dimension in logical_shape {
    writer.int64(dimension)
  }
  writer.digest_text(model_content_digest.as_hex())
  writer.byte(if fp4_codebook is Some(_) { 1 } else { 0 })
  writer.int64(payload_bytes)
  { hex: writer.hash.finalize_hex(), }
}
