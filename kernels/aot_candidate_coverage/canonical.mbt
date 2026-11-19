///|
priv struct CoverageWriter {
  hash : @canonical_sha256.CanonicalSha256
  mut bytes : Int32
  maximum_bytes : Int32
}

///|
fn CoverageWriter::new(maximum_bytes : Int32) -> Self {
  { hash: @canonical_sha256.CanonicalSha256::new(), bytes: 0, maximum_bytes, }
}

///|
fn CoverageWriter::reserve(
  self : Self,
  count : Int32,
) -> Unit raise AotCandidateCoverageError {
  if count < 0 || self.bytes > self.maximum_bytes - count {
    raise CoverageEncodingTooLarge
  }
  self.bytes += count
}

///|
fn CoverageWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise AotCandidateCoverageError {
  self.reserve(1)
  if !self.hash.update_byte(value) {
    abort("finalized AOT candidate coverage hash")
  }
}

///|
fn CoverageWriter::int(
  self : Self,
  value : Int32,
) -> Unit raise AotCandidateCoverageError {
  self.reserve(8)
  if !self.hash.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()) {
    abort("finalized AOT candidate coverage hash")
  }
}

///|
fn CoverageWriter::text(
  self : Self,
  value : String,
) -> Unit raise AotCandidateCoverageError {
  let encoded = @utf8.encode(value)
  self.int(encoded.length())
  self.reserve(encoded.length())
  if !self.hash.update_bytes(encoded) {
    abort("finalized AOT candidate coverage hash")
  }
}

///|
fn coverage_digest(
  source_identity : @spec.ModelIdentity,
  requirements_digest_sha256 : String,
  requirement_count : Int32,
  ordered : FixedArray[AotCandidateEvidenceInput?],
  maximum_encoding_bytes : Int32,
) -> AotCandidateCoverageDigest raise AotCandidateCoverageError {
  let writer = CoverageWriter::new(maximum_encoding_bytes)
  writer.text("lunaflux/aot-candidate-coverage/v1")
  writer.text(source_identity.content().as_hex())
  writer.text(source_identity.plan().as_hex())
  writer.text(requirements_digest_sha256)
  writer.int(requirement_count)
  for ordinal in 0..<requirement_count {
    writer.int(ordinal)
    match ordered[ordinal] {
      None => writer.byte(0)
      Some(candidate) => {
        writer.byte(1)
        writer.text(candidate.source_digest_sha256)
        writer.text(candidate.recipe_digest_sha256)
        writer.byte(if candidate.manifest_bindable { 1 } else { 0 })
      }
    }
  }
  { hex: writer.hash.finalize_hex(), }
}
