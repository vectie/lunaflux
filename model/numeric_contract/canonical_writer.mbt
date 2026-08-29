///|
priv struct NumericCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256
  expected : Int
  mut count : Int
}

///|
fn NumericCanonicalWriter::new_hashing(
  expected : Int,
) -> NumericCanonicalWriter {
  { hasher: @canonical_sha256.CanonicalSha256::new(), expected, count: 0 }
}

///|
priv struct NumericCanonicalMaterializationCursor {
  expected : Int
  mut next_index : Int
}

///|
fn NumericCanonicalMaterializationCursor::new(
  expected : Int,
) -> NumericCanonicalMaterializationCursor {
  { expected, next_index: 0 }
}

///|
fn NumericCanonicalMaterializationCursor::next(
  self : NumericCanonicalMaterializationCursor,
  index : Int,
  byte_at : (Int) -> Byte,
) -> Byte {
  if index != self.next_index || index < 0 || index >= self.expected {
    abort("non-sequential numeric canonical materialization")
  }
  let byte = byte_at(index)
  self.next_index += 1
  byte
}

///|
fn NumericCanonicalMaterializationCursor::check_complete(
  self : NumericCanonicalMaterializationCursor,
) -> Unit {
  if self.next_index != self.expected {
    abort("incomplete numeric canonical materialization")
  }
}

///|
fn materialize_numeric_canonical(
  expected : Int,
  byte_at : (Int) -> Byte,
) -> Bytes {
  if expected < 0 {
    abort("negative numeric canonical materialization length")
  }
  let cursor = NumericCanonicalMaterializationCursor::new(expected)
  let output = Bytes::makei(expected, index => cursor.next(index, byte_at))
  cursor.check_complete()
  output
}

///|
fn numeric_canonical_u64_byte(value : UInt64, offset : Int) -> Byte {
  if offset < 0 || offset >= 8 {
    abort("numeric canonical integer byte offset out of bounds")
  }
  (value >> (offset * 8)).to_byte()
}

///|
fn numeric_canonical_int_byte(value : Int, offset : Int) -> Byte {
  numeric_canonical_u64_byte(
    Int64::from_int(value).reinterpret_as_uint64(),
    offset,
  )
}

///|
fn numeric_canonical_int64_byte(value : Int64, offset : Int) -> Byte {
  numeric_canonical_u64_byte(value.reinterpret_as_uint64(), offset)
}

///|
fn NumericCanonicalWriter::reserve(
  self : NumericCanonicalWriter,
  amount : Int,
) -> Unit raise NumericContractError {
  if amount < 0 || self.count > self.expected - amount {
    raise InvalidNumericContract(CanonicalBytes)
  }
}

///|
fn NumericCanonicalWriter::write_byte(
  self : NumericCanonicalWriter,
  byte : Byte,
) -> Unit raise NumericContractError {
  self.reserve(1)
  if !self.hasher.update_byte(byte) {
    abort("finalized canonical hash")
  }
  self.count += 1
}

///|
fn NumericCanonicalWriter::write_bytes(
  self : NumericCanonicalWriter,
  bytes : BytesView,
) -> Unit raise NumericContractError {
  self.reserve(bytes.length())
  if !self.hasher.update_bytes(bytes) {
    abort("finalized canonical hash")
  }
  self.count += bytes.length()
}

///|
fn NumericCanonicalWriter::write_u64(
  self : NumericCanonicalWriter,
  value : UInt64,
) -> Unit raise NumericContractError {
  self.reserve(8)
  if !self.hasher.update_u64_le(value) {
    abort("finalized canonical hash")
  }
  self.count += 8
}

///|
fn NumericCanonicalWriter::write_int(
  self : NumericCanonicalWriter,
  value : Int,
) -> Unit raise NumericContractError {
  self.write_u64(Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn NumericCanonicalWriter::write_int64(
  self : NumericCanonicalWriter,
  value : Int64,
) -> Unit raise NumericContractError {
  self.write_u64(value.reinterpret_as_uint64())
}

///|
fn NumericCanonicalWriter::check_complete(
  self : NumericCanonicalWriter,
) -> Unit raise NumericContractError {
  if self.count != self.expected {
    raise InvalidNumericContract(CanonicalBytes)
  }
}

///|
fn NumericCanonicalWriter::finish_hex(
  self : NumericCanonicalWriter,
) -> String raise NumericContractError {
  self.check_complete()
  self.hasher.finalize_hex()
}
