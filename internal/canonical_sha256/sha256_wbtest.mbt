///|
fn digest_chunks(chunks : ArrayView[Bytes]) -> String raise {
  let hasher = CanonicalSha256::new()
  for chunk in chunks {
    assert_true(hasher.update_bytes(chunk))
  }
  hasher.finalize_hex()
}

///|
test "canonical SHA-256 matches NIST vectors and owns exactly fixed scratch" {
  inspect(
    digest_chunks([]),
    content="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  )
  inspect(
    digest_chunks([b"abc"]),
    content="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  )
  let hasher = CanonicalSha256::new()
  assert_eq(hasher.block.length(), 64)
  assert_eq(hasher.schedule.length(), 64)
  assert_eq(hasher.transformed_blocks, 0)
}

///|
test "canonical SHA-256 block boundaries and chunking match oracle" {
  for length in [55, 56, 63, 64, 65, 119, 120, 127, 128, 129] {
    let bytes = Bytes::make(length, 0x61)
    let expected = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
    assert_eq(digest_chunks([bytes]), expected)
    let chunks : Array[Bytes] = []
    for index in 0..<length {
      chunks.push(Bytes::make(1, bytes[index]))
    }
    assert_eq(digest_chunks(chunks), expected)
  }
}

///|
test "canonical SHA-256 one MiB differential and construction counters" {
  let bytes = Bytes::make(1024 * 1024, 0x61)
  let expected = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  let hasher = CanonicalSha256::new()
  assert_true(hasher.update_bytes(bytes))
  assert_eq(hasher.block.length(), 64)
  assert_eq(hasher.schedule.length(), 64)
  assert_eq(hasher.transformed_blocks, 16384)
  assert_eq(hasher.finalize_hex(), expected)
  assert_eq(hasher.transformed_blocks, 16385)
  assert_eq(hasher.finalize_hex(), expected)
  assert_eq(hasher.transformed_blocks, 16385)
  assert_false(hasher.update_byte(0x00))
  assert_false(hasher.update_bytes(b"later"))
  assert_false(hasher.update_u64_le(1UL))
}
