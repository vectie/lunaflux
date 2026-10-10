///|
test "checkpoint digest block feed preserves exact partial-tail boundaries" {
  for chunk in [1, 3, 63, 64, 65, 4096] {
    let scratch = FixedArray::makei(chunk, i => (i % 251).to_byte())
    let expected = @crypto.SHA256::new()
    let actual = @crypto.SHA256::new()
    for count in [chunk, chunk, 1, chunk.max(2) / 2] {
      let count = count.min(chunk)
      update_file_digest(actual, scratch, count)
      expected.update_from_iter(scratch[:count].iter())
    }
    assert_eq(
      @crypto.bytes_to_hex_string(actual.finalize()),
      @crypto.bytes_to_hex_string(expected.finalize()),
    )
  }
  let actual = @crypto.SHA256::new()
  update_file_digest(actual, FixedArray::from_array(b"abcX"[:].to_array()), 3)
  assert_eq(
    @crypto.bytes_to_hex_string(actual.finalize()),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  )
}

///|
test "checkpoint digest rejects empty or out-of-bounds chunks" {
  for count in [-1, 0, 5] {
    try
      update_file_digest(@crypto.SHA256::new(), FixedArray::make(4, 0), count)
    catch {
      InvalidLimit => ()
      _ => fail("unexpected invalid digest chunk error")
    } noraise {
      _ => fail("invalid digest chunk accepted")
    }
  }
}
