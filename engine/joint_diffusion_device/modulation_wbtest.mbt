///|
test "modulation indices validate every row before upload and preserve Int32 bytes" {
  let bytes = modulation_index_bytes([0, 2, 257], 258)
  assert_eq(bytes.length(), 12)
  assert_eq(bytes[4], 2)
  assert_eq(bytes[8], 1)
  assert_eq(bytes[9], 1)
  for (indices, count) in [([], 3), ([0], 0), ([-1], 3), ([0, 3], 3)] {
    try modulation_index_bytes(indices, count) catch {
      @device.DeviceError::InvalidArgument(CopyToDevice) => ()
      _ => fail("wrong index error")
    } noraise {
      _ => fail("invalid modulation row admitted")
    }
  }
}
