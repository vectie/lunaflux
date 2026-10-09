///|
test "little endian UInt64 readers preserve every bit at unaligned offsets" {
  let bytes : FixedArray[Byte] = FixedArray::make(18, b'\x5a')
  for offset in [0, 1, 9, 10] {
    for
      value in [
        0UL, 1UL, 0x0123456789abcdefUL, 0x8000000000000000UL, 0xffffffffffffffffUL,
      ] {
      write_u64(bytes, offset, value)
      assert_eq(read_u64(bytes, offset), value)
    }
  }
  let distinct : FixedArray[Byte] = [
    0xff, 0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x80,
  ]
  assert_eq(read_u64(distinct, 0), 0x80100804020100ffUL)
}
