///|
fn probe_i32_bytes(values : ReadOnlyArray[Int]) -> Bytes {
  let output = FixedArray::make(values.length() * 4, b'\x00')
  for index, value in values {
    output[index * 4] = value.to_byte()
    output[index * 4 + 1] = (value >> 8).to_byte()
    output[index * 4 + 2] = (value >> 16).to_byte()
    output[index * 4 + 3] = (value >> 24).to_byte()
  }
  Bytes::from_array(output[:])
}

///|
fn probe_bf16_bits(bytes : Bytes) -> Array[Int] raise {
  require_fused_probe(bytes.length() % 2 == 0, "BF16 readback is truncated")
  Array::makei(bytes.length() / 2, index => {
    bytes[index * 2].to_int() | (bytes[index * 2 + 1].to_int() << 8)
  })
}

///|
fn probe_u32(bytes : Bytes) -> UInt raise {
  require_fused_probe(bytes.length() == 4, "canary readback is truncated")
  (bytes[0].to_int() |
  (bytes[1].to_int() << 8) |
  (bytes[2].to_int() << 16) |
  (bytes[3].to_int() << 24)).reinterpret_as_uint()
}

///|
fn probe_zero_u32() -> Bytes {
  b"\x00\x00\x00\x00"
}
