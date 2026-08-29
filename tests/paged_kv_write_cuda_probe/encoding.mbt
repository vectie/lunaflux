///|
fn kv_i32_bytes(values : ReadOnlyArray[Int]) -> Bytes {
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
fn kv_bf16_bytes(values : ArrayView[Int]) -> Bytes {
  let output = FixedArray::make(values.length() * 2, b'\x00')
  for index, value in values {
    output[index * 2] = value.to_byte()
    output[index * 2 + 1] = (value >> 8).to_byte()
  }
  Bytes::from_array(output[:])
}

///|
fn kv_bf16_bits(bytes : Bytes) -> Array[Int] raise {
  require_kv_probe(bytes.length() % 2 == 0, "truncated BF16 readback")
  Array::makei(bytes.length() / 2, index => {
    bytes[index * 2].to_int() | (bytes[index * 2 + 1].to_int() << 8)
  })
}

///|
fn kv_u32(bytes : Bytes) -> UInt raise {
  require_kv_probe(bytes.length() == 4, "truncated canary readback")
  (bytes[0].to_int() |
  (bytes[1].to_int() << 8) |
  (bytes[2].to_int() << 16) |
  (bytes[3].to_int() << 24)).reinterpret_as_uint()
}
