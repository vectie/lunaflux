///|
pub struct EncoderTables {
  cosine : Bytes
  sine : Bytes
}

///|
fn bf16(value : Float) -> UInt {
  let bits = value.reinterpret_as_uint()
  (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16
}

///|
fn store16(bytes : FixedArray[Byte], offset : Int, value : UInt) -> Unit {
  bytes[offset] = (value & 255U).to_byte()
  bytes[offset + 1] = ((value >> 8) & 255U).to_byte()
}

///|
/// Exact axis selection and BF16-RNE storage; host transcendental rounding
/// still needs the same physical numerical qualification as other host tables.
pub fn EncoderConfig::rotary_tables(
  self : Self,
  positions : ArrayView[(Int, Int, Int)],
  max_bytes~ : Int64,
) -> EncoderTables raise EncoderConfigError {
  let bytes = positions.length().to_int64() * 128L * 2L
  guard positions.length() > 0 &&
    positions.length() <= 262144 &&
    bytes <= 2147483647L &&
    max_bytes > 0L &&
    bytes <= max_bytes / 2L else {
    raise ByteBudgetExceeded
  }
  let cosine : FixedArray[Byte] = FixedArray::make(bytes.to_int(), 0)
  let sine : FixedArray[Byte] = FixedArray::make(bytes.to_int(), 0)
  let frequencies = Array::make(64, 0.0F)
  for d in 0..<64 {
    let power = Float::from_double(self.theta.pow((2 * d).to_double() / 128.0))
    frequencies[d] = 1.0F / power
  }
  for row, (t, h, w) in positions {
    guard t >= 0 &&
      h >= 0 &&
      w >= 0 &&
      t < self.max_positions &&
      h < self.max_positions &&
      w < self.max_positions else {
      raise InvalidRequest
    }
    for d in 0..<64 {
      let coordinate = if d % 3 == 1 && d < self.mrope_height * 3 {
        h
      } else if d % 3 == 2 && d < self.mrope_width * 3 {
        w
      } else {
        t
      }
      let angle = Float::from_int(coordinate) * frequencies[d]
      let c = bf16(Float::from_double(@math.cos(angle.to_double())))
      let s = bf16(Float::from_double(@math.sin(angle.to_double())))
      for half in 0..<2 {
        let offset = (row * 128 + d + half * 64) * 2
        store16(cosine, offset, c)
        store16(sine, offset, s)
      }
    }
  }
  { cosine: Bytes::from_array(cosine[:]), sine: Bytes::from_array(sine[:]), }
}

///|
pub fn text_positions(
  rows : Int,
) -> ReadOnlyArray[(Int, Int, Int)] raise EncoderConfigError {
  guard rows > 0 && rows <= 262144 else { raise InvalidRequest }
  ReadOnlyArray::from_array(Array::makei(rows, i => (i, i, i)))
}

///|
pub fn encode_i32(values : ArrayView[Int]) -> Bytes {
  let bytes : Array[Byte] = []
  for value in values {
    for byte in value.to_le_bytes() {
      bytes.push(byte)
    }
  }
  Bytes::from_array(bytes)
}
