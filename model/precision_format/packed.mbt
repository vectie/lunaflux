///|
/// Owns immutable, canonical logical payload and scale bytes. Authentication
/// of a model/checkpoint remains the loader's responsibility, not this codec's.
pub struct PackedTensor {
  priv representation : Representation
  priv payload : Bytes
  priv scales : Bytes
  priv global_scale : Float
}

///|
fn read_code(bytes : Bytes, offset : Int, width : Int) -> UInt {
  for i in 0..<width; code = 0U {
    continue code | (bytes[offset + i].to_uint() << (8 * i))
  } nobreak {
    code
  }
}

///|
fn write_code(
  bytes : FixedArray[Byte],
  offset : Int,
  width : Int,
  code : UInt,
) -> Unit {
  for i in 0..<width {
    bytes[offset + i] = (code >> (8 * i)).to_byte()
  }
}

///|
fn PackedTensor::payload_code(
  self : PackedTensor,
  row : Int,
  column : Int,
) -> UInt {
  let offset = row.to_int64() * self.representation.row_bytes
  let bits = self.representation.element.bits()
  if bits == 4 {
    (
      self.payload[(offset + (column / 2).to_int64()).to_int()].to_uint() >>
      (column % 2 * 4)
    ) &
    15U
  } else {
    read_code(
      self.payload,
      (offset + column.to_int64() * (bits / 8).to_int64()).to_int(),
      bits / 8,
    )
  }
}

///|
pub fn PackedTensor::new(
  representation : Representation,
  payload~ : Bytes,
  scales~ : Bytes,
  global_scale? : Float = 1.0F,
) -> PackedTensor raise PrecisionError {
  if representation.payload_bytes != payload.length().to_int64() ||
    representation.scale_bytes != scales.length().to_int64() {
    raise PayloadMismatch
  }
  if global_scale.is_nan() ||
    global_scale.is_inf() ||
    global_scale <= 0.0F ||
    (!representation.global_scale && global_scale != 1.0F) {
    raise InvalidScale
  }
  let result = { representation, payload, scales, global_scale, }
  let scale_width = representation.scale_element.bytes()
  if scale_width > 0 {
    for offset = 0; offset < scales.length(); offset = offset + scale_width {
      ignore(
        decode_scale(
          representation.scale_element,
          read_code(scales, offset, scale_width),
        ),
      )
    }
  }
  for row in 0..<representation.rows {
    if representation.element.bits() == 4 && representation.columns % 2 != 0 {
      let tail = ((row + 1).to_int64() * representation.row_bytes - 1L).to_int()
      if (payload[tail].to_uint() & 0xf0U) != 0U {
        raise InvalidPadding
      }
    }
    for column in 0..<representation.columns {
      // Validate reconstruction too: finite codes/scales can overflow F32.
      ignore(result.decode(row, column))
    }
  }
  result
}

///|
pub fn PackedTensor::representation(self : PackedTensor) -> Representation {
  self.representation
}

///|
pub fn PackedTensor::payload(self : PackedTensor) -> Bytes {
  self.payload
}

///|
pub fn PackedTensor::scales(self : PackedTensor) -> Bytes {
  self.scales
}

///|
pub fn PackedTensor::global_scale(self : PackedTensor) -> Float {
  self.global_scale
}

///|
/// F32 reconstruction with two ordered multiplies for NVFP4. Zero points,
/// codebooks and storage swizzles cannot be silently inferred from filenames.
pub fn PackedTensor::decode(
  self : PackedTensor,
  row : Int,
  column : Int,
) -> Float raise PrecisionError {
  let rep = self.representation
  if row < 0 || row >= rep.rows || column < 0 || column >= rep.columns {
    raise IndexOutOfBounds
  }
  let value = decode_element(rep.element, self.payload_code(row, column))
  let scaled = if rep.scale_element == Absent {
    value
  } else {
    let index = rep.scale_index(row, column)
    let width = rep.scale_element.bytes()
    value *
    decode_scale(
      rep.scale_element,
      read_code(self.scales, (index * width.to_int64()).to_int(), width),
    )
  }
  let reconstructed = if rep.global_scale {
    scaled * self.global_scale
  } else {
    scaled
  }
  if reconstructed.is_nan() || reconstructed.is_inf() {
    raise NonFiniteValue
  }
  reconstructed
}

///|
/// Startup/offline bounded conversion into caller-owned storage. No allocation
/// or file/device calls occur here; this is NOT a token-step fallback.
pub fn PackedTensor::decode_bf16_chunk(
  self : PackedTensor,
  first_element : Int64,
  element_count : Int,
  destination : FixedArray[Byte],
  destination_offset? : Int = 0,
) -> Unit raise PrecisionError {
  let total = self.representation.rows.to_int64() *
    self.representation.columns.to_int64()
  if first_element < 0L ||
    element_count < 0 ||
    first_element > total ||
    element_count.to_int64() > total - first_element {
    raise IndexOutOfBounds
  }
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    element_count > (destination.length() - destination_offset) / 2 {
    raise OutputTooSmall
  }
  for i in 0..<element_count {
    let index = first_element + i.to_int64()
    let columns = self.representation.columns.to_int64()
    let value = self.decode(
      (index / columns).to_int(),
      (index % columns).to_int(),
    )
    write_code(
      destination,
      destination_offset + i * 2,
      2,
      encode_element(Bf16, value),
    )
  }
}

///|
/// Explicit deterministic offline quantization. Dynamic activation/KV kernels
/// lower the same scale grid; the scalar oracle is not a serving implementation.
pub fn quantize(
  representation : Representation,
  values : ArrayView[Float],
) -> PackedTensor raise PrecisionError {
  let rep = representation
  let elements = rep.rows.to_int64() * rep.columns.to_int64()
  if elements != values.length().to_int64() {
    raise PayloadMismatch
  }
  if rep.payload_bytes > 2147483647L || rep.scale_bytes > 2147483647L {
    raise SizeOverflow
  }
  let maxima = FixedArray::make(
    (rep.scale_bytes / rep.scale_element.bytes().max(1).to_int64()).to_int(),
    0.0F,
  )
  let maximum = for i, value in values; maximum = 0.0F {
    if value.is_nan() || value.is_inf() {
      raise NonFiniteValue
    }
    if rep.scale_element != Absent {
      let scale = rep.scale_index(i / rep.columns, i % rep.columns).to_int()
      maxima[scale] = maxima[scale].max(value.abs())
    }
    continue maximum.max(value.abs())
  } nobreak {
    maximum
  }
  let global = if rep.global_scale && maximum > 0.0F {
    // Decode convention: actual = element * block_scale * global_scale.
    (maximum / 2688.0F).max(Float::reinterpret_from_uint(1U))
  } else {
    1.0F
  }
  let scales = FixedArray::make(rep.scale_bytes.to_int(), b'\x00')
  let scale_width = rep.scale_element.bytes()
  for i, amax in maxima {
    let requested = if amax == 0.0F {
      1.0F
    } else {
      // Dividing in this order avoids NVFP4 underflow before global division.
      (amax / global / rep.element.maximum()).max(
        Float::reinterpret_from_uint(1U),
      )
    }
    write_code(
      scales,
      i * scale_width,
      scale_width,
      encode_scale(rep.scale_element, requested),
    )
  }
  let scale_bytes = Bytes::from_array(scales[:])
  let payload = FixedArray::make(rep.payload_bytes.to_int(), b'\x00')
  for i, value in values {
    let row = i / rep.columns
    let column = i % rep.columns
    let normalized = if rep.scale_element == Absent {
      value
    } else {
      let scale = rep.scale_index(row, column).to_int()
      let block_scale = decode_scale(
        rep.scale_element,
        read_code(scale_bytes, scale * scale_width, scale_width),
      )
      value / global / block_scale
    }
    let code = encode_element(rep.element, normalized)
    let offset = row.to_int64() * rep.row_bytes
    if rep.element.bits() == 4 {
      let index = (offset + (column / 2).to_int64()).to_int()
      payload[index] = (payload[index].to_uint() | (code << (column % 2 * 4))).to_byte()
    } else {
      let width = rep.element.bits() / 8
      write_code(
        payload,
        (offset + column.to_int64() * width.to_int64()).to_int(),
        width,
        code,
      )
    }
  }
  PackedTensor::new(
    rep,
    payload=Bytes::from_array(payload[:]),
    scales=scale_bytes,
    global_scale=global,
  )
}
