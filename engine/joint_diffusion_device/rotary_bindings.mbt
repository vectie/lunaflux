///|
/// Borrowed, immutable F32 rotary tables. The allocation is caller-owned even
/// when upload fails. Release frame queues before releasing this allocation.
/// This adapter knows rows and rotary width, not model family or positions.
pub struct RotaryBindings {
  priv allocation : @device.Allocation
  priv base : Int64
  priv table_bytes : Int64
  priv rows : Int
  priv width : Int
}

///|
fn rotary_chunk(values : ArrayView[Float]) -> Bytes {
  let bytes : Array[Byte] = Array::make(values.length() * 4, 0)
  for i, value in values {
    let bits = value.reinterpret_as_uint()
    for byte in 0..<4 {
      bytes[i * 4 + byte] = ((bits >> (byte * 8)) & 255U).to_byte()
    }
  }
  Bytes::from_array(bytes)
}

///|
/// Upload once before any execution. Chunking bounds temporary host bytes;
/// no table construction or upload is required at denoising-step submission.
/// Retry a failed upload in full before using the allocation.
pub fn RotaryBindings::upload(
  context : @device.Context,
  allocation : @device.Allocation,
  cosine : ArrayView[Float],
  sine : ArrayView[Float],
  rows~ : Int,
  rotary_width~ : Int,
  offset~ : Int64,
  chunk_elements~ : Int,
) -> Self raise @device.DeviceError {
  let elements = rows.to_int64() * rotary_width.to_int64()
  guard rows > 0 &&
    rotary_width > 0 &&
    rotary_width % 2 == 0 &&
    elements == cosine.length().to_int64() &&
    elements == sine.length().to_int64() &&
    chunk_elements > 0 &&
    chunk_elements <= 536870911 else {
    raise @device.DeviceError::InvalidArgument(CopyToDevice)
  }
  let table_bytes = elements * 4L
  context.validate_allocation_region(
    allocation,
    offset~,
    byte_count=table_bytes * 2L,
    alignment=4L,
  )
  // Validate both tables before the first write, so malformed coefficients
  // cannot leave an otherwise successful-looking partially uploaded pair.
  for values in [cosine, sine] {
    for value in values {
      guard (value.reinterpret_as_uint() & 0x7f800000U) != 0x7f800000U else {
        raise @device.DeviceError::InvalidArgument(CopyToDevice)
      }
    }
  }
  for table, values in [cosine, sine] {
    for start = 0; start < values.length(); {
      let count = Int::min(chunk_elements, values.length() - start)
      let bytes = rotary_chunk(values[start:start + count])
      allocation.copy_from_host(
        bytes,
        0L,
        offset + table.to_int64() * table_bytes + start.to_int64() * 4L,
        count.to_int64() * 4L,
      )
      continue start + count
    }
  }
  { allocation, base: offset, table_bytes, rows, width: rotary_width, }
}

///|
pub fn RotaryBindings::cosine(
  self : Self,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    self.allocation,
    offset=self.base,
    byte_count=self.table_bytes,
    alignment=4L,
  )
}

///|
pub fn RotaryBindings::sine(
  self : Self,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    self.allocation,
    offset=self.base + self.table_bytes,
    byte_count=self.table_bytes,
    alignment=4L,
  )
}
