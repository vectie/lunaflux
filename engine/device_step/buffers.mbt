///|
priv struct DeviceStepHostBuffers {
  counts : FixedArray[Byte]
  token_ids : FixedArray[Byte]
  token_positions : FixedArray[Byte]
  query_row_offsets : FixedArray[Byte]
  sequence_lengths : FixedArray[Byte]
  page_offsets : FixedArray[Byte]
  physical_page_indices : FixedArray[Byte]
}

///|
priv struct DeviceStepResources {
  counts : @device.Allocation
  token_ids : @device.Allocation
  token_positions : @device.Allocation
  query_row_offsets : @device.Allocation
  sequence_lengths : @device.Allocation
  page_offsets : @device.Allocation
  physical_page_indices : @device.Allocation
}

///|
priv struct PartialDeviceStepResources {
  mut counts : @device.Allocation?
  mut token_ids : @device.Allocation?
  mut token_positions : @device.Allocation?
  mut query_row_offsets : @device.Allocation?
  mut sequence_lengths : @device.Allocation?
  mut page_offsets : @device.Allocation?
  mut physical_page_indices : @device.Allocation?
}

///|
fn make_host_buffers(limits : DeviceStepLimits) -> DeviceStepHostBuffers {
  {
    counts: FixedArray::make(COUNTS_LENGTH * INT32_BYTES, b'\x00'),
    token_ids: FixedArray::make(limits.max_tokens() * INT32_BYTES, b'\x00'),
    token_positions: FixedArray::make(
      limits.max_tokens() * INT32_BYTES,
      b'\x00',
    ),
    query_row_offsets: FixedArray::make(
      row_offset_cells(limits) * INT32_BYTES,
      b'\x00',
    ),
    sequence_lengths: FixedArray::make(limits.max_rows() * INT32_BYTES, b'\x00'),
    page_offsets: FixedArray::make(
      row_offset_cells(limits) * INT32_BYTES,
      b'\x00',
    ),
    physical_page_indices: FixedArray::make(
      limits.max_pages() * INT32_BYTES,
      b'\x00',
    ),
  }
}

///|
fn empty_partial_resources() -> PartialDeviceStepResources {
  {
    counts: None,
    token_ids: None,
    token_positions: None,
    query_row_offsets: None,
    sequence_lengths: None,
    page_offsets: None,
    physical_page_indices: None,
  }
}

///|
fn write_i32(buffer : FixedArray[Byte], index : Int, value : Int) -> Unit {
  buffer.unsafe_write_uint32_le(
    index * INT32_BYTES,
    value.reinterpret_as_uint(),
  )
}
