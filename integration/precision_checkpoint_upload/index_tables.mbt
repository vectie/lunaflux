///|
/// One source chunk, one output chunk and one uniqueness row per table. This
/// pure startup footprint does not retain a full I64/I32 table on the host.
pub fn i64_table_host_bytes(
  tables : ArrayView[@materialize.CheckpointI64Table],
  chunk_bytes~ : Int,
) -> Int64 raise @device.DeviceError {
  guard chunk_bytes > 0 else { raise InvalidArgument(AllocationCreate) }
  let output_bytes = (chunk_bytes.to_int64() + 7L) / 8L * 4L
  for table in tables; bytes = chunk_bytes.to_int64() + output_bytes {
    let (_, columns) = table.geometry()
    guard columns.to_int64() <= (9223372036854775807L - bytes) / 8L else {
      raise InvalidArgument(AllocationCreate)
    }
    continue bytes + columns.to_int64() * 8L
  } nobreak {
    bytes
  }
}

///|
/// Startup-only owner of checked I32 device tables derived from inspected
/// I64 planes. Failed uploads remain owned and cannot expose allocation views.
pub struct CheckpointDeviceIndexTables {
  priv allocations : Array[@device.Allocation]
  priv mut status : State
  priv mut bytes : Int64
}

///|
pub fn CheckpointDeviceIndexTables::new() -> Self {
  { allocations: [], status: Fresh, bytes: 0L, }
}

///|
pub fn CheckpointDeviceIndexTables::upload(
  self : Self,
  context : @device.Context,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  tables : ArrayView[@materialize.CheckpointI64Table],
  minimum_value~ : Int,
  exclusive_maximum_value~ : Int,
  maximum_device_bytes~ : Int64,
  maximum_host_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.status is Fresh &&
    !tables.is_empty() &&
    exclusive_maximum_value > minimum_value &&
    i64_table_host_bytes(tables, chunk_bytes~) <= maximum_host_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let bytes = for table in tables; total = 0L {
    let size = table.i32_bytes()
    guard size > 0L && size <= maximum_device_bytes - total else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    continue total + size
  } nobreak {
    total
  }
  let streams = tables.map(table => {
    let (rows, columns) = table.geometry()
    @narrow.I64RowNarrowStream::new(
      row_count=rows,
      column_count=columns,
      minimum_value~,
      exclusive_maximum_value~,
      maximum_row_state_bytes=columns.to_int64() * 8L,
    )
  })
  defer (for stream in streams {
    stream.close()
  })
  let scratch = FixedArray::make(chunk_bytes, b'\x00')
  let output = FixedArray::make(
    ((chunk_bytes.to_int64() + 7L) / 8L * 4L).to_int(),
    b'\x00',
  )
  self.status = Failed
  for table in tables {
    let allocation = context.allocate(table.i32_bytes())
    self.allocations.push(allocation)
    self.bytes += table.i32_bytes()
  }
  let mut narrowing_error : @narrow.CheckedIntegerNarrowError? = None
  let mut device_error : @device.DeviceError? = None
  checkpoint.transfer_i64_tables(root, tables, scratch~, limits~, consume=(
    destination,
    offset,
    buffer,
    count,
  ) => {
    let result = streams[destination].consume(offset, buffer[:count], output) catch {
      error => {
        narrowing_error = Some(error)
        return false
      }
    }
    let (output_offset, output_bytes) = result
    if output_bytes == 0 {
      return true
    }
    try
      context.copy_from_fixed_host(
        self.allocations[destination],
        output,
        0L,
        output_offset,
        output_bytes.to_int64(),
      )
    catch {
      error => {
        device_error = Some(error)
        false
      }
    } noraise {
      _ => true
    }
  }) catch {
    error => {
      if narrowing_error is Some(cause) {
        raise cause
      }
      if device_error is Some(cause) {
        raise cause
      }
      raise error
    }
  }
  for stream in streams {
    stream.finish()
  }
  self.bytes = bytes
  self.status = Ready
}

///|
pub fn CheckpointDeviceIndexTables::allocations(
  self : Self,
) -> ArrayView[@device.Allocation] raise @device.DeviceError {
  guard self.status is Ready else { raise InvalidArgument(AllocationCreate) }
  self.allocations[:]
}

///|
pub fn CheckpointDeviceIndexTables::device_bytes(self : Self) -> Int64 {
  self.bytes
}

///|
pub fn CheckpointDeviceIndexTables::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  while self.allocations.last() is Some(allocation) {
    let bytes = allocation.byte_count()
    allocation.close()
    ignore(self.allocations.pop())
    self.bytes -= bytes
  }
  self.status = Closed
}
