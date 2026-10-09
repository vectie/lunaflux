///|
priv enum State {
  Fresh
  Failed
  Ready
  Closed
}

///|
/// Startup-only owner for exact checkpoint device allocations. One scratch
/// chunk streams all weights without a host arena or expanded bank. Borrowed
/// allocations remain valid until this owner closes, after executors close.
pub struct CheckpointDeviceWeights {
  priv allocations : Array[@device.Allocation]
  priv mut status : State
  priv mut bytes : Int64
}

///|
pub fn CheckpointDeviceWeights::new() -> Self {
  { allocations: [], status: Fresh, bytes: 0L, }
}

///|
pub fn CheckpointDeviceWeights::upload(
  self : Self,
  context : @device.Context,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  weights : ArrayView[@materialize.PackedCheckpointWeight],
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.status is Fresh && !weights.is_empty() && chunk_bytes > 0 else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let bytes = for weight in weights; total = 0L {
    let size = weight.buffer_bytes()
    guard size > 0L && size <= maximum_device_bytes - total else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    continue total + size
  } nobreak {
    total
  }
  self.status = Failed
  for weight in weights {
    let allocation = context.allocate(weight.buffer_bytes())
    self.allocations.push(allocation)
    self.bytes += weight.buffer_bytes()
  }
  let scratch = FixedArray::make(chunk_bytes, b'\x00')
  defer scratch.fill(b'\x00')
  checkpoint.transfer_weights(root, weights, scratch~, limits~, consume=(
    destination,
    offset,
    buffer,
    count,
  ) => {
    try
      context.copy_from_fixed_host(
        self.allocations[destination],
        buffer,
        0L,
        offset,
        count.to_int64(),
      )
    catch {
      _ => false
    } noraise {
      _ => true
    }
  })
  self.bytes = bytes
  self.status = Ready
}

///|
pub fn CheckpointDeviceWeights::allocation(
  self : Self,
  index : Int,
) -> @device.Allocation raise @device.DeviceError {
  guard self.status is Ready && self.allocations.get(index) is Some(allocation) else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  allocation
}

///|
pub fn CheckpointDeviceWeights::device_bytes(self : Self) -> Int64 {
  self.bytes
}

///|
pub fn CheckpointDeviceWeights::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  match self.status {
    Closed => return
    Fresh | Ready | Failed => ()
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
