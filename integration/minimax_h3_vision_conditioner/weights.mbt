///|
pub struct VisionWeights {
  priv allocations : Array[@device.Allocation]
  priv mut ready : Bool
  priv mut leases : Int
}

///|
pub fn VisionWeights::new() -> Self {
  { allocations: [], ready: false, leases: 0, }
}

///|
pub fn VisionWeights::prepare(
  self : Self,
  context : @device.Context,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  expected_identity~ : @spec.ModelIdentity,
  max_bytes~ : Int64,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard !self.ready && self.allocations.is_empty() && chunk_bytes > 0 else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let groups = [@vision.frontend_weights()]
  for i in 0..<27 {
    groups.push(@vision.block_weights(i))
  }
  for i in 0..<4 {
    groups.push(@vision.merger_weights(i))
  }
  for group in groups; remaining = max_bytes {
    guard group.bytes <= remaining else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    continue remaining - group.bytes
  } nobreak {
    ()
  }
  for group in groups {
    let allocation = context.allocate(group.bytes)
    self.allocations.push(allocation)
    @minimax_h3_block_upload.upload_vision_conditioner(
      context,
      allocation,
      host,
      group,
      expected_identity~,
      destination_offset=0L,
      chunk_bytes~,
    )
  }
  self.ready = true
}

///|
pub fn VisionWeights::close(self : Self) -> Unit raise @device.DeviceError {
  guard self.leases == 0 else {
    raise @device.DeviceError::InvalidArgument(AllocationClose)
  }
  self.ready = false
  while !self.allocations.is_empty() {
    let i = self.allocations.length() - 1
    self.allocations[i].close()
    self.allocations.remove(i) |> ignore
  }
}
