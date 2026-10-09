///|
pub struct EncoderWeights {
  priv allocations : Array[@device.Allocation]
  priv mut ready : Bool
  priv mut leases : Int
}

///|
pub fn EncoderWeights::new() -> Self {
  { allocations: [], ready: false, leases: 0, }
}

///|
/// Startup-only materialization. Partial failure retains every acquired owner.
pub fn EncoderWeights::prepare(
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
  let first = @text_plan.layer_weights(0)
  let table = @text_plan.embedding_region()
  guard max_bytes > 0L && table.bytes + 50L * first.byte_count <= max_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let embedding = context.allocate(table.bytes)
  self.allocations.push(embedding)
  @minimax_h3_block_upload.upload_text_encoder_embedding(
    context,
    embedding,
    host,
    expected_identity~,
    chunk_bytes~,
  )
  for layer in 0..<50 {
    let packing = @text_plan.layer_weights(layer)
    let allocation = context.allocate(packing.byte_count)
    self.allocations.push(allocation)
    @minimax_h3_block_upload.upload_text_encoder_layer(
      context,
      allocation,
      host,
      packing,
      expected_identity~,
      chunk_bytes~,
    )
  }
  self.ready = true
}

///|
pub fn EncoderWeights::close(self : Self) -> Unit raise @device.DeviceError {
  guard self.leases == 0 else {
    raise @device.DeviceError::InvalidArgument(AllocationClose)
  }
  self.ready = false
  while !self.allocations.is_empty() {
    let last = self.allocations.length() - 1
    self.allocations[last].close()
    self.allocations.remove(last) |> ignore
  }
}
