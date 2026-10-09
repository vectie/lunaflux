///|
/// Upload frontend, blocks and mergers directly from checkpoint slices. This
/// preserves the existing group order and runtime allocation ABI.
pub fn VisionWeights::prepare_streamed(
  self : Self,
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  component : @minimax_h3_host_materialization.MiniMaxH3StreamingComponent,
  max_bytes~ : Int64,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard !self.ready &&
    self.allocations.is_empty() &&
    chunk_bytes > 0 &&
    component.component() == TextConditioner else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let groups = [@vision.frontend_weights()]
  for i in 0..<27 {
    groups.push(@vision.block_weights(i))
  }
  for i in 0..<4 {
    groups.push(@vision.merger_weights(i))
  }
  let sizes : Array[Int64] = []
  let requests : Array[@streaming_safetensors.StreamingSafetensorsTransfer] = []
  let mut remaining = max_bytes
  for i, group in groups {
    guard group.bytes > 0L && group.bytes <= remaining else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    remaining -= group.bytes
    sizes.push(group.bytes)
    let copies : Array[@minimax_h3_weights.BlockWeightCopy] = group.regions[:].map(r => {
        @minimax_h3_weights.BlockWeightCopy::new(
          tensor_name=r.name,
          source_offset=0L,
          destination_offset=r.offset,
          byte_count=r.bytes,
        )
      },
    )
    requests.append(component.transfers(copies[:], destination_ordinal=i))
  }
  for bytes in sizes {
    self.allocations.push(context.allocate(bytes))
  }
  @minimax_h3_block_upload.upload_streamed(
    context,
    root,
    component,
    self.allocations[:],
    requests[:],
    destination_bytes=sizes[:],
    chunk_bytes~,
  )
  self.ready = true
}
