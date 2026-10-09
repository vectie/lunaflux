///|
/// Same embedding/layer packing as prepare, without a full host checkpoint
/// arena. All acquired allocations stay owned here if loading fails.
pub fn EncoderWeights::prepare_streamed(
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
  let groups : Array[Array[@text_plan.WeightRegion]] = [
    [@text_plan.embedding_region()],
  ]
  let sizes = [@text_plan.embedding_region().bytes]
  for layer in 0..<50 {
    let packing = @text_plan.layer_weights(layer)
    groups.push(packing.regions[:].to_owned())
    sizes.push(packing.byte_count)
  }
  let mut remaining = max_bytes
  let requests : Array[@streaming_safetensors.StreamingSafetensorsTransfer] = []
  for i, regions in groups {
    guard sizes[i] > 0L && sizes[i] <= remaining else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    remaining -= sizes[i]
    let copies : Array[@minimax_h3_weights.BlockWeightCopy] = regions.map(r => {
      @minimax_h3_weights.BlockWeightCopy::new(
        tensor_name=r.name,
        source_offset=0L,
        destination_offset=r.offset,
        byte_count=r.bytes,
      )
    })
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
