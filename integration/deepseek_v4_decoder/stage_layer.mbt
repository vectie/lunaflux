///|
priv struct StageLayer {
  index : Int
  block : DeepSeekDecoderBlock
  routed : @precision.ExpertMlpPrecision
  shared : @precision.ExpertMlpPrecision
  router : @upload.CheckpointDeviceWeights
  banks : Array[@device.Allocation]
}

///|
fn StageLayer::bank_bytes(self : Self) -> Int64 {
  self.routed.bank_bytes() + self.shared.bank_bytes()
}

///|
// Resolve the entire checkpoint recipe before any stage allocation. Packing
// remains generic; family names and expert ordinals stay in this adapter.
fn StageLayer::required_device_bytes(
  self : Self,
  spec : @spec.DeepSeekV4ModelSpec,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  ignore(
    @expert.bind_experts(
      checkpoint,
      spec,
      layer_index=self.index,
      expert_indices=self.routed.local_experts(),
    ),
  )
  ignore(@expert.bind_shared_expert(checkpoint, spec, layer_index=self.index))
  @expert.bind_router(checkpoint, spec, layer_index=self.index).fold(
    init=self.bank_bytes() + self.block.required_device_bytes(checkpoint),
    (bytes, weight) => bytes + weight.buffer_bytes(),
  )
}

///|
fn StageLayer::upload(
  self : Self,
  spec : @spec.DeepSeekV4ModelSpec,
  context : @device.Context,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  self.banks.push(context.allocate(self.routed.bank_bytes()))
  self.banks.push(context.allocate(self.shared.bank_bytes()))
  @expert.upload_expert_bank(
    context,
    checkpoint,
    root,
    spec,
    self.routed,
    self.banks[0],
    layer_index=self.index,
    chunk_bytes~,
    limits~,
  )
  @expert.upload_shared_expert_bank(
    context,
    checkpoint,
    root,
    spec,
    self.shared,
    self.banks[1],
    layer_index=self.index,
    chunk_bytes~,
    limits~,
  )
  let weights = @expert.bind_router(checkpoint, spec, layer_index=self.index)
  self.router.upload(
    context,
    checkpoint,
    root,
    weights,
    maximum_device_bytes=weights.fold(init=0L, (bytes, weight) => {
      bytes + weight.buffer_bytes()
    }),
    chunk_bytes~,
    limits~,
  )
}

///|
fn StageLayer::close(self : Self) -> Unit raise {
  self.block.close()
  self.router.close()
  while self.banks.last() is Some(bank) {
    bank.close()
    ignore(self.banks.pop())
  }
}
