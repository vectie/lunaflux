///|
priv struct GlmMoeBranch {
  routed : @precision_ir.ExpertMlpPrecision
  shared : @precision_ir.ExpertMlpPrecision
  program : @packed_expert_mlp.MoeProgram
  source : Bytes
  banks : Array[@device.Allocation]
  tensor_partition : @precision_ir.ExpertTensorPartition?
  shared_partition : @precision_ir.ExpertTensorPartition?
  mut tensor_boundary : @packed_expert_mlp.ExpertTensorBoundary?
}

///|
fn GlmMoeBranch::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  layer_index~ : Int,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  router_weight~ : @device.Allocation,
  correction_bias~ : @device.Allocation,
  transient_scratch? : @device.Allocation,
) -> Unit raise {
  let routed = context.allocate(self.routed.bank_bytes())
  self.banks.push(routed)
  let shared = context.allocate(self.shared.bank_bytes())
  self.banks.push(shared)
  match self.tensor_partition {
    Some(partition) =>
      upload_tensor_expert_bank(
        context,
        checkpoint,
        root,
        partition,
        routed,
        layer_index~,
        chunk_bytes~,
        limits~,
      )
    None =>
      upload_expert_bank(
        context,
        checkpoint,
        root,
        self.routed,
        routed,
        layer_index~,
        chunk_bytes~,
        limits~,
      )
  }
  if self.shared_partition is Some(partition) {
    let bound = bind_shared_expert(checkpoint, layer_index~)
    @packed_expert_mlp.upload_bank(
      context,
      checkpoint,
      root,
      self.shared,
      shared,
      weights=bound.mapi((i, weight) => weight.slice(partition.slices()[i])),
      chunk_bytes~,
      limits~,
    )
  } else {
    upload_shared_expert_bank(
      context,
      checkpoint,
      root,
      self.shared,
      shared,
      layer_index~,
      chunk_bytes~,
      limits~,
    )
  }
  if self.tensor_partition is Some(_) {
    self.tensor_boundary = Some(
      self.program.prepare_routed_tensor_frame(
        context,
        aot,
        maximum_workspace_bytes=self.program.workspace_bytes(),
        counts~,
        input~,
        router_weight~,
        correction_bias~,
        routed_bank=routed,
        shared_bank=shared,
        output~,
        transient_scratch?,
      ),
    )
  } else {
    self.program.prepare_routed_frame(
      context,
      aot,
      maximum_workspace_bytes=self.program.workspace_bytes(),
      counts~,
      input~,
      router_weight~,
      correction_bias~,
      routed_bank=routed,
      shared_bank=shared,
      output~,
      transient_scratch?,
    )
  }
}

///|
fn GlmMoeBranch::close(self : Self) -> Unit raise @device.DeviceError {
  self.program.close()
  while self.banks.last() is Some(bank) {
    bank.close()
    ignore(self.banks.pop())
  }
}
