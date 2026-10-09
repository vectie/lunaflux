///|
priv struct GlmMoeBranch {
  routed : @precision_ir.ExpertMlpPrecision
  shared : @precision_ir.ExpertMlpPrecision
  program : @packed_expert_mlp.MoeProgram
  source : Bytes
  banks : Array[@device.Allocation]
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
) -> Unit raise {
  let routed = context.allocate(self.routed.bank_bytes())
  self.banks.push(routed)
  let shared = context.allocate(self.shared.bank_bytes())
  self.banks.push(shared)
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
  )
}

///|
fn GlmMoeBranch::close(self : Self) -> Unit raise @device.DeviceError {
  self.program.close()
  while self.banks.last() is Some(bank) {
    bank.close()
    ignore(self.banks.pop())
  }
}
