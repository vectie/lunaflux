///|
/// Final state/activation/metadata allocations, separate from weights, AOT
/// modules and the text egress owner's private normalization/logit workspace.
pub(all) struct StorageBudget {
  owned_bytes : Int64
  output_workspace_bytes : Int64
}

///|
pub fn Program::storage_budget(self : Self) -> StorageBudget {
  let decoder = self.decoder.budget()
  let attention = self.decoder.attention_program()
  let recurrent = self.decoder.recurrent_program()
  let (_, _, _, rows, outputs) = self.text.plan().geometry()
  let mutable_bytes = decoder.activation_bytes * 3L +
    decoder.branch_workspace_bytes +
    decoder.moe_workspace_bytes +
    decoder.convolution_state_bytes +
    decoder.recurrent_state_bytes +
    decoder.key_cache_bytes +
    decoder.value_cache_bytes
  let metadata_bytes = attention.region(Counts).byte_count +
    attention.region(CanvasPositions).byte_count +
    attention.region(CausalPositions).byte_count +
    attention.region(RowOffsets).byte_count +
    attention.region(SequenceLengths).byte_count +
    attention.region(PageOffsets).byte_count +
    attention.region(PhysicalPages).byte_count +
    recurrent.region(RequestSlots).byte_count +
    recurrent.region(Reset).byte_count +
    rows.to_int64() * 4L +
    4L +
    outputs.to_int64() * 8L
  {
    owned_bytes: mutable_bytes + metadata_bytes,
    output_workspace_bytes: self.workspace_bytes(),
  }
}
