///|
/// Startup-only borrowed request ports. The stage owns no scheduler, HTTP
/// parser or transfer lifecycle. A batch worker may bind its stable addresses;
/// serial diagnostic callers use the compatibility preparation method below.
pub(all) struct GlmDecoderInputs {
  maximum_rows : Int
  maximum_history : Int
  maximum_sequences : Int
  maximum_outputs : Int
  vocabulary : Int
  counts : @device.Allocation
  token_ids : @device.Allocation
  sequence_offsets : @device.Allocation
  sequence_slots : @device.Allocation
  recurrent_resets : @device.Allocation
  sparse_resets : @device.Allocation
  output_count : @device.Allocation
  selected_rows : @device.Allocation
  sampled_tokens : @device.Allocation
  input_positions : @device.Allocation?
  page_ports : @history_frame.SparseHistoryPagePorts?
}

///|
pub fn GlmDecoderStage::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  ports : @serial_worker.SerialDecoderWorker,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  collective? : @device.TensorCollectiveOwner,
  generation? : UInt64 = 1UL,
  retirement? : @parallel.RetirementPolicy = EachCollective,
  replay? : @parallel.ReplayPolicy = Eager,
  replay_budget_bytes? : Int64 = 0L,
  overlap_experts? : Bool = false,
  fuse_hyper? : Bool = false,
  fuse_controls? : Bool = false,
  defer_hyper? : Bool = false,
) -> Unit raise {
  let (rows, history, vocabulary) = ports.geometry()
  self.prepare_inputs(
    context,
    aot,
    stream,
    checkpoint,
    root,
    {
      maximum_rows: rows,
      maximum_history: history,
      maximum_sequences: 1,
      maximum_outputs: 1,
      vocabulary,
      counts: ports.input(Counts),
      token_ids: ports.input(TokenIds),
      sequence_offsets: ports.input(SequenceOffsets),
      sequence_slots: ports.input(SequenceSlots),
      recurrent_resets: ports.input(RecurrentResets),
      sparse_resets: ports.input(SparseReset),
      output_count: ports.input(OutputCount),
      selected_rows: ports.input(SelectedRows),
      sampled_tokens: ports.sampled_tokens(),
      input_positions: None,
      page_ports: None,
    },
    maximum_device_bytes~,
    chunk_bytes~,
    limits~,
    collective?,
    generation~,
    retirement~,
    replay~,
    replay_budget_bytes~,
    overlap_experts~,
    fuse_hyper~,
    fuse_controls~,
    defer_hyper~,
  )
}
