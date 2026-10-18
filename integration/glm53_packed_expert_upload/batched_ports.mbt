///|
/// Model adapter only: scheduler-owned logical pages and recurrent slots remain
/// in the generic rank. All addresses are borrowed until rank.close().
pub fn GlmDecoderInputs::from_batched_rank(
  rank : @batch_worker.BatchedDecoderRank,
) -> Self raise {
  let (rows, sequences, _, history, vocabulary) = rank.geometry()
  {
    maximum_rows: rows,
    maximum_history: history,
    maximum_sequences: sequences,
    maximum_outputs: sequences,
    vocabulary,
    counts: rank.input(Counts),
    token_ids: rank.input(TokenIds),
    sequence_offsets: rank.input(SequenceOffsets),
    sequence_slots: rank.input(SequenceSlots),
    recurrent_resets: rank.input(RecurrentResets),
    sparse_resets: rank.input(SparseResets),
    output_count: rank.input(OutputCount),
    selected_rows: rank.input(SelectedRows),
    sampled_tokens: rank.sampled_tokens(),
    input_positions: Some(rank.input(Positions)),
    page_ports: Some(
      @history_frame.SparseHistoryPagePorts::new(
        row_sequences=rank.input(RowSequences),
        page_offsets=rank.input(PageOffsets),
        page_ids=rank.input(PageIds),
        page_generations=rank.input(PageGenerations),
      ),
    ),
  }
}

///|
pub fn GlmDecoderStage::bind_batched_rank(
  self : Self,
  rank : @batch_worker.BatchedDecoderRank,
  context : @device.Context,
) -> Unit raise {
  let outputs_token = self.range.text_boundary() is (Egress | Complete)
  guard self.status is Ready &&
    (
      rank.outputs_token() == outputs_token ||
      (!rank.outputs_token() && self.decoder.tensor_group is Some((0, _)))
    ) &&
    self.counts is Some(counts) &&
    counts.same_storage(rank.input(Counts)) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  rank.bind_executor(
    context,
    submit=rows => self.submit(live_rows=rows),
    poll=() => self.poll_submission(),
    close=() => self.close(),
    append_descriptors=self.append_descriptors(),
  )
}
