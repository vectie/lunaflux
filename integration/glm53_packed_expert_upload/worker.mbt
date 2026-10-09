///|
/// Only a complete, prepared text+decoder owner can publish a model token.
/// Partial stages are bound by a composed pipeline, not independently sampled.
pub fn GlmDecoderStage::bind_worker(
  self : Self,
  worker : @serial_worker.SerialDecoderWorker,
  context : @device.Context,
) -> Unit raise {
  guard self.status is Ready &&
    self.range.text_boundary() is Complete &&
    self.counts is Some(counts) &&
    counts.same_storage(worker.input(Counts)) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  worker.bind_executor(
    context,
    submit=rows => self.submit(live_rows=rows),
    poll=() => self.poll(),
    close=() => self.close(),
    append_descriptors=self.append_descriptors(),
  )
}

///|
/// Bind local execution only. Only an egress/complete text boundary may return
/// sampled output, and no rank publishes a whole-model wire completion.
pub fn GlmDecoderStage::bind_rank(
  self : Self,
  rank : @serial_worker.SerialDecoderRank,
  context : @device.Context,
) -> Unit raise {
  let outputs_token = self.range.text_boundary() is (Egress | Complete)
  guard self.status is Ready &&
    rank.outputs_token() == outputs_token &&
    self.counts is Some(counts) &&
    counts.same_storage(rank.ports().input(Counts)) else {
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
