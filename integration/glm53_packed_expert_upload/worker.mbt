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
