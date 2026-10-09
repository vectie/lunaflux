///|
/// Bind an already prepared decoder queue to the generic request bridge. The
/// caller prepared this slice's counts/CSR/reset inputs from worker.input(), and
/// included text ingress/egress in its launch list. Worker close drains this
/// borrowing queue before the caller releases text weights or residual frames.
/// A partial slice remains a stage, not a complete model; a multi-host runner
/// must bind the composed stage effects rather than publishing its first stage.
pub fn GlmDecoderSlice::bind_worker(
  self : Self,
  worker : @serial_worker.SerialDecoderWorker,
  context : @device.Context,
) -> Unit raise {
  guard self.status is Ready else {
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
