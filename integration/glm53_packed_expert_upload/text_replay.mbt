///|
/// Offline ports from GLM's actual mean-stream head, not a DeepSeek surrogate.
/// Retire the queue before capture; caller owns absolute token/position mapping.
pub fn GlmTextIo::replay_buffers(
  self : Self,
) -> Array[(@precision_ir.NumericBoundary, @device.Allocation)] raise {
  self.frame.replay_buffers()
}

///|
pub fn GlmTextIo::projection_suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  self.frame.projection_suffix()
}

///|
/// Borrow the already allocated vocabulary output for an offline observer.
/// Caller must retire execution before reading and close transfer leases before
/// this stage. Ordinary execution allocates no observer and performs no copy.
pub fn GlmDecoderStage::borrow_text_logits(
  self : Self,
) -> @device.Allocation raise {
  guard self.status is Ready && self.text_boundary().has_egress() else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  self.text.logits()
}
