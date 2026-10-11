///|
/// Borrow the existing vocabulary output for an explicit offline observer.
/// This creates no transfer, allocation or token-path diagnostic by itself.
/// The caller retires execution before reading and closes any transfer lease
/// before the stage. Ordinary generation does not use this accessor.
pub fn DeepSeekDecoderStage::borrow_text_logits(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared && self.egress is Some(head) else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  head.logits()
}
