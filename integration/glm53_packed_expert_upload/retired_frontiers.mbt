///|
/// Borrow persistent, post-retirement FFN boundaries for an offline observer.
/// No observation is installed in execution plans or token-step submission.
/// The caller must finish every read before resubmitting or closing this stage.
/// Returns live rows, full-stream attention publication, normalized FFN input,
/// and the combined FFN output. These buffers are distinct for every layer;
/// the slice's reusable residual ping-pong buffers are deliberately not exposed.
pub fn GlmDecoderStage::borrow_retired_ffn(
  self : Self,
  layer_index : Int,
) -> (Int, @device.Allocation, @device.Allocation, @device.Allocation) raise {
  let (first, end) = self.layer_range()
  guard self.status is Ready &&
    self.decoder.status is Ready &&
    self.active_rows > 0 &&
    layer_index >= first &&
    layer_index < end else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  let ffn = match self.decoder.layers[layer_index - first] {
    Recurrent(layer) => {
      guard layer.ffn is Some(ffn) else {
        raise @device.DeviceError::InvalidArgument(StreamingTransfer)
      }
      ffn
    }
    Sparse(layer) => layer.ffn
  }
  (
    self.active_rows,
    ffn.input(),
    ffn.envelope.branch_input(),
    ffn.envelope.branch_output(),
  )
}
