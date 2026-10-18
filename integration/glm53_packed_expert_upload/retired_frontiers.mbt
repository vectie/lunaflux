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

///|
/// Diagnostic-only leaf buffers; the stage guard establishes whole-program
/// retirement. Raw packed scratch includes inactive capacity and must not be
/// mistaken for a live mathematical output by an external comparison.
/// Dense layers have no expert buffers and return an empty view; lifecycle
/// and layer-range failures are still errors, never an empty success.
pub fn GlmDecoderStage::borrow_retired_experts(
  self : Self,
  layer_index : Int,
) -> ArrayView[(String, @device.Allocation)] raise {
  ignore(self.borrow_retired_ffn(layer_index))
  let ffn = match self.decoder.layers[layer_index - self.layer_range().0] {
    Recurrent(layer) => layer.ffn.unwrap()
    Sparse(layer) => layer.ffn
  }
  match ffn.branch {
    Dense(_) => []
    Moe(moe) => moe.program.borrow_retired_frontiers()
  }
}
