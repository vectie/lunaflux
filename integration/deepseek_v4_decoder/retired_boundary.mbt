///|
/// Visit a single-layer diagnostic's retained numerical boundaries. Only a
/// fully retired slice lends these ports; the observer must not close them.
/// The callback performs optional offline I/O, never ordinary serving work.
pub fn DeepSeekDecoderSlice::visit_retired_boundaries(
  self : Self,
  observe : (String, @device.Allocation, Int) -> Unit raise,
) -> Unit raise {
  guard self.state is Prepared && self.layers.length() == 1 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  let block = self.layers[0].block
  let width = self.spec.hidden_size()
  block.attention.with_branch_ports((input, output) => {
    observe("attention-input", input, width)
    observe("attention-output", output, width)
  })
  block.attention.visit_retired_components(observe)
  guard block.intermediate is Some(intermediate) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  observe(
    "attention-residual",
    intermediate,
    width * self.spec.hyper_connection_width(),
  )
  block.ffn.with_branch_ports((input, output) => {
    observe("ffn-input", input, width)
    observe("ffn-output", output, width)
  })
}
