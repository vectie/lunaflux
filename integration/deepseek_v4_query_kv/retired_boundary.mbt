///|
/// Offline observer of borrowed branch ports. The enclosing executor must be
/// retired before calling; this adds no launch, allocation or serving scan.
pub fn[T] DeepSeekAttentionEnvelope::with_branch_ports(
  self : Self,
  observe : (@device.Allocation, @device.Allocation) -> T raise,
) -> T raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  observe(self.hyper.branch_input(), self.hyper.branch_output())
}

///|
/// Diagnostic ownership traversal only, after enclosing executor retirement.
/// Window and compressed branches expose the same retained numerical ports.
pub fn DeepSeekAttentionEnvelope::visit_retired_components(
  self : Self,
  observe : (String, @device.Allocation, Int) -> Unit raise,
) -> Unit raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  let (inputs, sparse, spec, output) = match self.branch {
    Window(branch) =>
      (
        branch.attention.inputs.outputs(),
        branch.attention.output(),
        branch.attention.spec,
        branch.output,
      )
    Compressed(branch) =>
      (
        branch.attention.inputs.outputs(),
        branch.attention.output(),
        branch.attention.spec,
        branch.output,
      )
  }
  observe("attention-low-rank", inputs.low_rank_queries, spec.q_lora_rank())
  observe(
    "q-rope",
    inputs.queries,
    spec.attention_head_count() * spec.head_dimension(),
  )
  observe("kv-sim", inputs.shared_kv, spec.head_dimension())
  observe(
    "sparse-attention",
    sparse,
    spec.attention_head_count() * spec.head_dimension(),
  )
  output.frame.visit_retired_outputs(observe)
}
