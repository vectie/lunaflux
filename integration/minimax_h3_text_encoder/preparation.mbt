///|
/// Production preparation: one complete ordered queue, without observations.
pub fn EncoderRequest::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  weights : EncoderWeights,
  ids : ArrayView[Int],
  max_device_bytes~ : Int64,
  max_table_bytes~ : Int64,
  visual? : (@device.Allocation, @text_plan.VisualPlacement),
) -> Unit raise {
  self.prepare_execution(
    context,
    stream,
    aot_module,
    weights,
    ids,
    max_device_bytes~,
    max_table_bytes~,
    visual?,
    observation=Disabled,
  )
}

///|
/// Diagnostic preparation only. Partition the identical launch list at hidden
/// state boundaries; no different kernels, arithmetic or checkpoint weights.
/// Normal production preparation must not use this synchronized route.
pub fn EncoderRequest::prepare_observed(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  weights : EncoderWeights,
  ids : ArrayView[Int],
  max_device_bytes~ : Int64,
  max_table_bytes~ : Int64,
  visual? : (@device.Allocation, @text_plan.VisualPlacement),
) -> Unit raise {
  self.prepare_execution(
    context,
    stream,
    aot_module,
    weights,
    ids,
    max_device_bytes~,
    max_table_bytes~,
    visual?,
    observation=HiddenLayers,
  )
}

///|
/// Explicit text-only numerical diagnostic. Run the unchanged embedding and
/// first layer, observing each operator output. This prefix never publishes a
/// complete fifty-layer result through with_output().
pub fn EncoderRequest::prepare_operator_observed(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  weights : EncoderWeights,
  ids : ArrayView[Int],
  max_device_bytes~ : Int64,
  max_table_bytes~ : Int64,
) -> Unit raise {
  self.prepare_execution(
    context,
    stream,
    aot_module,
    weights,
    ids,
    max_device_bytes~,
    max_table_bytes~,
    observation=FirstLayerOperators,
  )
}
