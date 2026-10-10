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
    observe_layers=false,
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
    observe_layers=true,
  )
}
