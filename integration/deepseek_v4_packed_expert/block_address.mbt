///|
fn base_block(
  spec : @spec.DeepSeekV4ModelSpec,
  index : Int,
) -> @spec.DeepSeekV4BlockAddress raise {
  guard index >= 0 && index < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  BaseLayer(index)
}
