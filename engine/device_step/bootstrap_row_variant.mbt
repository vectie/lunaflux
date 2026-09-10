///|
/// Preserve the original row-variant encoding when the optional work-domain
/// launch is absent; new executables bind the extra launch choice at startup.
fn BootstrapEncoder::append_projection_variant(
  self : BootstrapEncoder,
  variant : @artifact.KernelRowVariant,
) -> Unit raise DeviceBootstrapError {
  self.append_int(variant.maximum_token_rows())
  self.append_string(variant.function_symbol())
  self.append_dimensions(variant.dimensions())
  if variant.companion_dimensions() is Some(companion) {
    self.append_string("companion-launch-v1")
    self.append_dimensions(companion)
  }
  if variant.bounded_work_grid_x() is Some(grid) {
    self.append_string("bounded-work-grid-x-v1")
    self.append_int(grid)
  }
}
