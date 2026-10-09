///|
fn append_streaming_contract(
  encoder : BootstrapDigestEncoder,
  capability : @streaming_ir.Capability,
) -> Unit {
  for
    value in [
      capability.slots(),
      capability.page_count(),
      capability.reserve_pages(),
      capability.transfer_timeout_millis(),
      capability.shape_count(),
    ] {
    encoder.append_int(value)
  }
  for
    value in [
      capability.host_bytes(),
      capability.page_bytes(),
      capability.restore_ns(),
      capability.recompute_ns(),
      capability.workspace_budget(),
    ] {
    encoder.append_int64(value)
  }
  for index in 0..<capability.shape_count() {
    encoder.append_int(capability.shape_at(index).tokens())
    encoder.append_int64(capability.shape_at(index).workspace_bytes())
  }
}
