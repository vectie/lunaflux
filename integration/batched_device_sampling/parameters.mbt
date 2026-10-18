///|
pub fn BatchedDeviceSampler::prepare(
  self : Self,
  plan : @wire.ValidatedPlanFrame,
) -> Unit raise {
  guard self.state is (Ready | Retired) else {
    raise @device.DeviceError::Busy(DeviceCopyCreate)
  }
  self.state = Failed
  self.count = 0
  let (_, _, maximum_rows) = self.program.geometry()
  for entry in 0..<(plan.prefill_count() + plan.decode_count()) {
    let prefill = entry < plan.prefill_count()
    if prefill && !plan.prefill_row(entry).produces_token() {
      continue
    }
    guard self.count < maximum_rows else {
      raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
    }
    let offset = 8 + self.count * 40
    if prefill {
      let row = plan.prefill_row(entry)
      put_i32(self.parameters, offset, row.sampling_mode_value())
      put_i32(self.parameters, offset + 4, row.sampling_top_k_value())
      put_u64(
        self.parameters,
        offset + 8,
        row.sampling_temperature().reinterpret_as_uint64(),
      )
      put_u64(
        self.parameters,
        offset + 16,
        row.sampling_top_p_value().reinterpret_as_uint64(),
      )
      put_u64(self.parameters, offset + 24, row.sampling_seed_value())
      put_i32(self.parameters, offset + 32, row.sample_index())
      put_i32(
        self.parameters,
        offset + 36,
        (if row.sampling_has_top_k() { 1 } else { 0 }) |
        (if row.sampling_has_top_p() { 2 } else { 0 }),
      )
    } else {
      let row = plan.decode_row(entry - plan.prefill_count())
      put_i32(self.parameters, offset, row.sampling_mode_value())
      put_i32(self.parameters, offset + 4, row.sampling_top_k_value())
      put_u64(
        self.parameters,
        offset + 8,
        row.sampling_temperature().reinterpret_as_uint64(),
      )
      put_u64(
        self.parameters,
        offset + 16,
        row.sampling_top_p_value().reinterpret_as_uint64(),
      )
      put_u64(self.parameters, offset + 24, row.sampling_seed_value())
      put_i32(self.parameters, offset + 32, row.sample_index())
      put_i32(
        self.parameters,
        offset + 36,
        (if row.sampling_has_top_k() { 1 } else { 0 }) |
        (if row.sampling_has_top_p() { 2 } else { 0 }),
      )
    }
    self.count += 1
  }
  put_i32(self.parameters, 0, self.count)
  let bytes = 8 + self.count * 40
  self.upload.write_host(
    self.parameters,
    host_offset=0L,
    source_offset=0,
    bytes~,
  )
  self.upload.enqueue(
    lane=0,
    host_offset=0L,
    device_offset=0L,
    bytes=bytes.to_int64(),
    to_device=true,
  )
  self.upload.record(0)
  self.state = Prepared
}
