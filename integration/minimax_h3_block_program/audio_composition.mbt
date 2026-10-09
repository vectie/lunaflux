///|
/// Preflight is mutation-free, including duplicate-owner rejection. The caller
/// allocates storage next, then claims without further fallible work. Confined
/// to the same preparation/control thread as request leases.
fn audio_composition_preflight(
  leases : ArrayView[AudioDecoderLease],
) -> Unit raise @device.DeviceError {
  for i, lease in leases {
    lease.require_idle()
    for j in 0..<i {
      guard !physical_equal(lease, leases[j]) else {
        raise @device.DeviceError::InvalidArgument(AllocationCreate)
      }
    }
  }
}

///|
fn audio_composition_claim(leases : ArrayView[AudioDecoderLease]) -> Unit {
  for lease in leases {
    lease.held = true
  }
}

///|
fn audio_composition_release(leases : ArrayView[AudioDecoderLease]) -> Unit {
  for lease in leases {
    lease.held = false
  }
}
