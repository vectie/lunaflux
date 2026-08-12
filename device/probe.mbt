///|
fn unavailable_reason(
  availability : @cuda.DriverAvailability,
) -> UnavailableReason {
  match availability {
    UnsupportedPlatform => UnsupportedPlatform
    DriverLibraryMissing => DriverLibraryMissing
    DriverAbiIncomplete => DriverAbiIncomplete
    DriverInitializationFailed => DriverInitializationFailed
    NoDevices => NoDevices
    InventoryFailed | Available => InventoryFailed
  }
}

///|
fn capability_from_raw(raw : @cuda.RawDeviceInfo) -> DeviceCapability {
  {
    compute_major: raw.compute_major(),
    compute_minor: raw.compute_minor(),
    total_memory_bytes: raw.total_memory_bytes(),
    supports_bf16: raw.supports_bf16(),
    supports_cublas_lt: raw.cublas_lt_available(),
  }
}

///|
/// Discovers the process-visible CUDA inventory without creating a context or
/// allocating device memory.
pub fn probe() -> ProbeResult {
  let inventory = @cuda.probe()
  match inventory.availability() {
    Available => {
      let devices = []
      for ordinal in 0..<inventory.device_count() {
        try @cuda.device_info(ordinal) catch {
          _ => return Unavailable(InventoryFailed)
        } noraise {
          raw =>
            devices.push({
              ordinal,
              name: raw.name(),
              capability: capability_from_raw(raw),
            })
        }
      }
      Available(
        driver_version=inventory.driver_version(),
        devices=ReadOnlyArray::from_array(devices),
      )
    }
    unavailable => Unavailable(unavailable_reason(unavailable))
  }
}
