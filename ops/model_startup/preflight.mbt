///|
fn physical_gate(
  reason : @hardware.UnavailableReason,
) -> LunaModelPreparationOutcome {
  LunaModelPhysicalGate(
    match reason {
      UnsupportedPlatform => UnsupportedPlatform
      DriverLibraryMissing => DriverLibraryMissing
      DriverAbiIncomplete => DriverAbiIncomplete
      DriverInitializationFailed => DriverInitializationFailed
      NoDevices => NoDevices
      InventoryFailed => InventoryFailed
    },
  )
}

///|
/// Performs the legacy one-root preflight. Because no independently supplied
/// runtime-descriptor digest exists in this form, it stops at that first absent
/// prerequisite and does not probe hardware out of order.
pub fn preflight(
  model_root : String,
) -> LunaModelPreparationOutcome raise LunaModelStartupError {
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root) catch {
    _ => raise Failed(ModelRootAdmission)
  }
  root.close() catch {
    _ => raise Failed(ModelRootClose)
  }
  LunaModelNotPrepared(DigestPinnedRuntimeDescriptorAdmission)
}

///|
pub fn outcome_line(outcome : LunaModelPreparationOutcome) -> String {
  match outcome {
    LunaModelPreflightComplete =>
      "preflight complete: production activation was not attempted"
    LunaModelNotPrepared(DigestPinnedRuntimeDescriptorAdmission) =>
      "not prepared: independently pinned runtime descriptor was not supplied"
    LunaModelPhysicalGate(reason) => {
      let label = match reason {
        UnsupportedPlatform => "unsupported platform"
        DriverLibraryMissing => "driver library missing"
        DriverAbiIncomplete => "driver ABI incomplete"
        DriverInitializationFailed => "driver initialization failed"
        NoDevices => "no devices"
        InventoryFailed => "inventory failed"
        AssignedDeviceMissing => "assigned device missing"
        DeviceTargetMismatch => "assigned device target mismatch"
      }
      "physical gate: " + label
    }
  }
}
