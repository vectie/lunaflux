///|
test "model startup hardware failures remain typed and payload safe" {
  let cases : FixedArray[(@hardware.UnavailableReason, LunaModelPhysicalGate)] = [
    (UnsupportedPlatform, UnsupportedPlatform),
    (DriverLibraryMissing, DriverLibraryMissing),
    (DriverAbiIncomplete, DriverAbiIncomplete),
    (DriverInitializationFailed, DriverInitializationFailed),
    (NoDevices, NoDevices),
    (InventoryFailed, InventoryFailed),
  ]
  for _, pair in cases {
    let (reason, expected) = pair
    assert_eq(physical_gate(reason), LunaModelPhysicalGate(expected))
  }
  assert_eq(
    outcome_line(LunaModelNotPrepared(DigestPinnedRuntimeDescriptorAdmission)),
    "not prepared: independently pinned runtime descriptor was not supplied",
  )
  assert_eq(
    outcome_line(LunaModelPreflightComplete),
    "preflight complete: production activation was not attempted",
  )
  assert_eq(
    outcome_line(LunaModelPhysicalGate(AssignedDeviceMissing)),
    "physical gate: assigned device missing",
  )
  assert_eq(
    outcome_line(LunaModelPhysicalGate(DeviceTargetMismatch)),
    "physical gate: assigned device target mismatch",
  )
}
