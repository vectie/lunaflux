///|
test "structured doctor is versioned bounded and explicitly non-authoritative" {
  let report = build_operator_doctor_report(
    available_doctor_observation(13010, 2),
  )
  let document = render_operator_doctor_json(report)
  guard @json.parse(document) is Object(root) &&
    root["schema_version"] is String("lunaflux.operator-doctor.v1") &&
    root["protocol_version"] is String(protocol) &&
    root["checks"] is Array(checks) &&
    root["cuda"] is Object(cuda) &&
    cuda["available"] is True &&
    cuda["reason"] is String("none") &&
    cuda["driver_version"] is Number(13010.0, ..) &&
    cuda["device_count"] is Number(2.0, ..) &&
    root["readiness"] is False &&
    root["physical_cuda_authority"] is False &&
    root["external_tls_authority"] is False &&
    root["external_auth_authority"] is False else {
    fail("structured doctor report changed authority or schema")
  }
  assert_eq(protocol, @contracts.protocol_version())
  assert_eq(checks.length(), 9)
  for forbidden in ["model_root", "kernel_root", "credential", "secret"] {
    assert_false(document.contains(forbidden))
  }
}

///|
test "structured doctor preserves stable unavailable reasons" {
  let cases : Array[(@hardware.UnavailableReason, String)] = [
    (UnsupportedPlatform, "unsupported platform"),
    (DriverLibraryMissing, "driver library missing"),
    (DriverAbiIncomplete, "driver ABI incomplete"),
    (DriverInitializationFailed, "driver initialization failed"),
    (NoDevices, "no devices"),
    (InventoryFailed, "inventory failed"),
  ]
  for entry in cases {
    let (reason, expected) = entry
    let observation = unavailable_doctor_observation(reason)
    let document = render_operator_doctor_json(
      build_operator_doctor_report(observation),
    )
    guard @json.parse(document) is Object(root) &&
      root["cuda"] is Object(cuda) &&
      cuda["available"] is False &&
      cuda["reason"] is String(actual) &&
      cuda["driver_version"] is Number(0.0, ..) &&
      cuda["device_count"] is Number(0.0, ..) &&
      root["readiness"] is False else {
      fail("unavailable doctor observation became ambiguous")
    }
    assert_eq(actual, expected)
  }
}

///|
test "human doctor remains aligned with the structured projection" {
  let lines = operator_doctor_human_lines(
    unavailable_doctor_observation(UnsupportedPlatform),
  )
  assert_eq(lines[0], "LunaFlux foundation inspection")
  assert_true(
    lines.contains("CUDA device probe: unavailable; unsupported platform"),
  )
  assert_eq(lines[lines.length() - 1], "readiness: false")
  assert_true(
    lines.contains(
      "physical CUDA correctness and resource gates: not proven on this host",
    ),
  )
}
