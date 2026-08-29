///|
priv struct OperatorDoctorCheck {
  code : String
  status : String
} derive(ToJson)

///|
priv struct OperatorDoctorCuda {
  available : Bool
  reason : String
  driver_version : Int
  device_count : Int
} derive(ToJson)

///|
priv struct OperatorDoctorReport {
  schema_version : String
  protocol_version : String
  checks : Array[OperatorDoctorCheck]
  cuda : OperatorDoctorCuda
  readiness : Bool
  physical_cuda_authority : Bool
  external_tls_authority : Bool
  external_auth_authority : Bool
} derive(ToJson)

///|
priv struct OperatorDoctorObservation {
  cuda : OperatorDoctorCuda
  cuda_line : String
}

///|
fn unavailable_device_reason(reason : @hardware.UnavailableReason) -> String {
  match reason {
    UnsupportedPlatform => "unsupported platform"
    DriverLibraryMissing => "driver library missing"
    DriverAbiIncomplete => "driver ABI incomplete"
    DriverInitializationFailed => "driver initialization failed"
    NoDevices => "no devices"
    InventoryFailed => "inventory failed"
  }
}

///|
fn available_doctor_observation(
  driver_version : Int,
  device_count : Int,
) -> OperatorDoctorObservation {
  {
    cuda: { available: true, reason: "none", driver_version, device_count },
    cuda_line: "CUDA device probe: available; driver \{driver_version}; devices \{device_count}",
  }
}

///|
fn unavailable_doctor_observation(
  reason : @hardware.UnavailableReason,
) -> OperatorDoctorObservation {
  let reason_text = unavailable_device_reason(reason)
  {
    cuda: {
      available: false,
      reason: reason_text,
      driver_version: 0,
      device_count: 0,
    },
    cuda_line: "CUDA device probe: unavailable; \{reason_text}",
  }
}

///|
fn probe_doctor_observation() -> OperatorDoctorObservation {
  match @hardware.probe() {
    Available(driver_version~, devices~) =>
      available_doctor_observation(driver_version, devices.length())
    Unavailable(reason) => unavailable_doctor_observation(reason)
  }
}

///|
fn operator_doctor_checks() -> Array[OperatorDoctorCheck] {
  [
    { code: "contracts", status: "ready" },
    { code: "configuration_validators", status: "ready" },
    { code: "bounded_tokenizer_json_admission", status: "ready" },
    { code: "bounded_model_and_weight_manifest", status: "ready" },
    { code: "offline_reference_executor", status: "correctness_only" },
    { code: "bounded_reference_bundle", status: "whole_file_snapshot" },
    {
      code: "streaming_device_weight_loading",
      status: "approved_read_only_regular_file_contract",
    },
    { code: "exact_static_device_executor", status: "implemented" },
    { code: "physical_cuda_correctness", status: "not_proven" },
  ]
}

///|
fn build_operator_doctor_report(
  observation : OperatorDoctorObservation,
) -> OperatorDoctorReport {
  {
    schema_version: "lunaflux.operator-doctor.v1",
    protocol_version: @contracts.protocol_version(),
    checks: operator_doctor_checks(),
    cuda: observation.cuda,
    readiness: false,
    physical_cuda_authority: false,
    external_tls_authority: false,
    external_auth_authority: false,
  }
}

///|
fn render_operator_doctor_json(report : OperatorDoctorReport) -> String {
  ToJson::to_json(report).stringify()
}

///|
fn operator_doctor_human_lines(
  observation : OperatorDoctorObservation,
) -> Array[String] {
  [
    "LunaFlux foundation inspection",
    "contracts: ready",
    "configuration validators: ready",
    "bounded tokenizer JSON admission: ready",
    "bounded model metadata and weight-manifest validation: ready",
    observation.cuda_line,
    "offline reference executor: ready (correctness only)",
    "bounded reference bundle loading: ready (whole-file snapshot)",
    "streaming device weight loading: implemented (approved read-only regular-file contract)",
    "exact static device executor: implemented",
    "physical CUDA correctness and resource gates: not proven on this host",
    "readiness: false",
  ]
}

///|
fn print_doctor() -> Unit {
  let observation = probe_doctor_observation()
  for line in operator_doctor_human_lines(observation) {
    println(line)
  }
}

///|
fn print_doctor_json() -> Unit {
  let observation = probe_doctor_observation()
  println(
    render_operator_doctor_json(build_operator_doctor_report(observation)),
  )
}
