///|
fn append_startup_int(output : Array[Byte], value : Int) -> Unit {
  for byte in Int64::from_int(value).reinterpret_as_uint64().to_le_bytes() {
    output.push(byte)
  }
}

///|
fn append_startup_text(output : Array[Byte], value : String) -> Unit {
  let encoded = @utf8.encode(value)
  append_startup_int(output, encoded.length())
  for byte in encoded {
    output.push(byte)
  }
}

///|
fn encode_fp8_startup_admission(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts,
  maximum : Int,
) -> Bytes raise Fp8StartupAdmissionError {
  let output : Array[Byte] = []
  for byte in b"lunaflux.fp8-startup-admission.v1\x00" {
    output.push(byte)
  }
  append_startup_int(output, 1)
  append_startup_text(output, plan.identity().content().as_hex())
  append_startup_text(output, plan.identity().plan().as_hex())
  append_startup_text(output, plan.numeric_binding().digest().as_hex())
  append_startup_text(output, device_numeric.digest().as_hex())
  append_startup_text(output, kernels.manifest().digest().as_hex())
  if output.length() > maximum {
    raise Fp8StartupRejected(CanonicalBytes)
  }
  Bytes::from_array(output)
}
