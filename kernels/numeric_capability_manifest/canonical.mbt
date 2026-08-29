///|
fn append_manifest_i64(output : Array[Byte], value : Int64) -> Unit {
  for byte in value.reinterpret_as_uint64().to_le_bytes() {
    output.push(byte)
  }
}

///|
fn append_manifest_int(output : Array[Byte], value : Int) -> Unit {
  append_manifest_i64(output, Int64::from_int(value))
}

///|
fn append_manifest_text(output : Array[Byte], value : String) -> Unit {
  let encoded = @utf8.encode(value)
  append_manifest_int(output, encoded.length())
  for byte in encoded {
    output.push(byte)
  }
}

///|
fn append_manifest_entry_point(
  output : Array[Byte],
  entry : @catalog.AotKernelEntryPoint,
) -> Unit {
  append_manifest_text(output, entry.family().module_digest().sha256())
  append_manifest_int(output, entry.family().family_id().as_int())
  append_manifest_int(output, entry.entry_point_id().as_int())
}

///|
fn encode_fp8_kernel_manifest(
  scope : Fp8KernelExecutionScope,
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  catalog_version : @catalog.CatalogVersion,
  entries : ReadOnlyArray[Fp8KernelManifestEntry],
  maximum : Int,
) -> Bytes raise Fp8KernelManifestError {
  let output : Array[Byte] = []
  for byte in b"lunaflux.fp8-kernel-capability-manifest.v1\x00" {
    output.push(byte)
  }
  append_manifest_int(output, 1)
  append_manifest_int(
    output,
    match scope {
      StatelessFullContextV1 => 1
      PagedFullGraphV1 => 2
    },
  )
  append_manifest_text(output, plan.identity().content().as_hex())
  append_manifest_text(output, plan.identity().plan().as_hex())
  append_manifest_text(output, plan.numeric_binding().digest().as_hex())
  append_manifest_text(output, device_numeric.digest().as_hex())
  append_manifest_int(output, target.compute_major())
  append_manifest_int(output, target.compute_minor())
  append_manifest_int(output, if target.supports_bf16() { 1 } else { 0 })
  append_manifest_int(output, if target.supports_cublas_lt() { 1 } else { 0 })
  append_manifest_int(output, catalog_version.as_int())
  append_manifest_int(output, entries.length())
  for entry in entries {
    append_manifest_int(output, entry.profile.as_int())
    append_manifest_int(output, entry.operation.as_int())
    append_manifest_text(output, entry.execution.as_hex())
    append_manifest_entry_point(output, entry.entry_point)
    append_manifest_int(output, entry.artifact_module_index)
    append_manifest_int(output, entry.weights.length())
    for weight in entry.weights {
      append_manifest_int(output, weight.tensor.as_int())
      append_manifest_text(output, weight.storage.as_hex())
      append_manifest_int(output, weight.scale.tensor().as_int())
      append_manifest_text(output, weight.scale_storage.as_hex())
    }
  }
  if output.length() > maximum {
    raise InvalidFp8KernelManifest(CanonicalBytes)
  }
  Bytes::from_array(output)
}
