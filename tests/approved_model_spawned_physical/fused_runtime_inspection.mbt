///|
fn fused_runtime_inspection_value(
  text : String,
  key : String,
) -> String raise ApprovedModelSpawnedPhysicalError {
  let prefix = key + "="
  let mut found : String? = None
  for line in text.split("\n") {
    if line.has_prefix(prefix) {
      if found is Some(_) || line.length() == prefix.length() {
        raise DeviceGreedyQualificationRejected
      }
      found = Some(line[prefix.length():].to_owned())
    }
  }
  found.unwrap_or_else(() => raise DeviceGreedyQualificationRejected)
}

///|
fn fused_runtime_inspection_decode_hex(
  text : String,
) -> Bytes raise ApprovedModelSpawnedPhysicalError {
  if text.is_empty() || text.length() % 2 != 0 {
    raise DeviceGreedyQualificationRejected
  }
  fn nibble(value : UInt16) -> Byte raise ApprovedModelSpawnedPhysicalError {
    if value >= 48 && value <= 57 {
      (value.to_int() - 48).to_byte()
    } else if value >= 97 && value <= 102 {
      (value.to_int() - 87).to_byte()
    } else {
      raise DeviceGreedyQualificationRejected
    }
  }
  Bytes::makei(text.length() / 2, index => {
    (nibble(text[index * 2]) << 4) | nibble(text[index * 2 + 1])
  })
}

///|
async fn inspect_fused_production_runtime(
  runtime_path : String,
  expected_sha256 : String,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  if !runtime_path.has_prefix("/") || expected_sha256.length() != 64 {
    raise DeviceGreedyQualificationRejected
  }
  let bytes = @fs.read_file(runtime_path).binary() catch {
    _ => raise DeviceGreedyQualificationRejected
  }
  let runtime = @luna_paged_composite_artifact_admission.admit_digest_pinned_fused_production_runtime(
    bytes, expected_sha256,
  ) catch {
    _ => raise DeviceGreedyQualificationRejected
  }
  let text = @utf8.decode(bytes) catch {
    _ => raise DeviceGreedyQualificationRejected
  }
  let qkv = runtime.fused_bundle().qkv_module()
  let residual = runtime.fused_bundle().residual_module()
  let readonly_symbol = @utf8.decode(
    fused_runtime_inspection_decode_hex(
      fused_runtime_inspection_value(text, "readonly_symbol_hex"),
    ),
  ) catch {
    _ => raise DeviceGreedyQualificationRejected
  }
  let evidence = "schema=lunaflux-fused-production-runtime-inspection.v1\n" +
    "runtime_sha256=" +
    expected_sha256 +
    "\nqkv_module_sha256=" +
    qkv.module_digest().as_hex() +
    "\nqkv_source_sha256=" +
    qkv.source_digest().as_hex() +
    "\nqkv_symbol=" +
    qkv.function_symbol() +
    "\nreadonly_module_sha256=" +
    fused_runtime_inspection_value(text, "readonly_module_sha256") +
    "\nreadonly_source_sha256=" +
    fused_runtime_inspection_value(text, "readonly_source_sha256") +
    "\nreadonly_symbol=" +
    readonly_symbol +
    "\nresidual_module_sha256=" +
    residual.module_digest().as_hex() +
    "\nresidual_source_sha256=" +
    residual.source_digest().as_hex() +
    "\nresidual_symbol=" +
    residual.function_symbol() +
    "\nnormal_descriptor_admission=pass\nqualification_only=true\n" +
    "promotion_authority=absent\n"
  println(
    evidence +
    "inspection_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
    "\n",
  )
}
