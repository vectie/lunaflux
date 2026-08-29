///|
fn readonly_physical_policy_value(
  lines : Array[String],
  index : Int,
  key : String,
) -> String raise PagedAttentionReadOnlyPhysicalEvidenceError {
  let prefix = key + "="
  if index < 0 ||
    index >= lines.length() ||
    !lines[index].has_prefix(prefix) ||
    lines[index].length() == prefix.length() {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Toolchain)
  }
  lines[index][prefix.length():].to_owned()
}

///|
fn readonly_physical_lower_hex(character : UInt16) -> Bool {
  (character >= '0' && character <= '9') ||
  (character >= 'a' && character <= 'f')
}

///|
fn readonly_physical_valid_device_uuid(value : String) -> Bool {
  if value.length() != 40 ||
    value[0] != 'G' ||
    value[1] != 'P' ||
    value[2] != 'U' {
    return false
  }
  for index in 3..<value.length() {
    if index == 3 || index == 12 || index == 17 || index == 22 || index == 27 {
      if value[index] != '-' {
        return false
      }
    } else if !readonly_physical_lower_hex(value[index]) {
      return false
    }
  }
  true
}

///|
fn readonly_physical_valid_device_pci_bus_id(value : String) -> Bool {
  if value.length() != 16 {
    return false
  }
  for index in 0..<value.length() {
    if index == 8 || index == 11 {
      if value[index] != ':' {
        return false
      }
    } else if index == 14 {
      if value[index] != '.' {
        return false
      }
    } else if !readonly_physical_lower_hex(value[index]) {
      return false
    }
  }
  true
}

///|
pub fn admit_approved_paged_attention_readonly_physical_policy(
  canonical : Bytes,
  expected_digest : PagedAttentionReadOnlyPhysicalEvidenceDigest,
) -> ApprovedPagedAttentionReadOnlyPhysicalPolicy raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if canonical.length() == 0 ||
    canonical.length() > 16384 ||
    readonly_physical_digest(canonical) != expected_digest {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Toolchain)
  }
  let document = @utf8.decode(canonical) catch {
    _ => raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Toolchain)
  }
  let lines = document.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() != 17 || lines[16] != "" {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Toolchain)
  }
  if readonly_physical_policy_value(lines, 0, "schema") !=
    "lunaflux-paged-attention-readonly-approved-physical-policy.v1" ||
    readonly_physical_policy_value(lines, 1, "target") != "sm_120" ||
    readonly_physical_policy_value(lines, 2, "compiler_version") != "13.1.115" ||
    readonly_physical_policy_value(lines, 14, "device_compute") != "12.0" ||
    readonly_physical_policy_value(lines, 15, "policy_authority") !=
    "deployment-approved" {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Toolchain)
  }
  let device_uuid = readonly_physical_policy_value(lines, 10, "device_uuid")
  let device_pci_bus_id = readonly_physical_policy_value(
    lines, 11, "device_pci_bus_id",
  )
  if !readonly_physical_valid_device_uuid(device_uuid) ||
    !readonly_physical_valid_device_pci_bus_id(device_pci_bus_id) {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Target)
  }
  let name = readonly_physical_policy_value(lines, 12, "device_name_hex")
  if name.length() == 0 || name.length() > 512 || name.length() % 2 != 0 {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Target)
  }
  for character in name {
    if !(character >= '0' && character <= '9') &&
      !(character >= 'a' && character <= 'f') {
      raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Target)
    }
  }
  let memory_text = readonly_physical_policy_value(
    lines, 13, "device_total_memory_bytes",
  )
  let memory : Int64 = @strconv.from_str(memory_text) catch {
    _ => raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Target)
  }
  if memory <= 0L || memory.to_string() != memory_text {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Target)
  }
  {
    canonical,
    digest: expected_digest,
    nvcc: readonly_physical_read_digest(
      readonly_physical_policy_value(lines, 3, "nvcc_sha256"),
    ),
    ptxas: readonly_physical_read_digest(
      readonly_physical_policy_value(lines, 4, "ptxas_sha256"),
    ),
    cuobjdump: readonly_physical_read_digest(
      readonly_physical_policy_value(lines, 5, "cuobjdump_sha256"),
    ),
    nvdisasm: readonly_physical_read_digest(
      readonly_physical_policy_value(lines, 6, "nvdisasm_sha256"),
    ),
    sanitizer: readonly_physical_read_digest(
      readonly_physical_policy_value(lines, 7, "compute_sanitizer_sha256"),
    ),
    driver_identity: readonly_physical_read_digest(
      readonly_physical_policy_value(lines, 8, "driver_identity_sha256"),
    ),
    driver_record: readonly_physical_read_digest(
      readonly_physical_policy_value(lines, 9, "driver_record_sha256"),
    ),
    device_uuid,
    device_pci_bus_id,
    device_name_hex: name,
    device_total_memory_bytes: memory,
  }
}

///|
pub fn ApprovedPagedAttentionReadOnlyPhysicalPolicy::digest(
  self : ApprovedPagedAttentionReadOnlyPhysicalPolicy,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest {
  self.digest
}

///|
pub fn ApprovedPagedAttentionReadOnlyPhysicalPolicy::canonical_bytes(
  self : ApprovedPagedAttentionReadOnlyPhysicalPolicy,
) -> Bytes {
  self.canonical
}
