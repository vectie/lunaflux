///|
priv struct LunaTensorCoreEvidenceManifestEntry {
  digest : LunaTensorCorePhysicalEvidenceDigest
  path : String
}

///|
fn tensor_core_evidence_safe_path(path : String) -> Bool {
  path.length() > 0 &&
  path.length() <= 255 &&
  !path.has_prefix("/") &&
  !path.contains("\\") &&
  !path.contains("//") &&
  !path.contains("../") &&
  !path.has_suffix("/..") &&
  path != "." &&
  path != ".." &&
  path != "FILES.sha256" &&
  path != "RESULT.txt" &&
  path != "OUTER_SEAL.sha256"
}

///|
fn tensor_core_evidence_manifest_entries(
  bytes : Bytes,
  maximum_bytes : Int,
) -> Array[LunaTensorCoreEvidenceManifestEntry] raise LunaTensorCorePhysicalEvidenceError {
  if bytes.length() == 0 || bytes.length() > maximum_bytes {
    raise LunaTensorCorePhysicalEvidenceLimitExceeded
  }
  let document = @utf8.decode(bytes) catch {
    _ => raise LunaTensorCorePhysicalEvidenceInvalid(Seal)
  }
  let lines = document.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() < 2 || lines[lines.length() - 1] != "" {
    raise LunaTensorCorePhysicalEvidenceInvalid(Seal)
  }
  let entries : Array[LunaTensorCoreEvidenceManifestEntry] = []
  let mut previous = ""
  for index in 0..<(lines.length() - 1) {
    let line = lines[index]
    if line.length() < 67 || line[64] != ' ' || line[65] != ' ' {
      raise LunaTensorCorePhysicalEvidenceInvalid(Seal)
    }
    let digest = tensor_core_evidence_read_digest(line[0:64].to_owned())
    let path = line[66:].to_owned()
    if !tensor_core_evidence_safe_path(path) ||
      (index > 0 && String::compare(previous, path) >= 0) {
      raise LunaTensorCorePhysicalEvidenceInvalid(Seal)
    }
    entries.push({ digest, path })
    previous = path
  }
  entries
}

///|
fn tensor_core_manifest_find(
  entries : Array[LunaTensorCoreEvidenceManifestEntry],
  path : String,
) -> LunaTensorCorePhysicalEvidenceDigest raise LunaTensorCorePhysicalEvidenceError {
  for entry in entries {
    if entry.path == path {
      return entry.digest
    }
  }
  raise LunaTensorCorePhysicalEvidenceMismatch(Seal)
}

///|
fn tensor_core_manifest_require(
  entries : Array[LunaTensorCoreEvidenceManifestEntry],
  path : String,
  expected : LunaTensorCorePhysicalEvidenceDigest,
  rule : LunaTensorCorePhysicalEvidenceRule,
) -> Unit raise LunaTensorCorePhysicalEvidenceError {
  if tensor_core_manifest_find(entries, path) != expected {
    raise LunaTensorCorePhysicalEvidenceMismatch(rule)
  }
}

///|
fn tensor_core_manifest_require_present(
  entries : Array[LunaTensorCoreEvidenceManifestEntry],
  path : String,
) -> Unit raise LunaTensorCorePhysicalEvidenceError {
  ignore(tensor_core_manifest_find(entries, path))
}

///|
fn tensor_core_manifest_require_empty(
  entries : Array[LunaTensorCoreEvidenceManifestEntry],
  path : String,
) -> Unit raise LunaTensorCorePhysicalEvidenceError {
  let empty = LunaTensorCorePhysicalEvidenceDigest::from_sha256(
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  )
  tensor_core_manifest_require(entries, path, empty, Resource)
}

///|
fn tensor_core_require_critical_manifest(
  entries : Array[LunaTensorCoreEvidenceManifestEntry],
  serial_translation : LunaTensorCorePhysicalEvidenceDigest,
  candidate : LunaTensorCorePhysicalEvidenceDigest,
  serial_source : LunaTensorCorePhysicalEvidenceDigest,
  tensor_source : LunaTensorCorePhysicalEvidenceDigest,
  serial_cubin : LunaTensorCorePhysicalEvidenceDigest,
  tensor_cubin : LunaTensorCorePhysicalEvidenceDigest,
  driver_record : LunaTensorCorePhysicalEvidenceDigest,
  nvcc_version : LunaTensorCorePhysicalEvidenceDigest,
  ptxas_version : LunaTensorCorePhysicalEvidenceDigest,
  cuobjdump_version : LunaTensorCorePhysicalEvidenceDigest,
  nvdisasm_version : LunaTensorCorePhysicalEvidenceDigest,
  sanitizer_version : LunaTensorCorePhysicalEvidenceDigest,
  cuobjdump_sass : LunaTensorCorePhysicalEvidenceDigest,
  nvdisasm_sass : LunaTensorCorePhysicalEvidenceDigest,
  memcheck : LunaTensorCorePhysicalEvidenceDigest,
  racecheck : LunaTensorCorePhysicalEvidenceDigest,
  initcheck : LunaTensorCorePhysicalEvidenceDigest,
) -> Unit raise LunaTensorCorePhysicalEvidenceError {
  for
    requirement in [
      ("fixture/serial.canonical", serial_translation, Oracle),
      ("fixture/tensor_core.canonical", candidate, Candidate),
      ("fixture/serial.cu", serial_source, Oracle),
      ("fixture/tensor_core.cu", tensor_source, Candidate),
      ("builds/serial/first/kernel.cubin", serial_cubin, Oracle),
      ("builds/serial/second/kernel.cubin", serial_cubin, Oracle),
      ("builds/serial/first/kernel.cu", serial_source, Oracle),
      ("builds/serial/second/kernel.cu", serial_source, Oracle),
      ("builds/tensor_core/first/kernel.cubin", tensor_cubin, Candidate),
      ("builds/tensor_core/second/kernel.cubin", tensor_cubin, Candidate),
      ("builds/tensor_core/first/kernel.cu", tensor_source, Candidate),
      ("builds/tensor_core/second/kernel.cu", tensor_source, Candidate),
      ("builds/tensor_core/resource/kernel.cubin", tensor_cubin, Candidate),
      ("builds/tensor_core/resource/kernel.cu", tensor_source, Candidate),
      ("measurements/driver-before.v1", driver_record, Device),
      ("measurements/driver-after.v1", driver_record, Device),
      ("measurements/nvcc-version-before.stdout", nvcc_version, Compiler),
      ("measurements/nvcc-version-after.stdout", nvcc_version, Compiler),
      ("measurements/ptxas-version-before.stdout", ptxas_version, Compiler),
      ("measurements/ptxas-version-after.stdout", ptxas_version, Compiler),
      (
        "measurements/cuobjdump-version-before.stdout",
        cuobjdump_version,
        Compiler,
      ),
      (
        "measurements/cuobjdump-version-after.stdout",
        cuobjdump_version,
        Compiler,
      ),
      (
        "measurements/nvdisasm-version-before.stdout",
        nvdisasm_version,
        Compiler,
      ),
      ("measurements/nvdisasm-version-after.stdout", nvdisasm_version, Compiler),
      (
        "measurements/compute-sanitizer-version-before.stdout",
        sanitizer_version,
        Compiler,
      ),
      (
        "measurements/compute-sanitizer-version-after.stdout",
        sanitizer_version,
        Compiler,
      ),
      ("measurements/cuobjdump-sass.stdout", cuobjdump_sass, Sass),
      ("measurements/nvdisasm-sass.stdout", nvdisasm_sass, Sass),
      ("measurements/memcheck.log", memcheck, Sanitizer),
      ("measurements/racecheck.log", racecheck, Sanitizer),
      ("measurements/initcheck.log", initcheck, Sanitizer),
    ] {
    let (path, digest, rule) = requirement
    tensor_core_manifest_require(entries, path, digest, rule)
  }
  for
    path in [
      "fixture/fixture.v1", "measurements/device-identity.v1", "measurements/cuobjdump-resources.stdout",
      "measurements/memcheck-runtime.stdout", "measurements/racecheck-runtime.stdout",
      "measurements/initcheck-runtime.stdout", "measurements/moon-build.stdout",
      "measurements/export.stdout", "builds/serial/first/compiler.stdout", "builds/serial/first/compiler.stderr",
      "builds/serial/second/compiler.stdout", "builds/serial/second/compiler.stderr",
      "builds/tensor_core/first/compiler.stdout", "builds/tensor_core/first/compiler.stderr",
      "builds/tensor_core/second/compiler.stdout", "builds/tensor_core/second/compiler.stderr",
      "builds/tensor_core/resource/compiler.stdout", "builds/tensor_core/resource/ptxas-resource.stderr",
    ] {
    tensor_core_manifest_require_present(entries, path)
  }
  for
    path in [
      "measurements/nvcc-version-before.stderr", "measurements/nvcc-version-after.stderr",
      "measurements/ptxas-version-before.stderr", "measurements/ptxas-version-after.stderr",
      "measurements/driver-before.stderr", "measurements/driver-after.stderr", "measurements/cuobjdump-version-before.stderr",
      "measurements/cuobjdump-version-after.stderr", "measurements/nvdisasm-version-before.stderr",
      "measurements/nvdisasm-version-after.stderr", "measurements/compute-sanitizer-version-before.stderr",
      "measurements/compute-sanitizer-version-after.stderr", "measurements/cuobjdump-sass.stderr",
      "measurements/nvdisasm-sass.stderr", "measurements/cuobjdump-resources.stderr",
      "measurements/memcheck-runtime.stderr", "measurements/racecheck-runtime.stderr",
      "measurements/initcheck-runtime.stderr", "measurements/export.stderr",
    ] {
    tensor_core_manifest_require_empty(entries, path)
  }
}

///|
fn tensor_core_require_outer_seal(
  result : Bytes,
  files_manifest : Bytes,
  outer_seal : Bytes,
  expected_outer_seal_digest : LunaTensorCorePhysicalEvidenceDigest,
) -> (
  LunaTensorCorePhysicalEvidenceDigest,
  LunaTensorCorePhysicalEvidenceDigest,
) raise LunaTensorCorePhysicalEvidenceError {
  let manifest_digest = tensor_core_evidence_sha256(files_manifest)
  let result_digest = tensor_core_evidence_sha256(result)
  let expected = @utf8.encode(
    manifest_digest.as_hex() +
    "  FILES.sha256\n" +
    result_digest.as_hex() +
    "  RESULT.txt\n",
  )
  if outer_seal != expected ||
    tensor_core_evidence_sha256(outer_seal) != expected_outer_seal_digest {
    raise LunaTensorCorePhysicalEvidenceMismatch(Seal)
  }
  (manifest_digest, result_digest)
}
