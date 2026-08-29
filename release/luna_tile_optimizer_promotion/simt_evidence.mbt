///|
priv struct LunaTileSimtManifestEntry {
  digest : LunaTileOptimizerDigest
  path : String
}

///|
fn luna_tile_simt_manifest_entries(
  bytes : Bytes,
) -> Array[LunaTileSimtManifestEntry] raise LunaTileOptimizerPromotionError {
  if bytes.length() <= 0 || bytes.length() > 262144 {
    raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
  }
  let document = @utf8.decode(bytes) catch {
    _ => raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
  }
  let lines = document.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() < 2 || lines[lines.length() - 1] != "" {
    raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
  }
  let entries : Array[LunaTileSimtManifestEntry] = []
  for index in 0..<(lines.length() - 1) {
    let line = lines[index]
    if line.length() < 67 || line[64] != ' ' || line[65] != ' ' {
      raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
    }
    let path = line[66:].to_owned()
    if path.is_empty() ||
      path.length() > 255 ||
      path.has_prefix("/") ||
      path.contains("..") ||
      path.contains("//") ||
      path.contains("\\") {
      raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
    }
    for earlier in entries {
      if earlier.path == path {
        raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
      }
    }
    entries.push({
      digest: LunaTileOptimizerDigest::from_sha256(line[0:64].to_owned()),
      path,
    })
  }
  entries
}

///|
fn luna_tile_simt_manifest_require(
  entries : ArrayView[LunaTileSimtManifestEntry],
  path : String,
  expected : LunaTileOptimizerDigest,
) -> Unit raise LunaTileOptimizerPromotionError {
  for entry in entries {
    if entry.path == path {
      if entry.digest != expected {
        raise LunaTileOptimizerPromotionMismatch(PhysicalEvidence)
      }
      return
    }
  }
  raise LunaTileOptimizerPromotionMismatch(PhysicalEvidence)
}

///|
fn luna_tile_simt_result_fields(
  result : Bytes,
) -> Map[String, String] raise LunaTileOptimizerPromotionError {
  if result.length() <= 0 || result.length() > 4096 {
    raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
  }
  let text = @utf8.decode(result) catch {
    _ => raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
  }
  let lines = text.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() != 11 || lines[10] != "" {
    raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
  }
  let fields : Map[String, String] = Map([])
  for index in 0..<10 {
    let parts = lines[index].split("=").map(part => part.to_owned()).to_array()
    if parts.length() != 2 || parts[0].is_empty() || fields.contains(parts[0]) {
      raise LunaTileOptimizerPromotionInvalid(PhysicalEvidence)
    }
    fields[parts[0]] = parts[1]
  }
  fields
}

///|
fn luna_tile_simt_require_field(
  fields : Map[String, String],
  key : String,
  expected : String,
) -> Unit raise LunaTileOptimizerPromotionError {
  if fields.get(key) != Some(expected) {
    raise LunaTileOptimizerPromotionMismatch(PhysicalEvidence)
  }
}

///|
/// Admits the exact immutable output of the isolated SIMT physical campaign.
/// The files-manifest digest is supplied by the publication boundary; critical
/// candidate, source, deterministic CUBIN and result entries are re-bound here.
pub fn admit_luna_tile_simt_physical_evidence(
  candidate : @luna_tile_ir.LunaTileParallelCandidate,
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  result : Bytes,
  files_manifest : Bytes,
  expected_files_manifest_digest~ : LunaTileOptimizerDigest,
) -> AdmittedLunaTileSimtPhysicalEvidence raise LunaTileOptimizerPromotionError {
  if candidate.serial_oracle_digest() != serial.digest() ||
    candidate.program_digest() != serial.program_digest() ||
    candidate.compute_capability() != (12, 0) ||
    luna_tile_promotion_digest(files_manifest) != expected_files_manifest_digest {
    raise LunaTileOptimizerPromotionMismatch(PhysicalEvidence)
  }
  let fields = luna_tile_simt_result_fields(result)
  luna_tile_simt_require_field(
    fields, "outcome", "lunatile-parallel-physical-campaign-pass",
  )
  let fixture_digest = LunaTileOptimizerDigest::from_sha256(
    fields["fixture_sha256"],
  )
  luna_tile_simt_require_field(
    fields,
    "serial_source_sha256",
    serial.source_digest().as_hex(),
  )
  luna_tile_simt_require_field(
    fields,
    "parallel_source_sha256",
    candidate.source_digest().as_hex(),
  )
  luna_tile_simt_require_field(
    fields,
    "parallel_candidate_sha256",
    candidate.digest().as_hex(),
  )
  luna_tile_simt_require_field(fields, "physical_cuda_observed", "true")
  luna_tile_simt_require_field(fields, "manifest_bindable", "false")
  luna_tile_simt_require_field(fields, "promotion_authority", "absent")
  let serial_module = LunaTileOptimizerDigest::from_sha256(
    fields["serial_cubin_sha256"],
  )
  let module_digest = LunaTileOptimizerDigest::from_sha256(
    fields["parallel_cubin_sha256"],
  )
  if module_digest == serial_module {
    raise LunaTileOptimizerPromotionMismatch(PhysicalEvidence)
  }
  let entries = luna_tile_simt_manifest_entries(files_manifest)
  let result_digest = luna_tile_promotion_digest(result)
  let empty = LunaTileOptimizerDigest::from_sha256(
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  )
  for
    requirement in [
      ("CAMPAIGN_RESULT.txt", result_digest),
      ("fixture/fixture.v1", fixture_digest),
      (
        "fixture/serial.canonical",
        LunaTileOptimizerDigest(serial.digest().as_hex()),
      ),
      (
        "fixture/parallel.canonical",
        LunaTileOptimizerDigest(candidate.digest().as_hex()),
      ),
      (
        "fixture/serial.cu",
        LunaTileOptimizerDigest(serial.source_digest().as_hex()),
      ),
      (
        "fixture/parallel.cu",
        LunaTileOptimizerDigest(candidate.source_digest().as_hex()),
      ),
      ("builds/serial/first/kernel.cubin", serial_module),
      ("builds/serial/second/kernel.cubin", serial_module),
      ("builds/parallel/first/kernel.cubin", module_digest),
      ("builds/parallel/second/kernel.cubin", module_digest),
      ("measurements/runtime.stderr", empty),
      ("measurements/racecheck.stderr", empty),
    ] {
    luna_tile_simt_manifest_require(entries, requirement.0, requirement.1)
  }
  {
    candidate: candidate.digest(),
    source: candidate.source_digest(),
    serial: serial.digest(),
    module_digest,
    serial_module_digest: serial_module,
    sealed_result: expected_files_manifest_digest,
    canonical: result,
    digest: result_digest,
  }
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::digest(
  self : AdmittedLunaTileSimtPhysicalEvidence,
) -> LunaTileOptimizerDigest {
  self.digest
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::module_digest(
  self : AdmittedLunaTileSimtPhysicalEvidence,
) -> LunaTileOptimizerDigest {
  self.module_digest
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::candidate_digest(
  self : AdmittedLunaTileSimtPhysicalEvidence,
) -> @luna_tile_ir.LunaTileParallelDigest {
  self.candidate
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::serial_fallback_digest(
  self : AdmittedLunaTileSimtPhysicalEvidence,
) -> @luna_tile_ir.LunaTileCudaTranslationDigest {
  self.serial
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::manifest_bindable(
  _self : AdmittedLunaTileSimtPhysicalEvidence,
) -> Bool {
  false
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::promotion_authorized(
  _self : AdmittedLunaTileSimtPhysicalEvidence,
) -> Bool {
  false
}
