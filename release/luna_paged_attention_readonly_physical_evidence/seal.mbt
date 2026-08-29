///|
priv struct ReadonlyPhysicalManifestEntry {
  digest : PagedAttentionReadOnlyPhysicalEvidenceDigest
  path : String
}

///|
fn readonly_physical_outer_seal(
  result : Bytes,
  manifest : Bytes,
  outer : Bytes,
  expected : PagedAttentionReadOnlyPhysicalEvidenceDigest,
) -> (
  PagedAttentionReadOnlyPhysicalEvidenceDigest,
  PagedAttentionReadOnlyPhysicalEvidenceDigest,
) raise PagedAttentionReadOnlyPhysicalEvidenceError {
  let manifest_digest = readonly_physical_digest(manifest)
  let result_digest = readonly_physical_digest(result)
  let canonical = @utf8.encode(
    manifest_digest.as_hex() +
    "  FILES.sha256\n" +
    result_digest.as_hex() +
    "  RESULT.txt\n",
  )
  if outer != canonical || readonly_physical_digest(outer) != expected {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Seal)
  }
  (manifest_digest, result_digest)
}

///|
fn readonly_physical_manifest_entries(
  bytes : Bytes,
  maximum : Int,
) -> Array[ReadonlyPhysicalManifestEntry] raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if bytes.length() == 0 || bytes.length() > maximum {
    raise PagedAttentionReadOnlyPhysicalEvidenceLimitExceeded
  }
  let document = @utf8.decode(bytes) catch {
    _ => raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Seal)
  }
  let lines = document.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() < 2 || lines[lines.length() - 1] != "" {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Seal)
  }
  let entries : Array[ReadonlyPhysicalManifestEntry] = []
  let mut previous = ""
  for index in 0..<(lines.length() - 1) {
    let line = lines[index]
    if line.length() < 67 || line[64] != ' ' || line[65] != ' ' {
      raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Seal)
    }
    let path = line[66:].to_owned()
    if path.length() == 0 ||
      path.length() > 255 ||
      path.has_prefix("/") ||
      path.contains("\\") ||
      path.contains("../") ||
      path.contains("//") ||
      path == "FILES.sha256" ||
      path == "RESULT.txt" ||
      path == "OUTER_SEAL.sha256" ||
      (index > 0 && String::compare(previous, path) >= 0) {
      raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Seal)
    }
    entries.push({
      digest: readonly_physical_read_digest(line[0:64].to_owned()),
      path,
    })
    previous = path
  }
  entries
}

///|
fn readonly_physical_manifest_require(
  entries : Array[ReadonlyPhysicalManifestEntry],
  path : String,
  expected : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  rule : PagedAttentionReadOnlyPhysicalEvidenceRule,
) -> Unit raise PagedAttentionReadOnlyPhysicalEvidenceError {
  for entry in entries {
    if entry.path == path {
      if entry.digest != expected {
        raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(rule)
      }
      return
    }
  }
  raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Seal)
}

///|
fn readonly_physical_require_pair(
  entries : Array[ReadonlyPhysicalManifestEntry],
  before : String,
  after : String,
  expected : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  rule : PagedAttentionReadOnlyPhysicalEvidenceRule,
) -> Unit raise PagedAttentionReadOnlyPhysicalEvidenceError {
  readonly_physical_manifest_require(entries, before, expected, rule)
  readonly_physical_manifest_require(entries, after, expected, rule)
}
