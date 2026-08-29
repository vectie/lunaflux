///|
pub fn export_positioned_paged_kv_write_qualification(
  candidate : PagedKvWriteCudaCandidate,
) -> PagedKvWriteQualificationExport raise PagedKvWriteAotError {
  if PagedKvWriteDigest(paged_kv_write_sha256(candidate.source)) !=
    candidate.source_digest ||
    PagedKvWriteDigest(paged_kv_write_sha256(candidate.recipe)) !=
    candidate.recipe_digest {
    raise Mismatch(Digest)
  }
  let canonical = @utf8.encode(
    "schema=lunaflux-positioned-paged-kv-write-source-export.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest.as_hex() +
    "\n" +
    "source_sha256=" +
    candidate.source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\n" +
    "symbol=" +
    candidate.symbol +
    "\nabi=" +
    candidate.abi.canonical_name() +
    "\n" +
    "qualification_only=true\ncompiled_artifact=absent\n" +
    "physical_evidence=absent\nmanifest_bindable=false\n" +
    "promotion_authority=false\n",
  )
  {
    candidate: candidate.candidate_digest,
    source: candidate.source,
    source_digest: candidate.source_digest,
    recipe: candidate.recipe,
    recipe_digest: candidate.recipe_digest,
    canonical,
    digest: PagedKvWriteDigest(paged_kv_write_sha256(canonical)),
  }
}

///|
pub fn PagedKvWriteQualificationExport::candidate_digest(
  self : PagedKvWriteQualificationExport,
) -> PagedKvWriteDigest {
  self.candidate
}

///|
pub fn PagedKvWriteQualificationExport::source_bytes(
  self : PagedKvWriteQualificationExport,
) -> Bytes {
  self.source
}

///|
pub fn PagedKvWriteQualificationExport::source_digest(
  self : PagedKvWriteQualificationExport,
) -> PagedKvWriteDigest {
  self.source_digest
}

///|
pub fn PagedKvWriteQualificationExport::recipe_bytes(
  self : PagedKvWriteQualificationExport,
) -> Bytes {
  self.recipe
}

///|
pub fn PagedKvWriteQualificationExport::recipe_digest(
  self : PagedKvWriteQualificationExport,
) -> PagedKvWriteDigest {
  self.recipe_digest
}

///|
pub fn PagedKvWriteQualificationExport::canonical_bytes(
  self : PagedKvWriteQualificationExport,
) -> Bytes {
  self.canonical
}

///|
pub fn PagedKvWriteQualificationExport::digest(
  self : PagedKvWriteQualificationExport,
) -> PagedKvWriteDigest {
  self.digest
}

///|
pub fn PagedKvWriteQualificationExport::manifest_bindable(
  _self : PagedKvWriteQualificationExport,
) -> Bool {
  false
}

///|
pub fn PagedKvWriteQualificationExport::physical_evidence(
  _self : PagedKvWriteQualificationExport,
) -> Bool {
  false
}

///|
pub fn PagedKvWriteQualificationExport::promotion_authority(
  _self : PagedKvWriteQualificationExport,
) -> Bool {
  false
}
