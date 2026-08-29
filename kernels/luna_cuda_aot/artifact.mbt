///|
fn luna_cuda_aot_evidence(
  lowering : LunaCudaAotLowering,
  artifact_digest : LunaCudaArtifactDigest,
  driver_identity_sha256 : String,
) -> Bytes {
  let compiler = lowering.compiler.identity()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-evidence-v1\n" +
    "source_sha256=" +
    lowering.source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    lowering.recipe_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    compiler.toolchain_digest_sha256() +
    "\n" +
    "driver_identity_sha256=" +
    driver_identity_sha256 +
    "\n" +
    "target=sm_" +
    lowering.target.compute_major().to_string() +
    lowering.target.compute_minor().to_string() +
    "\n" +
    "function_symbol=" +
    lowering.function_symbol +
    "\n" +
    "artifact_sha256=" +
    artifact_digest.as_hex() +
    "\n" +
    "first_build_sha256=" +
    artifact_digest.as_hex() +
    "\n" +
    "second_build_sha256=" +
    artifact_digest.as_hex() +
    "\n" +
    "deterministic=1\n",
  )
}

///|
/// Admits two independently produced offline CUBIN snapshots.
///
/// Admission checks the typed recipe receipt, exact toolchain identity, both
/// hashes, and byte-for-byte determinism. It never invokes a compiler.
pub fn admit_luna_cuda_aot_output(
  lowering : LunaCudaAotLowering,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaCudaOfflineCompileReceipt,
  limits : LunaCudaAotLimits,
) -> LunaCudaAotArtifact raise LunaCudaAotError {
  if first_module.length() == 0 || second_module.length() == 0 {
    raise Invalid(Artifact)
  }
  if first_module.length().to_int64() > limits.max_module_bytes ||
    second_module.length().to_int64() > limits.max_module_bytes {
    raise LimitExceeded(Artifact)
  }
  if receipt.source_digest != lowering.source_digest ||
    receipt.recipe_digest != lowering.recipe_digest {
    raise Mismatch(CompileReceipt)
  }
  if receipt.toolchain_digest_sha256 !=
    lowering.compiler.identity().toolchain_digest_sha256() {
    raise Mismatch(Toolchain)
  }
  let first_digest = LunaCudaArtifactDigest(luna_cuda_aot_sha256(first_module))
  let second_digest = LunaCudaArtifactDigest(
    luna_cuda_aot_sha256(second_module),
  )
  if first_digest != receipt.first_artifact_digest ||
    second_digest != receipt.second_artifact_digest {
    raise Mismatch(CompileReceipt)
  }
  if first_digest != second_digest || first_module != second_module {
    raise Mismatch(Determinism)
  }
  let evidence = luna_cuda_aot_evidence(
    lowering,
    first_digest,
    receipt.driver_identity_sha256,
  )
  if evidence.length() > limits.max_evidence_bytes {
    raise LimitExceeded(Evidence)
  }
  let module_digest = @catalog.AotArtifactDigest::from_sha256(
    first_digest.as_hex(),
  ) catch {
    _ => raise Invalid(Artifact)
  }
  {
    target: lowering.target,
    entry_point_id: lowering.entry_point_id,
    function_symbol: lowering.function_symbol,
    operand_byte_count: lowering.operand_byte_count,
    dimensions: lowering.dimensions,
    module_bytes: first_module,
    module_digest,
    source_digest: lowering.source_digest,
    recipe_digest: lowering.recipe_digest,
    driver_identity_sha256: receipt.driver_identity_sha256,
    evidence,
    evidence_digest: LunaCudaEvidenceDigest(luna_cuda_aot_sha256(evidence)),
  }
}

///|
pub fn LunaCudaAotArtifact::target(
  self : LunaCudaAotArtifact,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaCudaAotArtifact::module_digest(
  self : LunaCudaAotArtifact,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn LunaCudaAotArtifact::module_bytes(self : LunaCudaAotArtifact) -> Bytes {
  self.module_bytes
}

///|
/// Strict-relative path used by release assembly under the kernel root.
pub fn LunaCudaAotArtifact::module_relative_path(
  self : LunaCudaAotArtifact,
) -> String {
  "sha256/" + self.module_digest.sha256() + ".cubin"
}

///|
pub fn LunaCudaAotArtifact::function_symbol(
  self : LunaCudaAotArtifact,
) -> String {
  self.function_symbol
}

///|
pub fn LunaCudaAotArtifact::source_digest(
  self : LunaCudaAotArtifact,
) -> LunaCudaSourceDigest {
  self.source_digest
}

///|
pub fn LunaCudaAotArtifact::recipe_digest(
  self : LunaCudaAotArtifact,
) -> LunaCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn LunaCudaAotArtifact::driver_identity_sha256(
  self : LunaCudaAotArtifact,
) -> String {
  self.driver_identity_sha256
}

///|
pub fn LunaCudaAotArtifact::evidence_bytes(self : LunaCudaAotArtifact) -> Bytes {
  self.evidence
}

///|
pub fn LunaCudaAotArtifact::evidence_digest(
  self : LunaCudaAotArtifact,
) -> LunaCudaEvidenceDigest {
  self.evidence_digest
}

///|
