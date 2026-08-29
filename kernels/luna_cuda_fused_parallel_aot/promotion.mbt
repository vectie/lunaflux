///|
fn fused_promotion_record(
  family : FusedParallelFamily,
  candidate_digest : FusedParallelDigest,
  source_digest : FusedParallelDigest,
  recipe_digest : FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  evidence : FusedParallelPromotionEvidence,
) -> Bytes {
  let family_name = match family {
    QkvPositionedRopePagedKvWrite => "qkv-positioned-rope-paged-kvwrite"
    ResidualRmsNorm => "residual-rmsnorm"
  }
  @utf8.encode(
    "schema=lunaflux-fused-parallel-candidate-promotion-binding.v1\n" +
    "family=" +
    family_name +
    "\ncandidate_sha256=" +
    candidate_digest.as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nrecipe_sha256=" +
    recipe_digest.as_hex() +
    "\ntoolchain_sha256=" +
    compiler.identity().toolchain_digest_sha256() +
    "\nraw_evidence_sha256=" +
    evidence.digest().as_hex() +
    "\nstatic_resource_audit_sha256=" +
    evidence.static_resource_audit().as_hex() +
    "\ntrial_count=" +
    evidence.trial_count().to_string() +
    "\nevidence_schema=lunaflux-fused-parallel-raw-evidence.v1\n" +
    "physical_performance_authority=external\nperformance_claim=none\n" +
    "release_binding=candidate-only\nmanifest_bindable=false\n",
  )
}

///|
fn fused_bind_promotion(
  family : FusedParallelFamily,
  candidate_digest : FusedParallelDigest,
  source_digest : FusedParallelDigest,
  recipe_digest : FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  evidence : FusedParallelPromotionEvidence,
  limits : FusedParallelAotLimits,
) -> FusedParallelPromotionBinding raise FusedParallelAotError {
  if evidence.candidate_digest() != candidate_digest {
    raise FusedParallelMismatch(Evidence)
  }
  let record = fused_promotion_record(
    family, candidate_digest, source_digest, recipe_digest, compiler, evidence,
  )
  if record.length() > limits.maximum_promotion_bytes {
    raise FusedParallelLimitExceeded(Evidence)
  }
  {
    family,
    candidate_digest,
    source_digest,
    recipe_digest,
    evidence,
    record,
    digest: fused_parallel_digest(record),
  }
}

///|
pub fn bind_fused_qkv_rope_kv_promotion_evidence(
  candidate : FusedQkvRopeKvWriteCudaCandidate,
  evidence : FusedParallelPromotionEvidence,
  limits : FusedParallelAotLimits,
) -> FusedParallelPromotionBinding raise FusedParallelAotError {
  fused_bind_promotion(
    QkvPositionedRopePagedKvWrite,
    candidate.candidate_digest,
    candidate.source_digest,
    candidate.recipe_digest,
    candidate.compiler,
    evidence,
    limits,
  )
}

///|
pub fn bind_fused_residual_rmsnorm_promotion_evidence(
  candidate : FusedResidualRmsNormCudaCandidate,
  evidence : FusedParallelPromotionEvidence,
  limits : FusedParallelAotLimits,
) -> FusedParallelPromotionBinding raise FusedParallelAotError {
  fused_bind_promotion(
    ResidualRmsNorm,
    candidate.candidate_digest,
    candidate.source_digest,
    candidate.recipe_digest,
    candidate.compiler,
    evidence,
    limits,
  )
}

///|
pub fn FusedParallelPromotionBinding::family(
  self : FusedParallelPromotionBinding,
) -> FusedParallelFamily {
  self.family
}

///|
pub fn FusedParallelPromotionBinding::candidate_digest(
  self : FusedParallelPromotionBinding,
) -> FusedParallelDigest {
  self.candidate_digest
}

///|
pub fn FusedParallelPromotionBinding::source_digest(
  self : FusedParallelPromotionBinding,
) -> FusedParallelDigest {
  self.source_digest
}

///|
pub fn FusedParallelPromotionBinding::recipe_digest(
  self : FusedParallelPromotionBinding,
) -> FusedParallelDigest {
  self.recipe_digest
}

///|
pub fn FusedParallelPromotionBinding::evidence(
  self : FusedParallelPromotionBinding,
) -> FusedParallelPromotionEvidence {
  self.evidence
}

///|
pub fn FusedParallelPromotionBinding::record_bytes(
  self : FusedParallelPromotionBinding,
) -> Bytes {
  self.record
}

///|
pub fn FusedParallelPromotionBinding::digest(
  self : FusedParallelPromotionBinding,
) -> FusedParallelDigest {
  self.digest
}

///|
pub fn FusedParallelPromotionBinding::manifest_bindable(
  _self : FusedParallelPromotionBinding,
) -> Bool {
  false
}
