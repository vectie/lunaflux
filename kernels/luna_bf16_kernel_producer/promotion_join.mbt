///|
fn validate_promotion_release(
  candidates : LunaBf16CandidateSet,
  release : LunaBf16KernelReleaseInput,
) -> String raise LunaBf16PromotionError {
  let catalog = release.catalog()
  let contracts = release.contracts()
  let bundle = release.bundle()
  let operations = bundle.operations()
  if catalog.model_identity() != candidates.model_identity() ||
    contracts.model_identity() != candidates.model_identity() ||
    catalog.device_target() != candidates.target() ||
    contracts.device_target() != candidates.target() ||
    contracts.profiles().length() != 1 ||
    contracts.profiles()[0] != candidates.profile() ||
    contracts.contracts().length() != candidates.operation_count() ||
    operations.length() != candidates.operation_count() ||
    bundle.execution_manifest().operation_count() !=
    candidates.operation_count() {
    raise LunaBf16PromotionRejected(ReleaseAuthority)
  }
  let compiler = candidates.compiler_policy().identity()
  let mut driver_identity = ""
  for index, candidate in candidates.candidates() {
    let operation = operations[index]
    let contract = contracts.contracts()[index]
    if candidate.operation_id().as_int() != index ||
      operation.operation_id() != candidate.operation_id() ||
      contract.operation_id() != candidate.operation_id() ||
      operation.device_target() != candidates.target() ||
      operation.source_digest_sha256() != candidate.source_sha256() ||
      operation.recipe_digest_sha256() != candidate.recipe_sha256() ||
      operation.toolchain_digest_sha256() != compiler.toolchain_digest_sha256() {
      raise LunaBf16PromotionRejected(LaunchContract)
    }
    if index == 0 {
      driver_identity = operation.driver_identity_sha256()
    } else if operation.driver_identity_sha256() != driver_identity {
      raise LunaBf16PromotionRejected(Compiler)
    }
  }
  if driver_identity.is_empty() {
    raise LunaBf16PromotionRejected(Compiler)
  }
  driver_identity
}

///|
/// Joins structurally admitted profiler evidence to the exact locally produced
/// candidate/release/launch authority. External approval and physical replay
/// remain mandatory before deployment promotion.
pub fn promote_luna_bf16_kernel_release(
  candidates : LunaBf16CandidateSet,
  release : LunaBf16KernelReleaseInput,
  profile : @luna_profile_priority.LunaPagedKernelProfileCapture,
  evidence : LunaBf16ProfilerEvidence,
  limits : LunaBf16PromotionLimits,
) -> LunaBf16OfflinePromotion raise LunaBf16PromotionError {
  if evidence.identity != candidates.model_identity() ||
    evidence.target != candidates.target() ||
    evidence.profile != candidates.profile() ||
    evidence.candidate_set_digest != candidates.candidate_set_sha256() ||
    evidence.compiler_digest !=
    promotion_compiler_digest(candidates.compiler_policy()) {
    raise LunaBf16PromotionRejected(CandidateSet)
  }
  if profile.identity() != candidates.model_identity() ||
    profile.target() != candidates.target() ||
    profile.contract().kernel_profile() != candidates.profile() ||
    evidence.profile_capture_digest != profile.digest() {
    raise LunaBf16PromotionRejected(ProfileCapture)
  }
  let driver_identity = validate_promotion_release(candidates, release)
  let contract_bytes = encode_launch_contracts(
    release.contracts(),
    limits.maximum_canonical_bytes,
  )
  let contract_digest = LunaBf16PromotionDigest(sha256_bytes(contract_bytes))
  let manifest_digest = release.bundle().execution_manifest().digest_sha256()
  let output = StringBuilder()
  let written = Ref(0)
  for
    value in [
      "lunaflux.bf16-offline-promotion.v1",
      candidates.model_identity().content().as_hex(),
      candidates.model_identity().plan().as_hex(),
      candidates.candidate_set_sha256(),
      promotion_compiler_digest(candidates.compiler_policy()).as_hex(),
      driver_identity,
      profile.digest().as_hex(),
      evidence.digest().as_hex(),
      contract_digest.as_hex(),
      manifest_digest,
      candidates.operation_count().to_string(),
    ] {
    promotion_append(output, written, value, limits.maximum_canonical_bytes)
  }
  let canonical = @utf8.encode(output.to_string())
  {
    digest: LunaBf16PromotionDigest(sha256_bytes(canonical)),
    profiler_evidence_digest: evidence.digest(),
    candidate_set_digest: candidates.candidate_set_sha256(),
    release_manifest_digest: manifest_digest,
    launch_contract_digest: contract_digest,
    operation_count: candidates.operation_count(),
  }
}
