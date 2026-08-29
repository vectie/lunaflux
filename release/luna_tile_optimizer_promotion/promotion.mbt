///|
fn luna_tile_promotion_record_canonical(
  path : LunaTileOptimizerPath,
  deployment : LunaTilePromotionDeploymentIdentity,
  promotion_epoch : UInt64,
  candidate : LunaTileOptimizerDigest,
  source : LunaTileOptimizerDigest,
  fallback : LunaTileOptimizerDigest,
  artifact : LunaTilePromotionArtifactInput,
  physical : LunaTileOptimizerDigest,
  benchmark : LunaTileOptimizerDigest,
  grid_x : Int,
  block_x : Int,
  shared_memory_bytes : Int64,
) -> Bytes raise LunaTileOptimizerPromotionError {
  let canonical = @utf8.encode(
    "schema=lunaflux-lunatile-optimizer-promotion.v1\n" +
    "path=" +
    path.stable_name() +
    "\n" +
    "deployment_sha256=" +
    deployment.as_hex() +
    "\n" +
    "promotion_epoch=" +
    promotion_epoch.to_string() +
    "\n" +
    "target=sm_120\n" +
    "candidate_sha256=" +
    candidate.as_hex() +
    "\n" +
    "source_sha256=" +
    source.as_hex() +
    "\n" +
    "fallback_sha256=" +
    fallback.as_hex() +
    "\n" +
    "module_sha256=" +
    artifact.module_digest.as_hex() +
    "\n" +
    "entry_point=" +
    artifact.entry_point +
    "\n" +
    "launch=" +
    grid_x.to_string() +
    ",1,1;" +
    block_x.to_string() +
    ",1,1\n" +
    "shared_memory_bytes=" +
    shared_memory_bytes.to_string() +
    "\n" +
    "physical_evidence_sha256=" +
    physical.as_hex() +
    "\n" +
    "paired_benchmark_sha256=" +
    benchmark.as_hex() +
    "\n" +
    "exact_fallback_required=true\nexternal_approval_required=true\n" +
    "manifest_bindable=true\nstartup_selection=exact-or-baseline\n",
  )
  if canonical.length() > LUNA_TILE_PROMOTION_MAX_CANONICAL_BYTES {
    raise LunaTileOptimizerPromotionInvalid(CanonicalBytes)
  }
  canonical
}

///|
fn luna_tile_make_promotion_record(
  path : LunaTileOptimizerPath,
  deployment : LunaTilePromotionDeploymentIdentity,
  promotion_epoch : UInt64,
  candidate : LunaTileOptimizerDigest,
  source : LunaTileOptimizerDigest,
  fallback : LunaTileOptimizerDigest,
  artifact : LunaTilePromotionArtifactInput,
  physical : LunaTileOptimizerDigest,
  benchmark : AdmittedLunaTilePairedBenchmarkWin,
  grid_x : Int,
  block_x : Int,
  shared_memory_bytes : Int64,
) -> LunaTilePromotionApprovalRecord raise LunaTileOptimizerPromotionError {
  if promotion_epoch == 0UL ||
    benchmark.path != path ||
    benchmark.candidate != candidate ||
    benchmark.fallback != fallback ||
    benchmark.physical != physical {
    raise LunaTileOptimizerPromotionMismatch(Benchmark)
  }
  let canonical = luna_tile_promotion_record_canonical(
    path,
    deployment,
    promotion_epoch,
    candidate,
    source,
    fallback,
    artifact,
    physical,
    benchmark.digest,
    grid_x,
    block_x,
    shared_memory_bytes,
  )
  {
    path,
    deployment,
    promotion_epoch,
    candidate,
    source,
    fallback,
    module_digest: artifact.module_digest,
    entry_point: artifact.entry_point,
    physical,
    benchmark: benchmark.digest,
    canonical,
    digest: luna_tile_promotion_digest(canonical),
  }
}

///|
pub fn prepare_luna_tile_simt_promotion_record(
  candidate : @luna_tile_ir.LunaTileParallelCandidate,
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  physical : AdmittedLunaTileSimtPhysicalEvidence,
  benchmark : AdmittedLunaTilePairedBenchmarkWin,
  artifact : LunaTilePromotionArtifactInput,
  deployment~ : LunaTilePromotionDeploymentIdentity,
  promotion_epoch~ : UInt64,
) -> LunaTilePromotionApprovalRecord raise LunaTileOptimizerPromotionError {
  if physical.manifest_bindable() ||
    physical.promotion_authorized() ||
    candidate.digest() != physical.candidate ||
    candidate.serial_oracle_digest() != serial.digest() ||
    serial.digest() != physical.serial ||
    artifact.module_digest != physical.module_digest ||
    artifact.entry_point != candidate.entry_point() {
    raise LunaTileOptimizerPromotionMismatch(PhysicalEvidence)
  }
  let mapping = candidate.mapping()
  luna_tile_make_promotion_record(
    LunaTileParallelSimtV1,
    deployment,
    promotion_epoch,
    LunaTileOptimizerDigest(candidate.digest().as_hex()),
    LunaTileOptimizerDigest(candidate.source_digest().as_hex()),
    LunaTileOptimizerDigest(serial.digest().as_hex()),
    artifact,
    physical.digest,
    benchmark,
    mapping.block_count(),
    mapping.block_threads(),
    candidate.shared_plan().total_shared_memory_bytes(),
  )
}

///|
pub fn prepare_luna_tile_tensor_core_promotion_record(
  candidate : @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Candidate,
  parallel : @luna_tile_ir.LunaTileParallelCandidate,
  physical : @luna_tensor_core_physical_evidence.AdmittedLunaTensorCorePhysicalEvidence,
  benchmark : AdmittedLunaTilePairedBenchmarkWin,
  artifact : LunaTilePromotionArtifactInput,
  deployment~ : LunaTilePromotionDeploymentIdentity,
  promotion_epoch~ : UInt64,
) -> LunaTilePromotionApprovalRecord raise LunaTileOptimizerPromotionError {
  let module_digest = LunaTileOptimizerDigest(physical.cubin_digest().as_hex())
  if physical.manifest_bindable() ||
    physical.promotion_authorized() ||
    candidate.digest().as_hex() != physical.candidate_digest() ||
    candidate.parallel_plan_digest() != parallel.digest() ||
    parallel.digest() != physical.parallel_fallback_digest() ||
    artifact.module_digest != module_digest ||
    artifact.entry_point != candidate.entry_point() ||
    candidate.tile_count() <= 0 {
    raise LunaTileOptimizerPromotionMismatch(PhysicalEvidence)
  }
  luna_tile_make_promotion_record(
    LunaTileTensorCoreBf16Mma16x16x16V1,
    deployment,
    promotion_epoch,
    LunaTileOptimizerDigest(candidate.digest().as_hex()),
    LunaTileOptimizerDigest(candidate.source_digest().as_hex()),
    LunaTileOptimizerDigest(parallel.digest().as_hex()),
    artifact,
    LunaTileOptimizerDigest(physical.digest().as_hex()),
    benchmark,
    candidate.tile_count(),
    32,
    0L,
  )
}

///|
/// Consumes only the existing opaque authenticated approval capability. This
/// package never constructs verifier keys, receipts, or approval values.
pub fn admit_external_luna_tile_optimizer_promotion(
  record : LunaTilePromotionApprovalRecord,
  envelope : @luna_capability_manifest.LunaDetachedSignatureEnvelope,
  approval : @luna_capability_manifest.LunaAuthenticatedExternalApproval,
) -> ApprovedLunaTileOptimizerCatalogEntry raise LunaTileOptimizerPromotionError {
  let admitted = @luna_capability_manifest.admit_luna_external_approved_record(
    record.canonical,
    envelope,
    approval,
    maximum_canonical_bytes=LUNA_TILE_PROMOTION_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise LunaTileOptimizerPromotionMismatch(ExternalApproval)
  }
  if admitted.record_digest().as_hex() != record.digest.as_hex() {
    raise LunaTileOptimizerPromotionMismatch(ExternalApproval)
  }
  { record, approval: admitted }
}

///|
fn luna_tile_require_startup_subject(
  record : LunaTilePromotionApprovalRecord,
  deployment : LunaTilePromotionDeploymentIdentity,
  promotion_epoch : UInt64,
  candidate : LunaTileOptimizerDigest,
  fallback : LunaTileOptimizerDigest,
) -> Unit raise LunaTileOptimizerPromotionError {
  if record.deployment != deployment {
    raise LunaTileOptimizerPromotionMismatch(Deployment)
  }
  if record.promotion_epoch != promotion_epoch {
    raise LunaTileOptimizerPromotionMismatch(Replay)
  }
  if record.candidate != candidate {
    raise LunaTileOptimizerPromotionMismatch(Candidate)
  }
  if record.fallback != fallback {
    raise LunaTileOptimizerPromotionMismatch(Fallback)
  }
}

///|
/// Startup-only selection. Absence deterministically retains the baseline;
/// any present but stale or substituted approval fails closed.
pub fn select_luna_tile_optimizer_at_startup(
  approved : ApprovedLunaTileOptimizerCatalogEntry?,
  deployment~ : LunaTilePromotionDeploymentIdentity,
  promotion_epoch~ : UInt64,
  candidate~ : LunaTileOptimizerDigest,
  fallback~ : LunaTileOptimizerDigest,
) -> LunaTileStartupSelection raise LunaTileOptimizerPromotionError {
  match approved {
    None => LunaTileBaselineSelected
    Some(entry) => {
      luna_tile_require_startup_subject(
        entry.record,
        deployment,
        promotion_epoch,
        candidate,
        fallback,
      )
      LunaTileApprovedOptimizerSelected(entry)
    }
  }
}
