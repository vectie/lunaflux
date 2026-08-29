///|
fn fused_production_target_name(target : @catalog.DeviceTarget) -> String {
  "sm_" +
  target.compute_major().to_string() +
  target.compute_minor().to_string()
}

///|
fn fused_production_append_fallbacks(
  output : StringBuilder,
  prefix : String,
  fallbacks : ReadOnlyArray[
    @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
  ],
) -> Unit {
  output.write_string(
    prefix + "fallback_count=" + fallbacks.length().to_string() + "\n",
  )
  for index, fallback in fallbacks {
    output.write_string(
      prefix +
      "fallback_" +
      index.to_string() +
      "=" +
      fused_production_fallback_role_name(fallback.role()) +
      ",operation=" +
      fallback.operation_index().to_string() +
      ",source_sha256=" +
      fallback.source_digest().as_hex() +
      ",recipe_sha256=" +
      fallback.recipe_digest().as_hex() +
      "\n",
    )
  }
}

///|
fn fused_production_append_family(
  output : StringBuilder,
  prefix : String,
  family : ApprovedFusedProductionFamily,
) -> Unit {
  let compiled = family.compiled
  let launch = compiled.launch_dimensions()
  output.write_string(
    prefix + "candidate_sha256=" + compiled.candidate_digest().as_hex() + "\n",
  )
  output.write_string(
    prefix + "source_sha256=" + compiled.source_digest().as_hex() + "\n",
  )
  output.write_string(
    prefix + "recipe_sha256=" + compiled.recipe_digest().as_hex() + "\n",
  )
  output.write_string(
    prefix + "toolchain_sha256=" + compiled.toolchain_digest_sha256() + "\n",
  )
  output.write_string(
    prefix +
    "driver_identity_sha256=" +
    compiled.driver_identity_sha256() +
    "\n",
  )
  output.write_string(
    prefix + "target=" + fused_production_target_name(compiled.target()) + "\n",
  )
  output.write_string(
    prefix +
    "launch=" +
    launch.grid_x().to_string() +
    ":" +
    launch.grid_y().to_string() +
    ":" +
    launch.grid_z().to_string() +
    "," +
    launch.block_x().to_string() +
    ":" +
    launch.block_y().to_string() +
    ":" +
    launch.block_z().to_string() +
    ",shared=" +
    launch.shared_memory_bytes().to_string() +
    "\n",
  )
  output.write_string(
    prefix + "function_symbol=" + compiled.function_symbol() + "\n",
  )
  output.write_string(
    prefix + "typed_abi=" + compiled.kernel_abi().canonical_name() + "\n",
  )
  output.write_string(
    prefix +
    "dispatch_canary_per_token=" +
    family.dispatch_canary_per_token.to_string() +
    "\n",
  )
  output.write_string(
    prefix + "cubin_sha256=" + compiled.module_digest().as_hex() + "\n",
  )
  output.write_string(
    prefix +
    "compile_receipt_sha256=" +
    compiled.compile_receipt_digest().as_hex() +
    "\n",
  )
  output.write_string(
    prefix +
    "compiled_binding_sha256=" +
    compiled.binding_digest().as_hex() +
    "\n",
  )
  output.write_string(
    prefix + "benchmark_sha256=" + family.benchmark.as_hex() + "\n",
  )
  fused_production_append_fallbacks(output, prefix, family.fallbacks)
}

///|
fn fused_production_canonical(
  approved : ApprovedFusedManifestArtifacts,
  qkv : ApprovedFusedProductionFamily,
  residual : ApprovedFusedProductionFamily,
) -> Bytes {
  let record = approved.record
  let physical = record.physical
  let output = StringBuilder()
  output.write_string("schema=lunaflux-fused-production-manifest.v1\n")
  output.write_string("family_count=2\n")
  output.write_string(
    "target=" + fused_production_target_name(qkv.compiled.target()) + "\n",
  )
  output.write_string(
    "physical_evidence_sha256=" + physical.digest().as_hex() + "\n",
  )
  output.write_string(
    "physical_artifact_seal_sha256=" +
    physical.artifact_directory_seal_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "physical_evidence_seal_sha256=" +
    physical.evidence_directory_seal_digest().as_hex() +
    "\n",
  )
  output.write_string("approval_record_sha256=" + record.digest.as_hex() + "\n")
  output.write_string(
    "approved_source_sha256=" +
    approved.approved_source_identity().as_hex() +
    "\n",
  )
  output.write_string(
    "approved_key_sha256=" + approved.approved_key_identity().as_hex() + "\n",
  )
  output.write_string(
    "detached_signature_sha256=" +
    approved.detached_signature_digest().as_hex() +
    "\n",
  )
  fused_production_append_family(output, "qkv_", qkv)
  fused_production_append_family(output, "residual_", residual)
  output.write_string("standalone_fallback_required=true\n")
  output.write_string(
    "runtime_dispatch_authority=approved-exact-manifest-only\n",
  )
  @utf8.encode(output.to_string())
}

///|
fn fused_production_validate_retained(
  approved : ApprovedFusedManifestArtifacts,
) -> Unit raise FusedProductionManifestError {
  if !approved.manifest_bindable() || !approved.promotion_authorized() {
    raise MismatchedFusedProductionManifest(Authority)
  }
  let record = approved.record
  let expected_record = fused_manifest_approval_canonical(
    record.physical,
    record.qkv_compiled,
    record.residual_compiled,
    record.qkv_benchmark,
    record.residual_benchmark,
  )
  if expected_record != record.canonical ||
    fused_physical_digest(record.canonical) != record.digest ||
    approved.approval.record_digest().as_hex() != record.digest.as_hex() {
    raise MismatchedFusedProductionManifest(Approval)
  }
  if fused_physical_digest(record.physical.canonical_bytes()) !=
    record.physical.digest() {
    raise MismatchedFusedProductionManifest(PhysicalEvidence)
  }
  if fused_physical_digest(record.qkv_benchmark.canonical_bytes()) !=
    record.qkv_benchmark.digest() ||
    fused_physical_digest(record.residual_benchmark.canonical_bytes()) !=
    record.residual_benchmark.digest() {
    raise MismatchedFusedProductionManifest(BenchmarkEvidence)
  }
}

///|
fn fused_production_family(
  compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  benchmark : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence,
  physical_candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  physical_source : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  physical_recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  physical_module : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  physical_receipt : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  physical_fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity],
  expected_fallback_roles : ArrayView[
    @luna_fused_artifact_admission.FusedStandaloneFallbackRole,
  ],
  physical_dispatch_canary_per_token : UInt,
) -> ApprovedFusedProductionFamily raise FusedProductionManifestError {
  if compiled.manifest_bindable() ||
    compiled.candidate_digest() != physical_candidate ||
    compiled.source_digest() != physical_source ||
    compiled.recipe_digest() != physical_recipe ||
    compiled.module_digest() != physical_module ||
    compiled.compile_receipt_digest() != physical_receipt {
    raise MismatchedFusedProductionManifest(Artifact)
  }
  if benchmark.manifest_bindable() ||
    benchmark.promotion_authority() ||
    benchmark.family() != compiled.family() ||
    benchmark.candidate_digest() != compiled.candidate_digest() ||
    benchmark.observation_count() <= 0 {
    raise MismatchedFusedProductionManifest(BenchmarkEvidence)
  }
  let typed_fallbacks = fused_production_project_physical_fallbacks(
    physical_fallbacks, expected_fallback_roles,
  )
  let parsed = fused_production_fallbacks(benchmark, expected_fallback_roles)
  fused_production_require_fallback_match(typed_fallbacks, parsed)
  match compiled.artifact_purpose() {
    QualificationDiagnostics =>
      if physical_dispatch_canary_per_token == 0U ||
        compiled.dispatch_canary_per_token() !=
        physical_dispatch_canary_per_token {
        raise MismatchedFusedProductionManifest(KernelAbi)
      }
    ProductionFastPath =>
      if physical_dispatch_canary_per_token != 0U ||
        compiled.dispatch_canary_per_token() != 0U {
        raise MismatchedFusedProductionManifest(KernelAbi)
      }
  }
  {
    compiled,
    benchmark: benchmark.digest(),
    dispatch_canary_per_token: physical_dispatch_canary_per_token,
    fallbacks: typed_fallbacks,
  }
}

///|
/// Projects only the exact externally approved fused pair into a production
/// manifest. No other approval fragment or candidate type is accepted here.
pub fn admit_approved_fused_production_manifest(
  approved : ApprovedFusedManifestArtifacts,
) -> ApprovedFusedProductionManifest raise FusedProductionManifestError {
  fused_production_validate_retained(approved)
  let record = approved.record
  // The current approval record binds the canary-bearing qualification
  // artifact. It must never be reinterpreted as production runtime authority.
  // A successor record must bind the distinct canary-free compiled artifact
  // and its own physical result before this projection can succeed.
  if record.residual_compiled.artifact_purpose() != ProductionFastPath ||
    record.residual_compiled.kernel_abi() != ResidualRmsNormProductionAbiV2 ||
    record.residual_compiled.dispatch_canary_per_token() != 0U {
    raise MismatchedFusedProductionManifest(Artifact)
  }
  let physical = record.physical
  let qkv = fused_production_family(
    record.qkv_compiled,
    record.qkv_benchmark,
    physical.qkv.candidate_digest,
    physical.qkv.source_digest,
    physical.qkv.recipe_digest,
    physical.qkv.module_digest,
    physical.qkv.receipt_digest,
    physical.qkv.fallbacks,
    [QkvProjectionFallback, PositionedRopeFallback, PagedKvWriteFallback],
    physical.qkv.dispatch_canary_per_token,
  )
  let residual = fused_production_family(
    record.residual_compiled,
    record.residual_benchmark,
    physical.residual_norm.candidate_digest,
    physical.residual_norm.source_digest,
    physical.residual_norm.recipe_digest,
    physical.residual_norm.module_digest,
    physical.residual_norm.receipt_digest,
    physical.residual_norm.fallbacks,
    [ResidualAddFallback, RmsNormFallback],
    physical.residual_norm.dispatch_canary_per_token,
  )
  if qkv.compiled.family() != QkvPositionedRopePagedKvWrite ||
    residual.compiled.family() != ResidualRmsNorm {
    raise MismatchedFusedProductionManifest(Family)
  }
  let target = qkv.compiled.target()
  if residual.compiled.target() != target ||
    qkv.compiled.toolchain_digest_sha256() !=
    residual.compiled.toolchain_digest_sha256() ||
    qkv.compiled.driver_identity_sha256() !=
    residual.compiled.driver_identity_sha256() {
    raise MismatchedFusedProductionManifest(Compiler)
  }
  if target.compute_major() != 12 ||
    target.compute_minor() != 0 ||
    !target.supports_bf16() {
    raise MismatchedFusedProductionManifest(Target)
  }
  let canonical = fused_production_canonical(approved, qkv, residual)
  {
    qkv,
    residual,
    target,
    physical: physical.digest(),
    physical_artifact_seal: physical.artifact_directory_seal_digest(),
    physical_evidence_seal: physical.evidence_directory_seal_digest(),
    approval_record: record.digest,
    approved_source: approved.approved_source_identity(),
    approved_key: approved.approved_key_identity(),
    detached_signature: approved.detached_signature_digest(),
    approval: approved.external_approval_record(),
    canonical,
    digest: fused_physical_digest(canonical),
  }
}

///|
/// Offline adapter into the kernel-owned runtime catalog identity. The kernel
/// admission package retains no dependency on this release package.
pub fn adapt_approved_fused_kernel_catalog(
  manifest : ApprovedFusedProductionManifest,
) -> @luna_fused_artifact_admission.ApprovedFusedKernelCatalog raise FusedProductionManifestError {
  let qkv = manifest.qkv
  let residual = manifest.residual
  @luna_fused_artifact_admission.admit_approved_fused_kernel_catalog(
    manifest.digest,
    manifest.approval_record,
    manifest.target,
    qkv.compiled,
    residual.compiled,
    qkv.benchmark,
    residual.benchmark,
    qkv.dispatch_canary_per_token,
    residual.dispatch_canary_per_token,
    qkv.fallbacks,
    residual.fallbacks,
    manifest.approval,
  ) catch {
    _ => raise MismatchedFusedProductionManifest(Artifact)
  }
}

///|
/// Offline deployment adapter for the two kernel-owned runtime identities.
/// Device-worker startup consumes only this catalog and bundle; it never sees
/// the physical evidence or release-manifest types used to construct them.
pub fn adapt_approved_fused_runtime_identities(
  manifest : ApprovedFusedProductionManifest,
) -> (
  @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  @luna_fused_artifact_admission.ApprovedFusedKernelBundle,
) raise FusedProductionManifestError {
  let catalog = adapt_approved_fused_kernel_catalog(manifest)
  let qkv = manifest.qkv
  let residual = manifest.residual
  let bundle = @luna_fused_artifact_admission.admit_approved_fused_kernel_bundle(
    manifest.digest,
    manifest.approval_record,
    manifest.target,
    qkv.compiled,
    residual.compiled,
    qkv.benchmark,
    residual.benchmark,
    qkv.dispatch_canary_per_token,
    residual.dispatch_canary_per_token,
    qkv.fallbacks,
    residual.fallbacks,
    manifest.approval,
  ) catch {
    _ => raise MismatchedFusedProductionManifest(Artifact)
  }
  (catalog, bundle)
}

///|
/// Offline composition adapter for post-qualification deployment. Diagnostic
/// or mixed ABI inputs produce `None`; the exact QKV-V2/readonly-V2 pair
/// produces the opaque kernel-owned runtime artifact aggregate.
pub fn adapt_fused_qkv_readonly_production_artifacts(
  manifest : ApprovedFusedProductionManifest,
  readonly_identity : @luna_paged_composite_artifact_admission.ApprovedPagedAttentionReadOnlyRuntimeIdentity,
  kv_write : @luna_paged_composite_artifact_admission.QualifiedPagedKvWriteFallbackIdentity,
) -> @luna_paged_composite_artifact_admission.ApprovedFusedQkvReadOnlyProductionArtifacts? raise FusedProductionManifestError {
  let (catalog, bundle) = adapt_approved_fused_runtime_identities(manifest)
  @luna_paged_composite_artifact_admission.select_fused_qkv_readonly_production_artifacts(
    catalog, bundle, readonly_identity, kv_write,
  ) catch {
    _ => raise MismatchedFusedProductionManifest(Artifact)
  }
}
