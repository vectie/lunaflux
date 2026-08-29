///|
fn fused_production_fallback_role_name(
  role : @luna_fused_artifact_admission.FusedStandaloneFallbackRole,
) -> String {
  match role {
    QkvProjectionFallback => "qkv"
    PositionedRopeFallback => "positioned-rope"
    PagedKvWriteFallback => "paged-kv-write"
    ResidualAddFallback => "residual"
    RmsNormFallback => "rmsnorm"
  }
}

///|
fn fused_production_parse_fallback_line(
  line : String,
  expected : @luna_fused_artifact_admission.FusedStandaloneFallbackRole,
) -> @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity raise FusedProductionManifestError {
  let fields = line.split(",").to_array()
  if fields.length() != 4 ||
    fields[0] != "fallback=" + fused_production_fallback_role_name(expected) ||
    !fields[1].has_prefix("operation=") ||
    !fields[2].has_prefix("source_sha256=") ||
    !fields[3].has_prefix("recipe_sha256=") {
    raise InvalidFusedProductionManifest(Fallback)
  }
  let operation_text = fields[1]["operation=".length():].to_owned()
  let operation : Int = @strconv.from_str(operation_text) catch {
    _ => raise InvalidFusedProductionManifest(Fallback)
  }
  if operation < 0 || operation.to_string() != operation_text {
    raise InvalidFusedProductionManifest(Fallback)
  }
  let source = @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    fields[2]["source_sha256=".length():].to_owned(),
  ) catch {
    _ => raise InvalidFusedProductionManifest(Fallback)
  }
  let recipe = @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    fields[3]["recipe_sha256=".length():].to_owned(),
  ) catch {
    _ => raise InvalidFusedProductionManifest(Fallback)
  }
  @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity::new(
    role=expected,
    operation~,
    source~,
    recipe~,
  ) catch {
    _ => raise InvalidFusedProductionManifest(Fallback)
  }
}

///|
fn fused_production_fallbacks(
  benchmark : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence,
  expected : ArrayView[
    @luna_fused_artifact_admission.FusedStandaloneFallbackRole,
  ],
) -> ReadOnlyArray[
  @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
] raise FusedProductionManifestError {
  let document = @utf8.decode(benchmark.canonical_bytes()) catch {
    _ => raise InvalidFusedProductionManifest(BenchmarkEvidence)
  }
  let found : Array[String] = []
  for line in document.split("\n") {
    if line.has_prefix("fallback=") {
      found.push(line.to_owned())
    }
  }
  if found.length() != expected.length() {
    raise MismatchedFusedProductionManifest(Fallback)
  }
  let result : Array[
    @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
  ] = []
  for index, role in expected {
    result.push(fused_production_parse_fallback_line(found[index], role))
  }
  ReadOnlyArray::from_array(result)
}

///|
fn fused_production_require_fallback_match(
  retained : ReadOnlyArray[
    @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
  ],
  benchmark : ReadOnlyArray[
    @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
  ],
) -> Unit raise FusedProductionManifestError {
  if retained != benchmark {
    raise MismatchedFusedProductionManifest(Fallback)
  }
}

///|
fn fused_production_project_physical_fallbacks(
  physical : ReadOnlyArray[FusedStandaloneFallbackIdentity],
  roles : ArrayView[@luna_fused_artifact_admission.FusedStandaloneFallbackRole],
) -> ReadOnlyArray[
  @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
] raise FusedProductionManifestError {
  if physical.length() != roles.length() {
    raise MismatchedFusedProductionManifest(Fallback)
  }
  let projected : Array[
    @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
  ] = []
  for index, role in roles {
    let fallback = physical[index]
    projected.push(
      @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity::new(
        role~,
        operation=fallback.operation,
        source=fallback.source,
        recipe=fallback.recipe,
      ) catch {
        _ => raise InvalidFusedProductionManifest(Fallback)
      },
    )
  }
  ReadOnlyArray::from_array(projected)
}
