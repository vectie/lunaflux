///|
fn fused_production_test_fallback(
  role : @luna_fused_artifact_admission.FusedStandaloneFallbackRole,
  operation? : Int = 2,
  source? : String = "1",
  recipe? : String = "2",
) -> @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity raise {
  @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity::new(
    role~,
    operation~,
    source=@luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
      source.repeat(64),
    ),
    recipe=@luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
      recipe.repeat(64),
    ),
  )
}

///|
test "typed fallback authority rejects role operation source and recipe substitution" {
  let exact : ReadOnlyArray[
    @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
  ] = ReadOnlyArray::from_array([
    fused_production_test_fallback(QkvProjectionFallback),
  ])
  let substitutions : Array[
    ReadOnlyArray[
      @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
    ],
  ] = [
    ReadOnlyArray::from_array([
      fused_production_test_fallback(PositionedRopeFallback),
    ]),
    ReadOnlyArray::from_array([
      fused_production_test_fallback(QkvProjectionFallback, operation=3),
    ]),
    ReadOnlyArray::from_array([
      fused_production_test_fallback(QkvProjectionFallback, source="3"),
    ]),
    ReadOnlyArray::from_array([
      fused_production_test_fallback(QkvProjectionFallback, recipe="3"),
    ]),
  ]
  for substitution in substitutions {
    try fused_production_require_fallback_match(exact, substitution) catch {
      MismatchedFusedProductionManifest(Fallback) => ()
      _ => fail("unexpected typed fallback substitution rejection")
    } noraise {
      _ => fail("substituted typed fallback reached production manifest")
    }
  }
}
