///|
fn paged_v4_contract_for(
  launches : @launch_contract.PagedV4LaunchContractSet,
  profile : @launch_contract.KernelProfileId,
  operation : @plan.OperationId,
) -> @launch_contract.PagedV4AotLaunchContract? {
  launches.contract(profile, operation)
}

///|
/// Admits exact catalog-v4 full-graph FP8 claims. Unlike the legacy paged-v3
/// entry point, v4 also authenticates numeric execution digests and canonical
/// weight/scale operand pairs before this manifest is published.
pub fn admit_paged_v4_fp8_e4m3_w8a8(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  resolved : @catalog.ResolvedKernelCatalog,
  launches : @launch_contract.PagedV4LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  claims : ArrayView[Fp8KernelManifestClaim],
  limits : Fp8KernelManifestLimits,
) -> AdmittedFp8KernelArtifacts raise Fp8KernelManifestError {
  if plan.kv_execution() != PagedKeyValue {
    raise InvalidFp8KernelManifest(ExecutionScope)
  }
  if launches.catalog_version() != @catalog.CatalogVersion::v4() ||
    launches.model_identity() != plan.identity() ||
    launches.numeric_schema_digest() != plan.numeric_binding().digest() ||
    launches.device_target() != resolved.device_target() ||
    launches.profiles().is_empty() {
    raise InvalidFp8KernelManifest(ModelIdentity)
  }
  let expected = validate_base(
    plan,
    device_numeric,
    launches.device_target(),
    resolved,
    artifacts,
    limits,
  )
  if launches.profiles().length() > limits.max_entries / expected.length() {
    raise InvalidFp8KernelManifest(Limit)
  }
  let expected_count = launches.profiles().length() * expected.length()
  if claims.length() != expected_count {
    raise InvalidFp8KernelManifest(ClaimCount)
  }
  let entries : Array[Fp8KernelManifestEntry] = []
  let mut claim_index = 0
  for profile in launches.profiles() {
    for operation in expected {
      guard paged_v4_contract_for(launches, profile.id(), operation.operation)
        is Some(contract) else {
        raise InvalidFp8KernelManifest(ClaimEntryPoint)
      }
      if contract.operation_execution_digest() != operation.execution {
        raise InvalidFp8KernelManifest(ClaimExecution)
      }
      entries.push(
        validate_claim(
          claims[claim_index],
          profile.id(),
          operation,
          contract.entry_point(),
          artifacts,
        ),
      )
      claim_index += 1
    }
  }
  publish_admission(
    PagedFullGraphV1,
    plan,
    device_numeric,
    launches.device_target(),
    launches.catalog_version(),
    entries,
    artifacts,
    limits,
  )
}
