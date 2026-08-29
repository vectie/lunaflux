///|
/// Admits exact stateless FP8 launch claims and one already-admitted artifact
/// bundle. Every FP8 operation must be AOT; vendor/BF16 fallback is forbidden.
pub fn admit_stateless_fp8_e4m3_w8a8(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  resolved : @catalog.ResolvedKernelCatalog,
  launches : @launch_contract.LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  claims : ArrayView[Fp8KernelManifestClaim],
  limits : Fp8KernelManifestLimits,
) -> AdmittedFp8KernelArtifacts raise Fp8KernelManifestError {
  if plan.kv_execution() != StatelessFullContext {
    raise InvalidFp8KernelManifest(ExecutionScope)
  }
  if launches.model_identity() != plan.identity() ||
    launches.device_target() != resolved.device_target() ||
    launches.catalog_version() != resolved.catalog_version() ||
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
      guard launches.contract(profile.id(), operation.operation)
        is Some(contract) else {
        raise InvalidFp8KernelManifest(ClaimEntryPoint)
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
    StatelessFullContextV1,
    plan,
    device_numeric,
    launches.device_target(),
    launches.catalog_version(),
    entries,
    artifacts,
    limits,
  )
}

///|
fn paged_contract_for(
  launches : @launch_contract.PagedKvLaunchContractSet,
  profile : @launch_contract.KernelProfileId,
  operation : @plan.OperationId,
) -> @launch_contract.PagedKvAotLaunchContract? {
  for contract in launches.contracts() {
    if contract.profile().id() == profile &&
      contract.operation_id() == operation {
      return Some(contract)
    }
  }
  None
}

///|
/// Admits exact full-graph paged FP8 launch claims and one already-admitted
/// artifact bundle. The result is inert and is not physical execution proof.
pub fn admit_paged_fp8_e4m3_w8a8(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  resolved : @catalog.ResolvedKernelCatalog,
  launches : @launch_contract.PagedKvLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  claims : ArrayView[Fp8KernelManifestClaim],
  limits : Fp8KernelManifestLimits,
) -> AdmittedFp8KernelArtifacts raise Fp8KernelManifestError {
  if plan.kv_execution() != PagedKeyValue || launches.scope() != FullGraph {
    raise InvalidFp8KernelManifest(ExecutionScope)
  }
  if launches.model_identity() != plan.identity() ||
    launches.device_target() != resolved.device_target() ||
    launches.catalog_version() != resolved.catalog_version() ||
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
      guard paged_contract_for(launches, profile.id(), operation.operation)
        is Some(contract) else {
        raise InvalidFp8KernelManifest(ClaimEntryPoint)
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
