///|
fn valid_launch(launch : @launch_contract.AotLaunchDimensions) -> Bool {
  launch.grid_x() > 0 &&
  launch.grid_y() > 0 &&
  launch.grid_z() > 0 &&
  launch.block_x() > 0 &&
  launch.block_y() > 0 &&
  launch.block_z() > 0 &&
  launch.shared_memory_bytes() >= 0
}

///|
pub fn admit(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  target : @catalog.DeviceTarget,
  claims : ArrayView[GlmHybridArtifactClaimInput],
  limits : GlmHybridArtifactAdmissionLimits,
) -> GlmHybridArtifactAdmission raise GlmHybridArtifactAdmissionError {
  let expected = requirements.requirements()
  if expected.length() > limits.maximum_claims ||
    claims.length() != expected.length() {
    raise ClaimCountMismatch(expected=expected.length(), actual=claims.length())
  }
  let slots : FixedArray[GlmHybridArtifactClaimInput?] = FixedArray::make(
    expected.length(),
    None,
  )
  for claim in claims {
    if claim.ordinal < 0 || claim.ordinal >= slots.length() {
      raise MissingClaim(ordinal=claim.ordinal)
    }
    if slots[claim.ordinal] is Some(_) {
      raise DuplicateClaim(ordinal=claim.ordinal)
    }
    slots[claim.ordinal] = Some(claim)
  }
  let admitted : Array[GlmHybridAdmittedArtifact] = []
  let entry_points : Array[@catalog.AotKernelEntryPoint] = []
  let function_symbols : Array[String] = []
  let mut total_bytes = 0L
  let mut all_qualified = true
  for ordinal, requirement in expected {
    guard slots[ordinal] is Some(claim) else { raise MissingClaim(ordinal~) }
    if claim.source_digest != requirements.digest() {
      raise SourceDigestMismatch(ordinal~)
    }
    if claim.source_identity != requirements.source_identity() {
      raise SourceIdentityMismatch(ordinal~)
    }
    if claim.target != target {
      raise TargetMismatch(ordinal~)
    }
    if claim.capability != requirement.capability() {
      raise CapabilityMismatch(ordinal~)
    }
    if claim.operation != requirement.operation() {
      raise GeometryMismatch(ordinal~)
    }
    if claim.module_bytes.length() <= 0 ||
      claim.module_bytes.length() > limits.maximum_module_bytes {
      raise ArtifactDigestMismatch(ordinal~)
    }
    total_bytes += Int64::from_int(claim.module_bytes.length())
    if total_bytes > limits.maximum_total_module_bytes {
      raise ArtifactDigestMismatch(ordinal~)
    }
    if claim.entry_point.family().module_digest() != claim.module_digest {
      raise EntryPointModuleMismatch(ordinal~)
    }
    if entry_points.contains(claim.entry_point) {
      raise DuplicateEntryPoint(ordinal~)
    }
    if !valid_function_symbol(claim.function_symbol) {
      raise InvalidFunctionSymbol(ordinal~)
    }
    if function_symbols.contains(claim.function_symbol) {
      raise InvalidFunctionSymbol(ordinal~)
    }
    if !valid_launch(claim.launch) {
      raise InvalidLaunch(ordinal~)
    }
    if !exact_supported_launch_abi(requirements, claim) {
      raise InvalidLaunchOperands(ordinal~)
    }
    entry_points.push(claim.entry_point)
    function_symbols.push(claim.function_symbol)
    if claim.declared_qualification is None {
      all_qualified = false
    }
    admitted.push({
      ordinal,
      operation: claim.operation,
      capability: claim.capability,
      module_digest: claim.module_digest,
      module_byte_length: claim.module_bytes.length(),
      entry_point: claim.entry_point,
      function_symbol: claim.function_symbol,
      launch: claim.launch,
      operands: claim.operands,
      declared_qualification: claim.declared_qualification,
    })
  }
  let status = if all_qualified {
    NonRunnableWithDeclaredQualificationDigests
  } else {
    NonRunnableUnqualified
  }
  let digest = admission_digest(
    requirements,
    target,
    status,
    admitted,
    limits.maximum_canonical_bytes,
  )
  {
    source_digest: requirements.digest(),
    source_identity: requirements.source_identity(),
    target,
    status,
    artifacts: ReadOnlyArray::from_array(admitted),
    digest,
  }
}
