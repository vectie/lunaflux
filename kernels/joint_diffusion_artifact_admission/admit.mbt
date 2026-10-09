///|
fn launch_index(
  inputs : ArrayView[JointDiffusionLaunchEvidenceInput],
  ordinal : Int,
) -> Int? {
  for index, input in inputs {
    if input.requirement_ordinal == ordinal {
      return Some(index)
    }
  }
  None
}

///|
fn validate_launches(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  target : @catalog.DeviceTarget,
  inputs : ArrayView[JointDiffusionLaunchEvidenceInput],
  limits : JointDiffusionArtifactAdmissionLimits,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  let expected = requirements.requirements()
  if expected.length() > limits.maximum_requirements ||
    inputs.length() > limits.maximum_requirements {
    raise RequirementCountExceeded
  }
  for index, input in inputs {
    let ordinal = input.requirement_ordinal
    if input.requirements_digest != requirements.digest() {
      raise RequirementDigestMismatch(ordinal)
    }
    if input.target != target {
      raise TargetMismatch(ordinal)
    }
    if ordinal < 0 || ordinal >= expected.length() {
      raise MissingLaunch(ordinal)
    }
    for earlier in 0..<index {
      if inputs[earlier].requirement_ordinal == ordinal {
        raise DuplicateLaunch(ordinal)
      }
    }
    for earlier in 0..<index {
      if inputs[earlier].entry_point == input.entry_point {
        raise DuplicateEntryPoint(input.entry_point)
      }
    }
    if input.operation != expected[ordinal].operation() {
      raise WrongOperation(ordinal)
    }
    if input.capability != expected[ordinal].capability() {
      raise WrongCapability(ordinal)
    }
    if !operands_are_exact(
        input.operation,
        input.operands,
        limits.maximum_operands_per_launch,
      ) {
      raise InvalidLaunchOperands(ordinal)
    }
  }
  for ordinal in 0..<expected.length() {
    if launch_index(inputs, ordinal) is None {
      raise MissingLaunch(ordinal)
    }
  }
}

///|
fn required_entry_points(
  launches : ArrayView[JointDiffusionLaunchEvidenceInput],
  requirement_count : Int,
) -> Array[@catalog.AotKernelEntryPoint] {
  let required : Array[@catalog.AotKernelEntryPoint] = []
  for ordinal in 0..<requirement_count {
    guard launch_index(launches, ordinal) is Some(index) else {
      abort("validated joint diffusion launch missing")
    }
    let launch = launches[index]
    if !required.contains(launch.entry_point) {
      required.push(launch.entry_point)
    }
  }
  required
}

///|
fn entry_input_index(
  inputs : ArrayView[@artifact.KernelEntryPointInput],
  identity : @catalog.AotKernelEntryPoint,
) -> Int? {
  for index, input in inputs {
    if input.identity() == identity {
      return Some(index)
    }
  }
  None
}

///|
fn validate_entry_points(
  inputs : ArrayView[@artifact.KernelEntryPointInput],
  required : ArrayView[@catalog.AotKernelEntryPoint],
  limits : JointDiffusionArtifactAdmissionLimits,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  if inputs.length() > limits.maximum_entry_points {
    raise RequirementCountExceeded
  }
  let artifact_limits = try! @artifact.KernelArtifactLimits::new(
    max_modules=limits.maximum_modules,
    max_entry_points=limits.maximum_entry_points,
    max_module_bytes=limits.maximum_module_bytes,
    max_total_module_bytes=limits.maximum_total_module_bytes,
    max_function_symbol_bytes=limits.maximum_symbol_bytes,
  )
  @artifact.validate_entry_point_symbols(inputs, artifact_limits) catch {
    error => raise ArtifactSymbolFailure(error)
  }
  for index, input in inputs {
    let identity = input.identity()
    for earlier in 0..<index {
      if inputs[earlier].identity() == identity {
        raise DuplicateEntryPoint(identity)
      }
    }
    if !required.contains(identity) {
      raise UnreferencedEntryPoint(identity)
    }
  }
  for identity in required {
    if entry_input_index(inputs, identity) is None {
      raise MissingEntryPoint(identity)
    }
  }
}

///|
fn required_module_digests(
  entries : ArrayView[@catalog.AotKernelEntryPoint],
) -> Array[@catalog.AotArtifactDigest] {
  let required : Array[@catalog.AotArtifactDigest] = []
  for entry in entries {
    let digest = entry.family().module_digest()
    if !required.contains(digest) {
      required.push(digest)
    }
  }
  required
}

///|
fn module_input_index(
  inputs : ArrayView[JointDiffusionModuleInput],
  digest : @catalog.AotArtifactDigest,
) -> Int? {
  for index, input in inputs {
    if input.digest == digest {
      return Some(index)
    }
  }
  None
}

///|
fn digest_index(
  digests : ArrayView[@catalog.AotArtifactDigest],
  digest : @catalog.AotArtifactDigest,
) -> Int? {
  for index, candidate in digests {
    if candidate == digest {
      return Some(index)
    }
  }
  None
}

///|
fn validate_modules(
  inputs : ArrayView[JointDiffusionModuleInput],
  required : ArrayView[@catalog.AotArtifactDigest],
  limits : JointDiffusionArtifactAdmissionLimits,
) -> Int64 raise JointDiffusionArtifactAdmissionError {
  if inputs.length() > limits.maximum_modules {
    raise RequirementCountExceeded
  }
  let mut total = 0L
  for index, input in inputs {
    let digest = input.digest
    let byte_count = input.module_bytes.length().to_int64()
    if byte_count == 0L {
      raise EmptyModule(digest)
    }
    if byte_count > limits.maximum_module_bytes {
      raise ModuleTooLarge(digest)
    }
    if total > limits.maximum_total_module_bytes - byte_count {
      raise TotalModuleBytesExceeded
    }
    total += byte_count
    for earlier in 0..<index {
      if inputs[earlier].digest == digest {
        raise DuplicateModule(digest)
      }
    }
    if !required.contains(digest) {
      raise UnreferencedModule(digest)
    }
    let exact = @crypto.bytes_to_hex_string(@crypto.sha256(input.module_bytes))
    if exact != digest.sha256() {
      raise ModuleDigestMismatch(digest)
    }
  }
  for digest in required {
    if module_input_index(inputs, digest) is None {
      raise MissingModule(digest)
    }
  }
  total
}

///|
/// Joins exact diffusion requirements to inert content-addressed modules,
/// stable entry-point symbols, target evidence, and launch metadata.
pub fn admit(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  target~ : @catalog.DeviceTarget,
  module_inputs~ : ArrayView[JointDiffusionModuleInput],
  entry_point_inputs~ : ArrayView[@artifact.KernelEntryPointInput],
  launch_inputs~ : ArrayView[JointDiffusionLaunchEvidenceInput],
  limits~ : JointDiffusionArtifactAdmissionLimits,
) -> JointDiffusionArtifactAdmission raise JointDiffusionArtifactAdmissionError {
  validate_launches(requirements, target, launch_inputs, limits)
  let required_entries = required_entry_points(
    launch_inputs,
    requirements.requirements().length(),
  )
  validate_entry_points(entry_point_inputs, required_entries, limits)
  let required_digests = required_module_digests(required_entries)
  let total_module_bytes = validate_modules(
    module_inputs, required_digests, limits,
  )
  let modules : Array[JointDiffusionAdmittedModule] = []
  for digest in required_digests {
    guard module_input_index(module_inputs, digest) is Some(input_index) else {
      raise MissingModule(digest)
    }
    modules.push({
      digest,
      module_bytes: module_inputs[input_index].module_bytes,
    })
  }
  let entries : Array[JointDiffusionAdmittedEntryPoint] = []
  for identity in required_entries {
    guard entry_input_index(entry_point_inputs, identity) is Some(input_index) else {
      raise MissingEntryPoint(identity)
    }
    let module_digest = identity.family().module_digest()
    guard digest_index(required_digests, module_digest) is Some(module_index) else {
      raise MissingModule(module_digest)
    }
    entries.push({
      identity,
      module_index,
      function_symbol: entry_point_inputs[input_index].function_symbol(),
    })
  }
  let launches : Array[JointDiffusionAdmittedLaunch] = []
  for ordinal in 0..<requirements.requirements().length() {
    guard launch_index(launch_inputs, ordinal) is Some(input_index) else {
      raise MissingLaunch(ordinal)
    }
    let input = launch_inputs[input_index]
    launches.push({
      requirement_ordinal: ordinal,
      operation: input.operation,
      capability: input.capability,
      entry_point: input.entry_point,
      dimensions: input.dimensions,
      operands: input.operands,
    })
  }
  let source_identity = requirements.source_identity()
  let requirements_digest = requirements.digest()
  let digest = canonical_admission_digest(
    source_identity, requirements_digest, target, modules, entries, launches,
  )
  {
    schema_version: 1,
    source_identity,
    requirements_digest,
    target,
    status: InertUnqualified,
    digest,
    modules: ReadOnlyArray::from_array(modules),
    entry_points: ReadOnlyArray::from_array(entries),
    launches: ReadOnlyArray::from_array(launches),
    total_module_bytes,
  }
}
