///|
/// Offline resource-policy observations are scoped more narrowly than device
/// capability: exact shape, numeric/layout contract and compilation program
/// revision. Source/artifact digests are matched against emitted candidates by
/// the generic selector, never interpreted as a model-specific policy.
pub fn parse_projection_resource_records(
  bytes : Bytes,
  scope~ : @resource_policy.Scope,
) -> ReadOnlyArray[@resource_policy.Observation] raise ProjectionTuningError {
  guard bytes.length() <= 1048576 else { raise InvalidSnapshot }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidSnapshot }
  guard text.split("\n").to_array()
    is ["luna-projection-resources-v1", scope_line, _, .. lines, ""] &&
    scope_line.split("\t").to_array()
    is ["scope", device, toolchain, shape, program] else {
    raise InvalidSnapshot
  }
  let actual = @resource_policy.Scope::new(
    device=device.to_owned(),
    toolchain=toolchain.to_owned(),
    shape=shape.to_owned(),
    program=program.to_owned(),
  )
  guard actual == Some(scope) else { raise ScopeMismatch }
  ignore(parse_projection_resource_budget(bytes))
  guard !lines.is_empty() && lines.length() <= 256 else {
    raise InvalidSnapshot
  }
  let records : Array[@resource_policy.Observation] = []
  let identities : Array[(String, String)] = []
  for line in lines {
    guard line.split("\t").to_array()
      is ["record", policy, artifact, latency, samples, registers, residency] &&
      fold_digest(artifact) else {
      raise InvalidSnapshot
    }
    let identity = (policy.to_owned(), artifact.to_owned())
    guard !identities.contains(identity) else { raise InvalidSnapshot }
    guard @resource_policy.Observation::new(
        scope,
        policy_id=identity.0,
        artifact=identity.1,
        latency_ns=decimal(latency),
        samples=integer(samples),
        registers_per_thread=integer(registers),
        resident_groups=integer(residency),
      )
      is Some(record) else {
      raise InvalidSnapshot
    }
    records.push(record)
    identities.push(identity)
  }
  ReadOnlyArray::from_array(records)
}

///|
/// The measured target envelope is intersected with backend launch limits by
/// the caller. Reading a larger envelope cannot grant unsupported resources.
pub fn parse_projection_resource_budget(
  bytes : Bytes,
) -> @resource_policy.ResourceBudget raise ProjectionTuningError {
  guard bytes.length() <= 1048576 else { raise InvalidSnapshot }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidSnapshot }
  guard text.split("\n").to_array()
    is ["luna-projection-resources-v1", _, budget, .., ""] &&
    budget.split("\t").to_array()
    is ["budget", threads, shared, registers, resident] else {
    raise InvalidSnapshot
  }
  guard @resource_policy.ResourceBudget::new(
      max_threads=integer(threads),
      max_shared_bytes=decimal(shared),
      max_registers_per_thread=Some(integer(registers)),
      min_resident_groups=Some(integer(resident)),
    )
    is Some(value) else {
    raise InvalidSnapshot
  }
  value
}
