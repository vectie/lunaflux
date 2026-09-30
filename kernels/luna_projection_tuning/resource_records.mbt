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
    is [version, scope_line, _, .. all_lines, ""] &&
    (
      version == "luna-projection-resources-v1" ||
      version == "luna-projection-resources-v2" ||
      version == "luna-projection-resources-v3"
    ) &&
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
  let selected_scope = projection_resource_selection_scope(bytes, scope~)
  let lines = if version == "luna-projection-resources-v3" {
    guard all_lines.length() >= 4 else { raise InvalidSnapshot }
    all_lines[3:]
  } else if version == "luna-projection-resources-v2" {
    guard all_lines.length() >= 3 else { raise InvalidSnapshot }
    all_lines[2:]
  } else {
    all_lines
  }
  ignore(parse_projection_resource_budget(bytes))
  guard !lines.is_empty() && lines.length() <= 256 else {
    raise InvalidSnapshot
  }
  let records : Array[@resource_policy.Observation] = []
  let identities : Array[(String, String)] = []
  for line in lines {
    guard line.split("\t").to_array()
      is [
        "record",
        policy,
        artifact,
        latency,
        samples,
        registers,
        residency,
        .. machine,
      ] &&
      fold_digest(artifact) else {
      raise InvalidSnapshot
    }
    let identity = (policy.to_owned(), artifact.to_owned())
    guard !identities.contains(identity) else { raise InvalidSnapshot }
    let machine_work = match (version, machine) {
      ("luna-projection-resources-v3", [instructions, local_bytes, barriers]) =>
        @resource_policy.MachineWork::new(
          instructions=nonnegative(instructions),
          local_bytes=nonnegative(local_bytes),
          barriers=if barriers == "-" {
            None
          } else {
            Some(nonnegative(barriers))
          },
        )
      ("luna-projection-resources-v3", _) => raise InvalidSnapshot
      (_, []) => None
      _ => raise InvalidSnapshot
    }
    guard @resource_policy.Observation::new(
        selected_scope,
        policy_id=identity.0,
        artifact=identity.1,
        latency_ns=decimal(latency),
        samples=integer(samples),
        registers_per_thread=integer(registers),
        resident_groups=integer(residency),
        machine_work~,
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
  guard text.split("\n").to_array() is [version, _, budget, .., ""] &&
    (
      version == "luna-projection-resources-v1" ||
      version == "luna-projection-resources-v2" ||
      version == "luna-projection-resources-v3"
    ) &&
    budget.split("\t").to_array()
    is ["budget", threads, shared, registers, resident] else {
    raise InvalidSnapshot
  }
  let machine_limits = if version == "luna-projection-resources-v3" {
    guard text.split("\n").to_array() is [_, _, _, _, _, work, .., ""] &&
      work.split("\t").to_array()
      is ["machine-budget", instructions, local_bytes, barriers] else {
      raise InvalidSnapshot
    }
    @resource_policy.MachineWork::new(
      instructions=nonnegative(instructions),
      local_bytes=nonnegative(local_bytes),
      barriers=if barriers == "-" { None } else { Some(nonnegative(barriers)) },
    )
  } else {
    None
  }
  guard @resource_policy.ResourceBudget::new(
      max_threads=integer(threads),
      max_shared_bytes=decimal(shared),
      max_registers_per_thread=Some(integer(registers)),
      min_resident_groups=Some(integer(resident)),
      machine_limits~,
    )
    is Some(value) else {
    raise InvalidSnapshot
  }
  value
}

///|
/// V2 contains exactly one timing objective, including actual and captured
/// bucket geometry. UUID is retained as provenance, never confused with a
/// performance compatibility class. V1 remains readable as a legacy scope.
pub fn projection_resource_selection_scope(
  bytes : Bytes,
  scope~ : @resource_policy.Scope,
  max_query_tokens? : Int = 1048576,
  max_batch_rows? : Int = 1048576,
) -> @resource_policy.Scope raise ProjectionTuningError {
  guard bytes.length() <= 1048576 else { raise InvalidSnapshot }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidSnapshot }
  match text.split("\n").to_array() {
    ["luna-projection-resources-v1", .., ""] => scope
    [version, _, _, workload, provenance, .., ""] if version ==
      "luna-projection-resources-v2" ||
      version == "luna-projection-resources-v3" => {
      guard workload.split("\t").to_array()
        is [
          "workload",
          phase,
          queries,
          rows,
          history,
          bucket_queries,
          bucket_rows,
        ] &&
        provenance.split("\t").to_array() is ["provenance", uuid] &&
        !uuid.is_empty() else {
        raise InvalidSnapshot
      }
      guard @resource_policy.Workload::new(
          phase=phase.to_owned(),
          query_tokens=integer(queries),
          batch_rows=integer(rows),
          prior_history=if history == "0" { 0 } else { integer(history) },
          bucket_tokens=integer(bucket_queries),
          bucket_rows=integer(bucket_rows),
        )
        is Some(value) &&
        value.fits(max_query_tokens~, max_batch_rows~) &&
        value.scope(scope) is Some(selected) else {
        raise InvalidSnapshot
      }
      selected
    }
    _ => raise InvalidSnapshot
  }
}

///|
fn nonnegative(text : StringView) -> Int64 raise ProjectionTuningError {
  if text == "0" {
    0L
  } else {
    decimal(text)
  }
}
