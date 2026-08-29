///|
const MAX_GRAPH_MEMORY_INT64 : Int64 = 0x7fffffffffffffffL

///|
fn checked_graph_memory_add(
  left : Int64,
  right : Int64,
) -> Int64 raise RuntimeCapacityError {
  if left < 0L || right < 0L || left > MAX_GRAPH_MEMORY_INT64 - right {
    raise Invalid(DeviceMemoryUpperBound)
  }
  left + right
}

///|
/// Resolves graph-memory accounting from one already-admitted execution
/// blueprint and the independently authenticated runtime-descriptor ceiling.
///
/// Workspace is already a subregion of `activation_workspace_reserved_bytes`
/// and is deliberately not accepted as another input. A descriptor without a
/// graph ceiling is eager-only, so even future capture-safe metadata remains
/// explicitly absent from its capacity projection. A positive ceiling
/// activates accounting only when matching positive capture evidence exists.
pub fn resolve_graph_memory(
  blueprint : @device_step.PagedKvExecutionBlueprint,
  descriptor_ceiling_bytes : Int64?,
  weight_reserved_bytes~ : Int64,
  activation_workspace_reserved_bytes~ : Int64,
  kv_reserved_bytes~ : Int64,
) -> ResolvedGraphMemoryCapacity raise RuntimeCapacityError {
  let graph = blueprint.graph_memory_evidence()
  let evidence_availability = match graph.availability() {
    GraphMemoryAbsent => GraphAccountingAbsent
    GraphMemoryUnsupported => GraphAccountingUnsupported
    GraphMemoryBounded => GraphAccountingBounded
  }
  resolve_graph_memory_parts(
    evidence_availability,
    graph.declared_upper_bound_bytes(),
    graph.alignment_bytes(),
    descriptor_ceiling_bytes,
    weight_reserved_bytes~,
    activation_workspace_reserved_bytes~,
    kv_reserved_bytes~,
  )
}

///|
fn resolve_graph_memory_parts(
  availability : RuntimeGraphMemoryAvailability,
  declared_upper_bound_bytes : Int64?,
  declared_alignment_bytes : Int64?,
  descriptor_ceiling_bytes : Int64?,
  weight_reserved_bytes~ : Int64,
  activation_workspace_reserved_bytes~ : Int64,
  kv_reserved_bytes~ : Int64,
) -> ResolvedGraphMemoryCapacity raise RuntimeCapacityError {
  let arena_reserved_total_bytes = checked_graph_memory_add(
    checked_graph_memory_add(
      weight_reserved_bytes, activation_workspace_reserved_bytes,
    ),
    kv_reserved_bytes,
  )
  if descriptor_ceiling_bytes is None {
    match availability {
      GraphAccountingAbsent | GraphAccountingUnsupported =>
        if declared_upper_bound_bytes is Some(_) ||
          declared_alignment_bytes is Some(_) {
          raise Invalid(GraphMemoryEvidence)
        }
      GraphAccountingBounded => {
        guard declared_upper_bound_bytes is Some(bound) &&
          declared_alignment_bytes is Some(alignment) else {
          raise Invalid(GraphMemoryEvidence)
        }
        if bound <= 0L ||
          alignment <= 0L ||
          (alignment & (alignment - 1L)) != 0L ||
          bound % alignment != 0L {
          raise Invalid(GraphMemoryEvidence)
        }
      }
    }
    return {
      availability: GraphAccountingAbsent,
      arena_reserved_total_bytes,
      declared_upper_bound_bytes: None,
      declared_alignment_bytes: None,
      descriptor_ceiling_bytes: None,
      device_startup_upper_bound_bytes: None,
    }
  }
  match availability {
    GraphAccountingAbsent => {
      if declared_upper_bound_bytes is Some(_) ||
        declared_alignment_bytes is Some(_) {
        raise Invalid(GraphMemoryEvidence)
      }
      raise Invalid(GraphMemoryEvidence)
    }
    GraphAccountingUnsupported => {
      if declared_upper_bound_bytes is Some(_) ||
        declared_alignment_bytes is Some(_) {
        raise Invalid(GraphMemoryEvidence)
      }
      raise Invalid(GraphMemoryEvidence)
    }
    GraphAccountingBounded => {
      guard declared_upper_bound_bytes is Some(bound) &&
        declared_alignment_bytes is Some(alignment) else {
        raise Invalid(GraphMemoryEvidence)
      }
      guard descriptor_ceiling_bytes is Some(ceiling) else {
        raise Invalid(GraphMemoryCeiling)
      }
      if bound <= 0L ||
        alignment <= 0L ||
        (alignment & (alignment - 1L)) != 0L ||
        bound % alignment != 0L ||
        ceiling <= 0L ||
        bound > ceiling {
        raise Invalid(GraphMemoryCeiling)
      }
      let device_startup_upper_bound_bytes = checked_graph_memory_add(
        arena_reserved_total_bytes, bound,
      )
      {
        availability: GraphAccountingBounded,
        arena_reserved_total_bytes,
        declared_upper_bound_bytes: Some(bound),
        declared_alignment_bytes: Some(alignment),
        descriptor_ceiling_bytes: Some(ceiling),
        device_startup_upper_bound_bytes: Some(device_startup_upper_bound_bytes),
      }
    }
  }
}

///|
pub fn ResolvedGraphMemoryCapacity::availability(
  self : ResolvedGraphMemoryCapacity,
) -> RuntimeGraphMemoryAvailability {
  self.availability
}

///|
pub fn ResolvedGraphMemoryCapacity::arena_reserved_total_bytes(
  self : ResolvedGraphMemoryCapacity,
) -> Int64 {
  self.arena_reserved_total_bytes
}

///|
pub fn ResolvedGraphMemoryCapacity::declared_upper_bound_bytes(
  self : ResolvedGraphMemoryCapacity,
) -> Int64? {
  self.declared_upper_bound_bytes
}

///|
pub fn ResolvedGraphMemoryCapacity::declared_alignment_bytes(
  self : ResolvedGraphMemoryCapacity,
) -> Int64? {
  self.declared_alignment_bytes
}

///|
pub fn ResolvedGraphMemoryCapacity::descriptor_ceiling_bytes(
  self : ResolvedGraphMemoryCapacity,
) -> Int64? {
  self.descriptor_ceiling_bytes
}

///|
pub fn ResolvedGraphMemoryCapacity::device_startup_upper_bound_bytes(
  self : ResolvedGraphMemoryCapacity,
) -> Int64? {
  self.device_startup_upper_bound_bytes
}
