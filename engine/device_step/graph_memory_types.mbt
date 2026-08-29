///|
/// Exact availability of authenticated CUDA-graph memory accounting.
///
/// `Absent` means the execution manifest carried no Phase-5 authorization.
/// `Unsupported` means it carried an authorization that explicitly does not
/// admit capture. Neither state is represented by a fabricated zero-byte
/// bound.
pub(all) enum PagedGraphMemoryAvailability {
  GraphMemoryAbsent
  GraphMemoryUnsupported
  GraphMemoryBounded
} derive(Debug, Eq)

///|
/// Opaque graph-memory evidence derived only from authenticated kernel
/// metadata during blueprint admission.
pub struct PagedGraphMemoryEvidence {
  priv availability : PagedGraphMemoryAvailability
  priv declared_upper_bound_bytes : Int64?
  priv alignment_bytes : Int64?
} derive(Debug, Eq)

///|
/// Returns authenticated graph-memory evidence for startup admission and
/// operator reporting. A numeric value exists only in the bounded state.
pub fn PagedKvExecutionBlueprint::graph_memory_evidence(
  self : PagedKvExecutionBlueprint,
) -> PagedGraphMemoryEvidence {
  self.graph_memory
}

///|
pub fn PagedGraphMemoryEvidence::availability(
  self : PagedGraphMemoryEvidence,
) -> PagedGraphMemoryAvailability {
  self.availability
}

///|
pub fn PagedGraphMemoryEvidence::declared_upper_bound_bytes(
  self : PagedGraphMemoryEvidence,
) -> Int64? {
  self.declared_upper_bound_bytes
}

///|
pub fn PagedGraphMemoryEvidence::alignment_bytes(
  self : PagedGraphMemoryEvidence,
) -> Int64? {
  self.alignment_bytes
}
