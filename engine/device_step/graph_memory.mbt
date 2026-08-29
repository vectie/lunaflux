///|
/// Converts authenticated manifest graph metadata into the sole graph-memory
/// scalar retained by a paged blueprint.
///
/// Capture-safe metadata must declare a positive, aligned upper bound. A
/// non-capture authorization remains explicitly unsupported, while a missing
/// authorization remains explicitly absent. No state uses zero as a stand-in
/// for missing accounting.
fn derive_graph_memory_evidence(
  graph : @luna_capability_manifest.LunaGraphCapabilityMetadata?,
) -> PagedGraphMemoryEvidence raise DeviceStepError {
  match graph {
    None =>
      {
        availability: GraphMemoryAbsent,
        declared_upper_bound_bytes: None,
        alignment_bytes: None,
      }
    Some(graph) => {
      if !graph.capture_safe() {
        if graph.graph_memory_bytes() != 0L {
          raise blueprint_error(GraphMemory)
        }
        return {
          availability: GraphMemoryUnsupported,
          declared_upper_bound_bytes: None,
          alignment_bytes: None,
        }
      }
      let bound = graph.graph_memory_bytes()
      let alignment = graph.graph_memory_alignment()
      if bound <= 0L ||
        alignment <= 0L ||
        (alignment & (alignment - 1L)) != 0L ||
        bound % alignment != 0L {
        raise blueprint_error(GraphMemory)
      }
      {
        availability: GraphMemoryBounded,
        declared_upper_bound_bytes: Some(bound),
        alignment_bytes: Some(alignment),
      }
    }
  }
}
