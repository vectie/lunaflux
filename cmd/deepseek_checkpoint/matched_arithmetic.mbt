///|
/// Offline modes share the same policy at source export and execution.
fn matched_control_dot_law(mode : String) -> @precision.DotReductionLaw {
  match mode {
    "replay-layer-hierarchical-controls" | "replay-layer-butterfly-controls" =>
      GroupedVectorFloat32(
        groups=18,
        partitions=128,
        vector_width=2,
        publication_partitions=4,
        multiply_add=FusedMultiplyAdd,
      )
    "replay-layer-refined"
    | "replay-layer-nonlinear"
    | "replay-layer-vector-rms" => StripedTreeFloat32(partitions=32)
    _ => OrderedFloat32
  }
}
