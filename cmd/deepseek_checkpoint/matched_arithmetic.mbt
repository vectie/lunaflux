///|
/// Offline modes share the same policy at source export and execution.
fn matched_control_dot_law(mode : String) -> @precision.DotReductionLaw {
  match mode {
    "replay-layer-refined"
    | "replay-layer-nonlinear"
    | "replay-layer-vector-rms" => StripedTreeFloat32(partitions=32)
    _ => OrderedFloat32
  }
}
