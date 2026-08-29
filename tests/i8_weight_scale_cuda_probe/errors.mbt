///|
suberror I8PhysicalProbeFailure {
  I8PhysicalProbeFailure(String)
}

///|
fn require_probe(
  condition : Bool,
  message : String,
) -> Unit raise I8PhysicalProbeFailure {
  if !condition {
    raise I8PhysicalProbeFailure(message)
  }
}
