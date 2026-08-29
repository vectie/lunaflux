///|
suberror LunaTensorCoreProbeError {
  LunaTensorCoreProbeRejected(String)
}

///|
fn require_tensor_core_probe(condition : Bool, reason : String) -> Unit raise {
  if !condition {
    raise LunaTensorCoreProbeRejected(reason)
  }
}
