///|
pub(all) suberror FusedCudaProbeError {
  FusedCudaProbeRejected(String)
} derive(Debug, Eq)

///|
fn require_fused_probe(condition : Bool, reason : String) -> Unit raise {
  if !condition {
    raise FusedCudaProbeRejected(reason)
  }
}
