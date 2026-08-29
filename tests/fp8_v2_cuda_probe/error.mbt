///|
suberror Fp8V2CudaProbeError {
  Rejected(String)
} derive(Debug, Eq)

///|
fn require_fp8_probe(condition : Bool, reason : String) -> Unit raise {
  if !condition {
    raise Fp8V2CudaProbeError::Rejected(reason)
  }
}
