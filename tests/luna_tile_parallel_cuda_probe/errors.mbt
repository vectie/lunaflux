///|
suberror LunaTileParallelCudaProbeError {
  LunaTileParallelCudaProbeRejected(String)
}

///|
fn require_luna_tile_parallel_probe(
  condition : Bool,
  reason : String,
) -> Unit raise {
  if !condition {
    raise LunaTileParallelCudaProbeRejected(reason)
  }
}
