///|
/// An explicit experimental CUDA catalog candidate, not a default selector.
/// These bounded row records were measured independently of model semantics;
/// callers must opt into queue fusion. See the deferred-HC benchmark report.
fn deferred_candidate(
  plan : @precision.HyperConnectionPrecision,
) -> @physical.DeferredPostPreDispatch raise {
  let (_, _, rows) = plan.geometry()
  let buckets = [(1, 1)]
  if rows > 1 {
    buckets.push((Int::min(rows, 32), 2))
  }
  if rows > 32 {
    buckets.push((Int::min(rows, 63), 4))
  }
  if rows >= 64 {
    buckets.push((64, 8))
  }
  if rows > 64 {
    buckets.push((rows, 4))
  }
  @physical.DeferredPostPreDispatch::new(
    plan,
    plan,
    buckets,
    workgroup_budget=128,
  )
}

///|
pub fn HyperConnectionCudaSource::deferred_dispatch(
  self : Self,
) -> @physical.DeferredPostPreDispatch raise {
  guard self.deferred else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  deferred_candidate(self.plan())
}
