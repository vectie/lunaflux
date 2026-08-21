///|
fn OnlineWorkerLease::release_semantic_retention_if_present(
  self : OnlineWorkerLease,
) -> Unit {
  if self.semantic_retention.is_live() {
    try! self.semantic_retention.release()
  }
}

///|
fn OnlineWorkerLease::release_required_semantic_retention(
  self : OnlineWorkerLease,
) -> Unit {
  if !self.semantic_retention.is_live() {
    abort("online semantic retention lifecycle invariant")
  }
  try! self.semantic_retention.release()
}
