///|
priv struct E2ESemanticLeaseOwner {
  slots : FixedArray[@inference.LunaRequestSemanticLease?]
  mut count : Int
}

///|
let e2e_semantic_lease_owner : E2ESemanticLeaseOwner = {
  slots: FixedArray::make(256, None),
  count: 0,
}

///|
fn e2e_stop_token_projection(
  limits : @inference.InferenceLimits,
  scope : BytesView,
) -> @inference.LunaRequestStopTokenView raise {
  if e2e_semantic_lease_owner.count >= e2e_semantic_lease_owner.slots.length() {
    abort("worker-service E2E semantic lease capacity")
  }
  let storage = @inference.LunaRequestSemanticStorage::new(
    limits,
    @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=scope.length(),
    cache_permission=Disabled,
  )
  for byte in scope {
    write.push_cache_scope_byte(byte)
  }
  let work = write.finish()
  while work.progress().is_pending() {

  }
  let lease = work.take_lease()
  e2e_semantic_lease_owner.slots[e2e_semantic_lease_owner.count] = Some(lease)
  e2e_semantic_lease_owner.count += 1
  lease.stop_token_view()
}
