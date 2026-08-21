///|
fn wb_service_stop_token_projection(
  limits : @inference.InferenceLimits,
  token_ids? : ArrayView[Int] = [],
) -> @inference.LunaRequestStopTokenView {
  wb_service_semantic_stop_authority(limits, token_ids).1
}

///|
fn wb_service_semantic_stop_authority(
  limits : @inference.InferenceLimits,
  token_ids : ArrayView[Int],
) -> (@inference.LunaRequestSemanticLease, @inference.LunaRequestStopTokenView) {
  let storage = try! @inference.LunaRequestSemanticStorage::new(
    limits,
    try! @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = try! storage.begin(
    stop_token_count=token_ids.length(),
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  for token in token_ids {
    try! write.push_stop_token(token)
  }
  try! write.push_cache_scope_byte(b'x')
  let work = try! write.finish()
  while (try! work.progress()).is_pending() {

  }
  let lease = try! work.take_lease()
  (lease, try! lease.stop_token_view())
}
