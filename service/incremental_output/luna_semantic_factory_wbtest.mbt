///|
fn wb_semantic_factory_authority(
  inference : @inference.InferenceLimits,
) -> (@inference.LunaRequestSemanticLease, @inference.LunaRequestSemanticView) {
  let storage = try! @inference.LunaRequestSemanticStorage::new(
    inference,
    try! @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = try! storage.begin(
    stop_token_count=0,
    stop_string_count=2,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  try! write.push_stop_string_length(1)
  try! write.push_stop_string_byte(b'a')
  try! write.push_stop_string_length(3)
  try! write.push_stop_string_byte(b'E')
  try! write.push_stop_string_byte(b'N')
  try! write.push_stop_string_byte(b'D')
  try! write.push_cache_scope_byte(b'x')
  let work = try! write.finish()
  while (try! work.progress()).is_pending() {

  }
  let lease = try! work.take_lease()
  (lease, try! lease.view())
}

///|
test "semantic compatibility factory allocates exact authenticated shape" {
  let tokenizer = fixture_tokenizer()
  let inference = fixture_inference()
  let (semantic_lease, view) = wb_semantic_factory_authority(inference)
  let workspace = LunaIncrementalOutputWorkspace::new_for_luna_request_semantics(
    tokenizer,
    inference,
    LunaIncrementalOutputStepBudget::new(1),
    view,
  )
  assert_eq(workspace.pattern_lengths.length(), 2)
  assert_eq(workspace.stop_string_byte_capacity, 3)
  assert_eq(workspace.pattern_bytes.length(), 6)
  let work = workspace.begin_luna_request_semantics(view, special_policy=Skip)
  while work.progress() is LunaIncrementalOutputPending {

  }
  let output = work.take_lease()
  try! semantic_lease.release()
  let destination = FixedArray::make(8, b'!')
  assert_true(
    output.push_token_status(69, destination, destination_offset=0).is_ready(),
  )
  try! output.release()
}

///|
test "semantic compatibility factory rejects stale view before publication" {
  let tokenizer = fixture_tokenizer()
  let inference = fixture_inference()
  let (semantic_lease, view) = wb_semantic_factory_authority(inference)
  try! semantic_lease.release()
  try
    LunaIncrementalOutputWorkspace::new_for_luna_request_semantics(
      tokenizer,
      inference,
      LunaIncrementalOutputStepBudget::new(1),
      view,
    )
  catch {
    Invalid(rule=LunaIncrementalOutputEpoch, issue=Stale) => ()
    _ => fail("unexpected stale semantic factory error")
  } noraise {
    _ => fail("stale semantic view allocated a compatibility workspace")
  }
}
