///|
fn ready_text_handoff(
  storage : LunaTextRequestHandoffStorage,
) -> LunaTextRequestHandoffLease raise {
  let limits = authority_limits()
  let write = storage.begin(
    prompt_byte_limit=8,
    stop_token_count=1,
    stop_string_count=1,
    request_id=RequestId::new(41UL),
    effective=EffectiveLimits::new(max_new_tokens=3, context_ceiling=10, limits),
    sampling=LunaValidatedSampling::sample(
      temperature=0.75,
      has_top_k=true,
      top_k=7,
      has_top_p=true,
      top_p=0.9,
      limits,
    ),
    sampling_seed=SamplingSeed::new(99UL),
    stream_preference=Streaming,
    deadline=DeadlineBudget::from_milliseconds(500UL, limits),
  )
  for byte in b"hi" {
    write.push_prompt_byte(byte)
  }
  write.push_stop_token(9)
  write.push_stop_string_length(3)
  for byte in b"END" {
    write.push_stop_string_byte(byte)
  }
  let work = write.finish()
  while work.progress().is_pending() {

  }
  work.take_lease()
}

///|
test "typed text handoff preserves one validated semantic lease across transfer" {
  let limits = authority_limits()
  let storage = LunaTextRequestHandoffStorage::new(
    limits,
    authority_model_identity(),
    CacheScope::from_ascii(b"openai", limits),
    LunaRequestSemanticStepBudget::new(1),
  )
  assert_eq(LunaTextRequestHandoffStorage::required_byte_cells(limits), 42UL)
  assert_eq(LunaTextRequestHandoffStorage::required_int_cells(limits), 10UL)
  assert_eq(LunaTextRequestHandoffStorage::required_reference_cells(), 3UL)
  let lease = ready_text_handoff(storage)
  let semantic_view = lease.semantic_view()
  assert_eq(lease.request_id().value(), 41UL)
  assert_eq(lease.prompt_length(), 2)
  assert_eq(lease.prompt_byte_at(1), b'i')
  assert_eq(lease.effective_limits().max_new_tokens(), 3)
  assert_eq(lease.sampling_seed().value(), 99UL)
  assert_eq(lease.deadline().milliseconds(), 500UL)
  assert_eq(semantic_view.stop_token_at(0), 9)
  assert_eq(
    semantic_view.stop_string_byte_at(string_index=0, byte_index=2),
    b'D',
  )
  assert_eq(semantic_view.cache_scope_byte_at(5), b'i')

  let claim = lease.take_claim()
  try lease.prompt_length() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected transferred lease rejection")
  } noraise {
    _ => fail("transferred handoff lease must be stale")
  }
  assert_eq(claim.prompt_byte_at(0), b'h')
  let semantics = claim.take_semantics()
  claim.release()
  let released_epoch = storage.epoch
  try
    storage.begin(
      prompt_byte_limit=8,
      stop_token_count=0,
      stop_string_count=0,
      request_id=RequestId::new(2UL),
      effective=EffectiveLimits::new(
        max_new_tokens=2,
        context_ceiling=8,
        limits,
      ),
      sampling=LunaValidatedSampling::greedy(),
      sampling_seed=SamplingSeed::new(2UL),
      stream_preference=Buffered,
      deadline=DeadlineBudget::from_milliseconds(10UL, limits),
    )
  catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageBusy, index=0) => ()
    _ => fail("unexpected transferred semantic busy rejection")
  } noraise {
    _ => fail("handoff storage reused while transferred semantics stayed live")
  }
  assert_eq(storage.epoch, released_epoch)
  try claim.prompt_length() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected released claim rejection")
  } noraise {
    _ => fail("released handoff claim must be stale")
  }
  assert_eq(semantics.view().stop_token_at(0), 9)
  semantics.release()

  let reused = ready_text_handoff(storage)
  reused.release()
  try semantic_view.stop_token_at(0) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale semantic view rejection")
  } noraise {
    _ => fail("old semantic view must be stale after reuse")
  }
}

///|
test "typed text handoff failure and abort deterministically restore storage" {
  let limits = authority_limits()
  let storage = LunaTextRequestHandoffStorage::new(
    limits,
    authority_model_identity(),
    CacheScope::from_ascii(b"openai", limits),
    LunaRequestSemanticStepBudget::new(1),
  )
  let write = storage.begin(
    prompt_byte_limit=8,
    stop_token_count=2,
    stop_string_count=0,
    request_id=RequestId::new(1UL),
    effective=EffectiveLimits::new(max_new_tokens=2, context_ceiling=8, limits),
    sampling=LunaValidatedSampling::greedy(),
    sampling_seed=SamplingSeed::new(1UL),
    stream_preference=Buffered,
    deadline=DeadlineBudget::from_milliseconds(10UL, limits),
  )
  write.push_prompt_byte(b'x')
  write.push_stop_token(7)
  write.push_stop_token(7)
  let work = write.finish()
  while work.progress().is_pending() {

  }
  assert_true(work.progress().is_failed())
  work.abort()
  ready_text_handoff(storage).release()
}
