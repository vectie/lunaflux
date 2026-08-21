///|
fn wb_semantic_limits() -> InferenceLimits raise {
  InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=16,
    max_new_tokens=8,
    max_context_tokens=24,
    max_token_id=99,
    max_stop_token_ids=3,
    max_stop_strings=2,
    max_stop_string_bytes=5,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=16,
    max_deadline_millis=1000UL,
    max_top_k=100,
    max_temperature=4.0,
  )
}

///|
test "Luna semantic generations exhaust and stale aliases cannot affect reuse" {
  let limits = wb_semantic_limits()
  let storage = LunaRequestSemanticStorage::new(
    limits,
    LunaRequestSemanticStepBudget::new(1),
  )
  let old = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  old.push_cache_scope_byte(b'x')
  let work = old.finish()
  work.abort()
  let current = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  try old.abort() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale Write error")
  } noraise {
    _ => fail("retained Write must be stale")
  }
  current.abort()
  storage.epoch = 0xffffffffffffffffUL
  try
    storage.begin(
      stop_token_count=0,
      stop_string_count=0,
      cache_scope_byte_count=1,
      cache_permission=Disabled,
    )
  catch {
    Invalid(field=LunaRequestSemantic, issue=OutOfRange, index=0) => ()
    _ => fail("unexpected generation exhaustion error")
  } noraise {
    _ => fail("expected generation exhaustion")
  }
  assert_eq(storage.phase, LUNA_REQUEST_SEMANTIC_IDLE)
}

///|
test "forged limit magnitude is rejected before capacity conversion" {
  let valid = wb_semantic_limits()
  let forged : InferenceLimits = {
    max_text_bytes: valid.max_text_bytes(),
    max_input_tokens: valid.max_input_tokens(),
    max_new_tokens: valid.max_new_tokens(),
    max_context_tokens: valid.max_context_tokens(),
    max_token_id: valid.max_token_id(),
    max_stop_token_ids: valid.max_stop_token_ids(),
    max_stop_strings: 2147483647,
    max_stop_string_bytes: valid.max_stop_string_bytes(),
    max_trace_bytes: valid.max_trace_bytes(),
    max_cache_scope_bytes: valid.max_cache_scope_bytes(),
    max_decoded_delta_bytes: valid.max_decoded_delta_bytes(),
    max_deadline_millis: valid.max_deadline_millis(),
    max_top_k: valid.max_top_k(),
    max_temperature: valid.max_temperature(),
  }
  try LunaRequestSemanticStorage::required_byte_cells(forged) catch {
    Invalid(field=Limit, issue=OutOfRange, index=0) => ()
    _ => fail("unexpected forged-limit sizing error")
  } noraise {
    _ => fail("expected forged-limit rejection")
  }
}
