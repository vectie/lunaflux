///|
fn token_bound_limits() -> @inference.InferenceLimits {
  try! @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=8,
    max_new_tokens=4,
    max_context_tokens=16,
    max_token_id=63,
    max_stop_token_ids=4,
    max_stop_strings=2,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn token_bound_request(
  id : UInt64,
  input_tokens : @inference.TokenBuffer,
  limits : @inference.InferenceLimits,
) -> TokenizedRequest {
  try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(id),
    model_identity=selected_identity(),
    input_tokens~,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(id),
    stop_tokens=stop_token_projection(limits, []),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=1,
      context_ceiling=input_tokens.length() + 1,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(100UL, limits),
    ),
  )
}

///|
test "admission uses the exact prompt maximum rather than its validation ceiling" {
  let limits = token_bound_limits()
  let candidate = token_bound_request(
    901UL,
    try! @inference.TokenBuffer::new([31, 1, 2], limits),
    limits,
  )
  let ordinary = scheduler()
  ignore(ordinary.admit(candidate, 0UL))
  assert_eq(ordinary.live_request_count(), 1)
  let exclusive = scheduler()
  let prepared = exclusive.prepare_exclusive_admission(candidate)
  exclusive.abort_exclusive_admission(prepared)
  assert_true(exclusive.is_pristine())
}

///|
test "ordinary and exclusive admission reject an exact maximum out of range without mutation" {
  let limits = token_bound_limits()
  let candidate = token_bound_request(
    902UL,
    try! @inference.TokenBuffer::new([1, 32, 2], limits),
    limits,
  )
  let ordinary = scheduler()
  try ordinary.admit(candidate, 0UL) catch {
    error => assert_eq(error, Invalid(field=Request, issue=OutOfRange))
  } noraise {
    _ => fail("ordinary admission accepted an out-of-range prompt maximum")
  }
  assert_true(ordinary.is_pristine())
  let exclusive = scheduler()
  try exclusive.prepare_exclusive_admission(candidate) catch {
    error => assert_eq(error, Invalid(field=Request, issue=OutOfRange))
  } noraise {
    _ => fail("exclusive admission accepted an out-of-range prompt maximum")
  }
  assert_true(exclusive.is_pristine())
}

///|
test "ordinary and exclusive admission reject a stale leased maximum without mutation" {
  let limits = token_bound_limits()
  let storage = @inference.LunaTokenBufferStorage::new(capacity=3, limits~)
  let write = storage.begin(token_count=3)
  write.push(1)
  write.push(31)
  write.push(2)
  let lease = write.seal()
  let candidate = token_bound_request(903UL, lease.token_buffer(), limits)
  lease.release()
  let ordinary = scheduler()
  try ordinary.admit(candidate, 0UL) catch {
    error => assert_eq(error, Invalid(field=Request, issue=Stale))
  } noraise {
    _ => fail("ordinary admission accepted stale Luna token storage")
  }
  assert_true(ordinary.is_pristine())
  let exclusive = scheduler()
  try exclusive.prepare_exclusive_admission(candidate) catch {
    error => assert_eq(error, Invalid(field=Request, issue=Stale))
  } noraise {
    _ => fail("exclusive admission accepted stale Luna token storage")
  }
  assert_true(exclusive.is_pristine())
}

///|
test "exclusive commit consumes a backing invalidated after prepare" {
  let limits = token_bound_limits()
  let storage = @inference.LunaTokenBufferStorage::new(capacity=2, limits~)
  let write = storage.begin(token_count=2)
  write.push(1)
  write.push(31)
  let lease = write.seal()
  let candidate = token_bound_request(904UL, lease.token_buffer(), limits)
  let owner = scheduler()
  let prepared = owner.prepare_exclusive_admission(candidate)
  assert_false(owner.is_pristine())
  lease.release()
  assert_eq(
    owner.commit_exclusive_admission(prepared, 100UL),
    ExclusiveAdmissionInvalidated,
  )
  assert_true(owner.is_pristine())
  assert_eq(owner.live_request_count(), 0)
  assert_eq(
    owner.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionInvalidated,
  )
}
