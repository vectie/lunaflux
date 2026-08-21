///|
test "admission validates identity duplicates capacity and received deadline" {
  let owner = scheduler()
  let first = owner.admit(request(1UL), 10UL)
  assert_eq(first.request_id().value(), 1UL)
  assert_eq(first.request_generation().value(), 1UL)
  assert_eq(owner.live_request_count(), 1)
  assert_eq(owner.waiting_request_count(), 1)
  assert_eq(
    owner.status(first.request_id()).unwrap().deadline_at_millis(),
    100UL,
  )
  try owner.admit(request(1UL), 10UL) catch {
    error => assert_eq(error, Invalid(field=Request, issue=Duplicate))
  } noraise {
    _ => fail("expected duplicate rejection")
  }
  let other_identity = test_identity(
    "1111111111111111111111111111111111111111111111111111111111111111",
  )
  try owner.admit(request(2UL, identity=other_identity), 10UL) catch {
    error => assert_eq(error, Invalid(field=ModelIdentity, issue=Mismatch))
  } noraise {
    _ => fail("expected model identity rejection")
  }
  ignore(owner.admit(request(2UL), 10UL))
  try owner.admit(request(3UL), 10UL) catch {
    error => assert_eq(error, Invalid(field=WaitingCapacity, issue=Exhausted))
  } noraise {
    _ => fail("expected waiting capacity rejection")
  }
  let fresh = scheduler()
  try fresh.admit(request(4UL, deadline=10UL), 10UL) catch {
    error => assert_eq(error, Invalid(field=Deadline, issue=Expired))
  } noraise {
    _ => fail("request delay incorrectly rebased its deadline")
  }
}

///|
test "scheduler constructor binds recipe identity and loaded generation" {
  let limits = worker_limits()
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let wrong_identity = test_identity(
    "1111111111111111111111111111111111111111111111111111111111111111",
  )
  let wrong_recipe = try! @worker_protocol.RowCapabilityRecipe::new(
    identity=wrong_identity,
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    limits,
  )
  try
    Scheduler::new(
      resolved_capacity(),
      selected_identity(),
      generation,
      wrong_recipe,
      wrong_recipe,
      wrong_recipe,
    )
  catch {
    error => assert_eq(error, Invalid(field=ModelIdentity, issue=Mismatch))
  } noraise {
    _ => fail("expected recipe provenance rejection")
  }
}

///|
test "admission rejects requests that can never preserve decode reserve" {
  let owner = tight_scheduler()
  // Two prompt pages cannot preserve two reserved pages in a three-page arena.
  let request = request_with_tokens(30UL, [1, 2, 3, 4, 5], 1, 6)
  try owner.admit(request, 0UL) catch {
    error => assert_eq(error, Invalid(field=Context, issue=OutOfRange))
  } noraise {
    _ => fail("expected impossible prefill reserve rejection")
  }
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.waiting_request_count(), 0)
  assert_eq(owner.free_page_count(), 3)
}

///|
test "admission rejects token IDs outside the loaded worker vocabulary" {
  let broad = try! @inference.InferenceLimits::new(
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
  let candidate = try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(35UL),
    model_identity=selected_identity(),
    input_tokens=try! @inference.TokenBuffer::new([40], broad),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(35UL),
    stop_tokens=stop_token_projection(broad, []),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=1,
      context_ceiling=2,
      broad,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(100UL, broad),
    ),
  )
  let owner = scheduler()
  try owner.admit(candidate, 0UL) catch {
    error => assert_eq(error, Invalid(field=Request, issue=OutOfRange))
  } noraise {
    _ => fail("expected worker vocabulary rejection")
  }
  assert_eq(owner.live_request_count(), 0)
}

///|
test "waiting cancellation publishes slot-free notice and never aliases re-admission" {
  let owner = scheduler()
  let old = owner.admit(request(7UL), 0UL)
  owner.cancel(old)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.waiting_request_count(), 0)
  assert_eq(owner.terminal_notice_count(), 1)
  let current = owner.admit(request(7UL), 1UL)
  assert_true(current.slot() != old.slot())
  let notice = owner.take_terminal_notice()
  assert_eq(notice.request_id(), old.request_id())
  assert_eq(notice.request_generation().value(), 2UL)
  assert_eq(notice.reason(), CancelledByCaller)
  assert_true(
    current.request_generation().value() > notice.request_generation().value(),
  )
  try owner.cancel(old) catch {
    error => assert_eq(error, Invalid(field=Request, issue=Stale))
  } noraise {
    _ => fail("expected old handle rejection")
  }
  assert_eq(owner.status(current.request_id()).unwrap().handle(), current)
}

///|
test "exact cancellation returns the terminal generation it published" {
  let owner = scheduler()
  let handle = owner.admit(request(8UL), 0UL)
  let generation = owner.cancel_exact(handle)
  let notice = owner.take_terminal_notice()
  assert_eq(generation, notice.request_generation())
  assert_eq(notice.request_id(), handle.request_id())
  assert_eq(notice.reason(), CancelledByCaller)
}

///|
test "exact deadline leaves an unexpired request retryable" {
  let owner = scheduler()
  let handle = owner.admit(request(9UL, deadline=10UL), 0UL)
  assert_eq(owner.expire_exact(handle, 9UL), None)
  assert_eq(owner.status(handle.request_id()).unwrap().handle(), handle)
  let generation = owner.expire_exact(handle, 10UL).unwrap()
  let notice = owner.take_terminal_notice()
  assert_eq(generation, notice.request_generation())
  assert_eq(notice.reason(), DeadlineReached)
}

///|
test "deadline sweep preflights terminal capacity without partial mutation" {
  let owner = scheduler()
  for id in 1..<=4 {
    let handle = owner.admit(request(id.to_uint64()), 0UL)
    owner.cancel(handle)
  }
  let retained = owner.admit(request(20UL, deadline=10UL), 0UL)
  try owner.expire_deadlines(10UL) catch {
    error =>
      assert_eq(
        error,
        Invalid(field=TerminalNoticeCapacity, issue=Backpressured),
      )
  } noraise {
    _ => fail("expected transactional deadline backpressure")
  }
  let status = owner.status(retained.request_id()).unwrap()
  assert_eq(status.handle(), retained)
  assert_eq(status.state(), Waiting)
  assert_eq(owner.live_request_count(), 1)
  assert_eq(owner.waiting_request_count(), 1)
}

///|
test "deadline expiry is deterministic and leaves resource owners balanced" {
  let owner = scheduler()
  ignore(owner.admit(request(10UL, deadline=10UL), 5UL))
  ignore(owner.admit(request(11UL, deadline=20UL), 5UL))
  assert_eq(owner.expire_deadlines(9UL), 0)
  assert_eq(owner.expire_deadlines(10UL), 1)
  let first = owner.take_terminal_notice()
  assert_eq(first.request_id().value(), 10UL)
  assert_eq(first.reason(), DeadlineReached)
  assert_eq(owner.expire_deadlines(20UL), 1)
  let second = owner.take_terminal_notice()
  assert_eq(second.request_id().value(), 11UL)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
}

///|
test "full terminal ring backpressures transactionally" {
  let owner = scheduler()
  for id in 1..<=4 {
    let handle = owner.admit(request(id.to_uint64()), 0UL)
    owner.cancel(handle)
  }
  assert_eq(owner.terminal_notice_count(), 4)
  let retained = owner.admit(request(9UL), 0UL)
  try owner.cancel(retained) catch {
    error =>
      assert_eq(
        error,
        Invalid(field=TerminalNoticeCapacity, issue=Backpressured),
      )
  } noraise {
    _ => fail("expected terminal publication backpressure")
  }
  let status = owner.status(retained.request_id()).unwrap()
  assert_eq(status.state(), Waiting)
  assert_eq(status.handle(), retained)
  assert_eq(owner.live_request_count(), 1)
  assert_eq(owner.waiting_request_count(), 1)
}

///|
test "tokenized boundary rejects context that cannot hold prompt plus output" {
  let limits = inference_limits()
  let input = try! @inference.TokenBuffer::new([1, 2, 3], limits)
  try
    TokenizedRequest::new(
      request_id=try! @inference.RequestId::new(99UL),
      model_identity=selected_identity(),
      input_tokens=input,
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=try! @inference.SamplingSeed::new(99UL),
      stop_tokens=stop_token_projection(limits, []),
      stream_preference=Streaming,
      effective_limits=try! @inference.EffectiveLimits::new(
        max_new_tokens=2,
        context_ceiling=4,
        limits,
      ),
      admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
        0UL,
        try! @inference.DeadlineBudget::from_milliseconds(10UL, limits),
      ),
    )
  catch {
    error => assert_eq(error, Invalid(field=Context, issue=OutOfRange))
  } noraise {
    _ => fail("expected tokenized context rejection")
  }
}

///|
test "tokenized scheduler boundary cannot observe semantic stop strings" {
  let limits = inference_limits()
  let candidate = try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(100UL),
    model_identity=selected_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(100UL),
    stop_tokens=stop_token_projection(limits, [], stop_strings=[b"stop"]),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=1,
      context_ceiling=2,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(10UL, limits),
    ),
  )
  let owner = scheduler()
  ignore(owner.admit(candidate, 0UL))
  assert_eq(owner.live_request_count(), 1)
}
