///|
fn wb_request_with_semantic_lease(
  id : UInt64,
) -> (TokenizedRequest, @inference.LunaRequestSemanticLease) {
  let limits = try! @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=8,
    max_new_tokens=4,
    max_context_tokens=16,
    max_token_id=31,
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
  let storage = try! @inference.LunaRequestSemanticStorage::new(
    limits,
    try! @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let write = try! storage.begin(
    stop_token_count=1,
    stop_string_count=0,
    cache_scope_byte_count=5,
    cache_permission=Disabled,
  )
  try! write.push_stop_token(7)
  for byte in b"scope" {
    try! write.push_cache_scope_byte(byte)
  }
  let work = try! write.finish()
  while (try! work.progress()).is_pending() {

  }
  let lease = try! work.take_lease()
  let candidate = try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(id),
    model_identity=wb_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1, 2], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(id),
    stop_tokens=try! lease.stop_token_view(),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=2,
      context_ceiling=8,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(100UL, limits),
    ),
  )
  (candidate, lease)
}

///|
test "exclusive commit fails closed if an unretained semantic lease releases" {
  let owner = wb_owner()
  let (candidate, lease) = wb_request_with_semantic_lease(801UL)
  let prepared = try! owner.prepare_exclusive_admission(candidate)
  try! lease.release()
  assert_eq(
    owner.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionInvalidated,
  )
  assert_true(owner.is_pristine())
}

///|
test "worker and scheduler retentions pin semantics until exact retirement" {
  let owner = wb_owner()
  let (candidate, lease) = wb_request_with_semantic_lease(802UL)
  let prepared = try! owner.prepare_exclusive_admission(candidate)
  let worker_retention = @inference.LunaRequestStopTokenRetentionSlot::new()
  try! prepared.retain_stop_tokens_into(worker_retention)
  let duplicate = @inference.LunaRequestStopTokenRetentionSlot::new()
  try prepared.retain_stop_tokens_into(duplicate) catch {
    Invalid(field=Request, issue=InvalidTransition) => ()
    _ => fail("unexpected repeated prepared-retention error")
  } noraise {
    _ => fail("prepared admission issued two worker retentions")
  }
  try lease.release() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageBusy, index=0) => ()
    _ => fail("unexpected precommit semantic-release error")
  } noraise {
    _ => fail("precommit release invalidated the worker retention")
  }
  assert_eq(
    owner.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionCommitted,
  )
  let handle = prepared.request_handle()
  worker_retention.release()
  try lease.release() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageBusy, index=0) => ()
    _ => fail("unexpected admitted semantic-release error")
  } noraise {
    _ => fail("semantic lease released before scheduler retirement")
  }
  ignore(try! owner.cancel_exact(handle))
  try! lease.release()
  assert_eq(owner.live_request_count(), 0)
}

///|
test "ordinary commit rejects retention saturation before scheduler mutation" {
  let owner = wb_owner()
  let (candidate, lease) = wb_request_with_semantic_lease(803UL)
  let external = @inference.LunaRequestStopTokenRetentionSlot::new()
  try! (try! lease.stop_token_view()).retain_into(external)
  let prepared = owner.prepare_admission(candidate)
  let worker = @inference.LunaRequestStopTokenRetentionSlot::new()
  try! prepared.retain_stop_tokens_into(worker)
  try owner.commit_admission(prepared, 0UL) catch {
    Invalid(field=Request, issue=Backpressured) => ()
    _ => fail("unexpected ordinary retention-saturation error")
  } noraise {
    _ => fail("ordinary admission exceeded exact retention capacity")
  }
  assert_true(owner.is_pristine())
  try! worker.release()
  try! external.release()
  try! lease.release()
}

///|
test "exclusive commit invalidates retention saturation before mutation" {
  let owner = wb_owner()
  let (candidate, lease) = wb_request_with_semantic_lease(804UL)
  let external = @inference.LunaRequestStopTokenRetentionSlot::new()
  try! (try! lease.stop_token_view()).retain_into(external)
  let prepared = try! owner.prepare_exclusive_admission(candidate)
  let worker = @inference.LunaRequestStopTokenRetentionSlot::new()
  try! prepared.retain_stop_tokens_into(worker)
  assert_eq(
    owner.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionInvalidated,
  )
  assert_true(owner.is_pristine())
  try! worker.release()
  try! external.release()
  try! lease.release()
}

///|
test "waiting deadline recycle releases the scheduler semantic retention" {
  let owner = wb_owner()
  let (candidate, lease) = wb_request_with_semantic_lease(805UL)
  let handle = try! owner.admit(candidate, 0UL)
  assert_true((try! owner.expire_exact(handle, 100UL)) is Some(_))
  try! lease.release()
  assert_eq(owner.live_request_count(), 0)
}

///|
test "waiting instance-loss recycle releases the scheduler semantic retention" {
  let owner = wb_owner()
  let (candidate, lease) = wb_request_with_semantic_lease(806UL)
  ignore(try! owner.admit(candidate, 0UL))
  assert_eq(try! owner.drain_instance_loss(), Complete)
  try! lease.release()
  assert_eq(owner.live_request_count(), 0)
}

///|
test "active stop completion releases scheduler semantics after membership" {
  let owner = wb_owner()
  let (candidate, lease) = wb_request_with_semantic_lease(807UL)
  ignore(try! owner.admit(candidate, 0UL))
  try lease.release() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageBusy, index=0) => ()
    _ => fail("unexpected active semantic-release error")
  } noraise {
    _ => fail("active scheduler request did not retain semantics")
  }
  let plan = wb_next_plan(owner)
  ignore(try! owner.complete(plan, wb_final_completion(owner, plan, 7)))
  try! lease.release()
  assert_eq(owner.live_request_count(), 0)
  ignore(try! owner.take_generated_token())
  assert_eq(try! owner.take_terminal_notice().reason(), StopTokenReached)
}

///|
test "active failed completion releases scheduler semantics" {
  let owner = wb_owner()
  let (candidate, lease) = wb_request_with_semantic_lease(808UL)
  ignore(try! owner.admit(candidate, 0UL))
  let plan = wb_next_plan(owner)
  let row = try! plan.prefill_row(0)
  let writer = try! owner.completion_writer(plan)
  try! writer.append_failed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    failure=ExecutionFailed,
  )
  ignore(try! owner.complete(plan, try! writer.submit()))
  try! lease.release()
  assert_eq(owner.live_request_count(), 0)
  assert_eq(try! owner.take_terminal_notice().reason(), WorkerFailed)
}
