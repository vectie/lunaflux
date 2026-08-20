///|
#valtype
priv enum BridgeFrameOutcome {
  BridgePrefill
  BridgeFinalPrefill(Int)
  BridgeDecode(Int)
  BridgeFailure(@worker_protocol.WorkerFailure)
}

///|
fn wb_bridge_request(
  request_value : UInt64,
  tokens : Array[Int],
) -> @core.TokenizedRequest {
  let limits = wb_service_inference_limits()
  try! @core.TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(request_value),
    model_identity=wb_service_identity(),
    input_tokens=try! @inference.TokenBuffer::new(tokens, limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(request_value),
    stops=try! @inference.StopConditions::new([], [], limits),
    cache=@inference.CachePolicy::new(
      try! @inference.CacheScope::from_ascii(b"bridge", limits),
      Disabled,
    ),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=2,
      context_ceiling=4,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(100UL, limits),
    ),
  )
}

///|
fn wb_bridge_plan(
  scheduler : @core.Scheduler,
) -> @worker_protocol.SubmittedSchedulePlan {
  let outcome = try! scheduler.build_next()
  try! scheduler.submitted_plan(outcome)
}

///|
fn wb_bridge_frame(
  plan : @worker_protocol.SubmittedSchedulePlan,
  outcome : BridgeFrameOutcome,
) -> @worker_wire.ValidatedCompletionFrame {
  let plan_owner = try! @worker_wire.PlanFrameBuffer::new(
    wb_service_worker_limits(),
    wb_service_inference_limits(),
  )
  let wire_plan = try! plan_owner.encode_plan(plan)
  let completion_owner = try! @worker_wire.CompletionFrameBuffer::new(
    wb_service_worker_limits(),
    wb_service_inference_limits(),
  )
  let writer = try! completion_owner.writer_for(wire_plan)
  match outcome {
    BridgePrefill =>
      try! writer.append_prefill_completed(try! wire_plan.prefill_row(0))
    BridgeFinalPrefill(token_id) =>
      try! writer.append_final_prefill_completed(
        try! wire_plan.prefill_row(0),
        token_id,
      )
    BridgeDecode(token_id) =>
      try! writer.append_decode_completed(
        try! wire_plan.decode_row(0),
        token_id,
      )
    BridgeFailure(failure) =>
      if (try! wire_plan.prefill_count()) == 1 {
        try! writer.append_prefill_failed(
          try! wire_plan.prefill_row(0),
          failure,
        )
      } else {
        try! writer.append_decode_failed(try! wire_plan.decode_row(0), failure)
      }
  }
  try! writer.submit()
}

///|
test "completion bridge copies every typed success outcome" {
  let scheduler = wb_service_scheduler()
  ignore(try! scheduler.admit(wb_bridge_request(1001UL, [1, 2]), 0UL))
  let ordinary = wb_bridge_plan(scheduler)
  let ordinary_completion = bridge_completion(
    scheduler,
    ordinary,
    wb_bridge_frame(ordinary, BridgePrefill),
  )
  assert_eq(try! ordinary_completion.entry(0).kind(), PrefillCompletion)
  assert_eq(try! ordinary_completion.entry(0).processed_tokens(), 1)
  assert_eq(
    try! scheduler.complete(ordinary, ordinary_completion),
    CompletionCommitted,
  )

  let final_prefill = wb_bridge_plan(scheduler)
  let final_completion = bridge_completion(
    scheduler,
    final_prefill,
    wb_bridge_frame(final_prefill, BridgeFinalPrefill(7)),
  )
  assert_eq(try! final_completion.entry(0).kind(), FinalPrefillCompletion)
  assert_eq(try! final_completion.entry(0).processed_tokens(), 1)
  assert_eq(try! final_completion.entry(0).token_id(), 7)
  assert_eq(
    try! scheduler.complete(final_prefill, final_completion),
    CompletionCommitted,
  )
  assert_eq(try! scheduler.take_generated_token().token_id(), 7)

  let decode = wb_bridge_plan(scheduler)
  let decode_completion = bridge_completion(
    scheduler,
    decode,
    wb_bridge_frame(decode, BridgeDecode(8)),
  )
  assert_eq(try! decode_completion.entry(0).kind(), DecodeCompletion)
  assert_eq(try! decode_completion.entry(0).token_id(), 8)
  assert_eq(
    try! scheduler.complete(decode, decode_completion),
    CompletionCommitted,
  )
}

///|
test "completion bridge copies typed worker failure without lower error leakage" {
  let scheduler = wb_service_scheduler()
  ignore(try! scheduler.admit(wb_bridge_request(1002UL, [1]), 0UL))
  let plan = wb_bridge_plan(scheduler)
  let completion = bridge_completion(
    scheduler,
    plan,
    wb_bridge_frame(plan, BridgeFailure(ExecutionFailed)),
  )
  assert_eq(try! completion.entry(0).kind(), FailedCompletion)
  assert_eq(try! completion.entry(0).failure(), ExecutionFailed)
  assert_eq(try! scheduler.complete(plan, completion), CompletionCommitted)

  let decode_scheduler = wb_service_scheduler()
  ignore(try! decode_scheduler.admit(wb_bridge_request(1012UL, [1]), 0UL))
  let prefill = wb_bridge_plan(decode_scheduler)
  let accepted_prefill = bridge_completion(
    decode_scheduler,
    prefill,
    wb_bridge_frame(prefill, BridgeFinalPrefill(12)),
  )
  assert_eq(
    try! decode_scheduler.complete(prefill, accepted_prefill),
    CompletionCommitted,
  )
  ignore(try! decode_scheduler.take_generated_token())
  let decode = wb_bridge_plan(decode_scheduler)
  let failed_decode = bridge_completion(
    decode_scheduler,
    decode,
    wb_bridge_frame(decode, BridgeFailure(ResourceUnavailable)),
  )
  assert_eq(try! failed_decode.entry(0).kind(), FailedCompletion)
  assert_eq(try! failed_decode.entry(0).failure(), ResourceUnavailable)
  assert_eq(
    try! decode_scheduler.complete(decode, failed_decode),
    CompletionCommitted,
  )
}

///|
test "foreign completion is rejected before scheduler writer acquisition" {
  let scheduler = wb_service_scheduler()
  ignore(try! scheduler.admit(wb_bridge_request(1003UL, [1]), 0UL))
  let plan = wb_bridge_plan(scheduler)
  let foreign_scheduler = wb_service_scheduler()
  ignore(try! foreign_scheduler.admit(wb_bridge_request(1004UL, [1]), 0UL))
  let foreign_plan = wb_bridge_plan(foreign_scheduler)
  let foreign_frame = wb_bridge_frame(foreign_plan, BridgeFinalPrefill(9))
  try bridge_completion(scheduler, plan, foreign_frame) catch {
    Invalid(WorkerOutput) => ()
    _ => fail("foreign bridge rejection leaked a lower-layer error")
  } noraise {
    _ => fail("foreign completion crossed the service bridge")
  }
  let writer = try! scheduler.completion_writer(plan)
  try! writer.abort()
}

///|
test "scheduler writer rejection is bounded to worker output" {
  let scheduler = wb_service_scheduler()
  ignore(try! scheduler.admit(wb_bridge_request(1013UL, [1]), 0UL))
  let plan = wb_bridge_plan(scheduler)
  let frame = wb_bridge_frame(plan, BridgeFinalPrefill(13))
  let held_writer = try! scheduler.completion_writer(plan)
  try bridge_completion(scheduler, plan, frame) catch {
    Invalid(WorkerOutput) => ()
    _ => fail("writer acquisition leaked a scheduler protocol error")
  } noraise {
    _ => fail("bridge acquired a duplicate scheduler writer")
  }
  try! held_writer.abort()
}

///|
test "post-open bridge rejection aborts writer and permits exact reuse" {
  let scheduler = wb_service_scheduler()
  ignore(try! scheduler.admit(wb_bridge_request(1005UL, [1]), 0UL))
  let prefill = wb_bridge_plan(scheduler)

  let decode_scheduler = wb_service_scheduler()
  ignore(try! decode_scheduler.admit(wb_bridge_request(1006UL, [1]), 0UL))
  let decode_prefill = wb_bridge_plan(decode_scheduler)
  let decode_prefill_completion = bridge_completion(
    decode_scheduler,
    decode_prefill,
    wb_bridge_frame(decode_prefill, BridgeFinalPrefill(10)),
  )
  assert_eq(
    try! decode_scheduler.complete(decode_prefill, decode_prefill_completion),
    CompletionCommitted,
  )
  ignore(try! decode_scheduler.take_generated_token())
  let decode = wb_bridge_plan(decode_scheduler)
  let decode_frame = wb_bridge_frame(decode, BridgeDecode(11))

  let writer = try! scheduler.completion_writer(prefill)
  try submit_bridge_after_open(writer, prefill, decode_frame) catch {
    Invalid(WorkerOutput) => ()
    _ => fail("post-open bridge rejection leaked a lower-layer error")
  } noraise {
    _ => fail("decode completion crossed a prefill bridge")
  }
  let retry = try! scheduler.completion_writer(prefill)
  try! retry.abort()
}
