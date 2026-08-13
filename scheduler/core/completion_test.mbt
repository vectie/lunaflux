///|
fn request_with_stop_token(id : UInt64, stop_token : Int) -> TokenizedRequest {
  let limits = inference_limits()
  try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(id),
    model_identity=selected_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1, 2], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(id),
    stops=try! @inference.StopConditions::new([stop_token], [], limits),
    cache=@inference.CachePolicy::new(
      try! @inference.CacheScope::from_ascii(b"scope", limits),
      Disabled,
    ),
    stream_preference=Streaming,
    effective_limits=try! @inference.EffectiveLimits::new(
      max_new_tokens=3,
      context_ceiling=8,
      limits,
    ),
    admission_deadline=try! @inference.AdmissionDeadline::from_receipt(
      0UL,
      try! @inference.DeadlineBudget::from_milliseconds(100UL, limits),
    ),
  )
}

///|
fn append_final_result(
  writer : @worker_protocol.CompletionWriter,
  row : @worker_protocol.SubmittedPrefillRow,
  token : Int,
) -> Unit raise {
  writer.append_final_prefill_completed(
    slot=row.completion_slot(),
    request_id=row.request_id(),
    request_generation=row.request_generation(),
    processed_tokens=row.tokens().length(),
    token_id=token,
  )
}

///|
fn append_decode_result(
  writer : @worker_protocol.CompletionWriter,
  row : @worker_protocol.SubmittedDecodeRow,
  token : Int,
) -> Unit raise {
  writer.append_decode_completed(
    slot=row.completion_slot(),
    request_id=row.request_id(),
    request_generation=row.request_generation(),
    token_id=token,
  )
}

///|
test "completion publishes in plan order through overlapping plans" {
  let owner = scheduler()
  let first = owner.admit(request(401UL), 0UL)
  let second = owner.admit(request(402UL), 0UL)
  let prefill = next_plan(owner)
  let prefill_writer = owner.completion_writer(prefill)
  append_final_result(prefill_writer, prefill.prefill_row(1), 12)
  append_final_result(prefill_writer, prefill.prefill_row(0), 11)
  let prefill_completion = prefill_writer.submit()
  owner.complete(prefill, prefill_completion)
  assert_eq(owner.generated_token_count(), 2)

  let decode = next_plan(owner)
  assert_eq(decode.decode_row(0).sampling_seed_value(), 401UL)
  assert_eq(decode.decode_row(1).sampling_seed_value(), 402UL)
  assert_eq(decode.decode_row(0).sample_index(), 1)
  assert_eq(decode.decode_row(1).sample_index(), 1)
  let decode_writer = owner.completion_writer(decode)
  append_decode_result(decode_writer, decode.decode_row(1), 22)
  append_decode_result(decode_writer, decode.decode_row(0), 21)
  let decode_completion = decode_writer.submit()
  owner.complete(decode, decode_completion)
  assert_eq(owner.generated_token_count(), 4)
  let first_output = owner.take_generated_token()
  let second_output = owner.take_generated_token()
  let third_output = owner.take_generated_token()
  let fourth_output = owner.take_generated_token()
  assert_eq(first_output.request_id(), first.request_id())
  assert_eq(first_output.token_id(), 11)
  assert_eq(first_output.position(), 0)
  assert_eq(second_output.request_id(), second.request_id())
  assert_eq(second_output.token_id(), 12)
  assert_eq(second_output.position(), 0)
  assert_eq(third_output.request_id(), first.request_id())
  assert_eq(third_output.token_id(), 21)
  assert_eq(third_output.position(), 1)
  assert_eq(fourth_output.request_id(), second.request_id())
  assert_eq(fourth_output.token_id(), 22)
  assert_eq(fourth_output.position(), 1)
  assert_eq(owner.terminal_notice_count(), 2)
  assert_eq(owner.take_terminal_notice().reason(), MaximumOutputReached)
  assert_eq(owner.take_terminal_notice().reason(), MaximumOutputReached)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
}

///|
test "intermediate prefill advances without output or terminal publication" {
  let owner = scheduler()
  let handle = owner.admit(
    request_with_tokens(411UL, [1, 2, 3, 4, 5], 1, 6),
    0UL,
  )
  let plan = next_plan(owner)
  let row = plan.prefill_row(0)
  assert_false(row.produces_token())
  let writer = owner.completion_writer(plan)
  writer.append_prefill_completed(
    slot=row.completion_slot(),
    request_id=row.request_id(),
    request_generation=row.request_generation(),
    processed_tokens=row.tokens().length(),
  )
  owner.complete(plan, writer.submit())
  let status = owner.status(handle.request_id()).unwrap()
  assert_eq(status.state(), Prefill)
  assert_eq(status.processed_input_tokens(), 2)
  assert_false(status.in_flight())
  assert_eq(owner.generated_token_count(), 0)
  assert_eq(owner.terminal_notice_count(), 0)
}

///|
test "stop token is published before terminal resource release" {
  let owner = scheduler()
  let handle = owner.admit(request_with_stop_token(421UL, 9), 0UL)
  let plan = next_plan(owner)
  let writer = owner.completion_writer(plan)
  append_final_result(writer, plan.prefill_row(0), 9)
  owner.complete(plan, writer.submit())
  let output = owner.take_generated_token()
  assert_eq(output.request_id(), handle.request_id())
  assert_eq(output.token_id(), 9)
  assert_eq(output.position(), 0)
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_eq(terminal.reason(), StopTokenReached)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
}

///|
test "cancelled submitted work retires without post-cancel token" {
  let owner = scheduler()
  let handle = owner.admit(request(431UL), 0UL)
  let plan = next_plan(owner)
  let writer = owner.completion_writer(plan)
  append_final_result(writer, plan.prefill_row(0), 7)
  let completion = writer.submit()
  owner.cancel(handle)
  owner.complete(plan, completion)
  assert_eq(owner.generated_token_count(), 0)
  assert_eq(owner.terminal_notice_count(), 1)
  assert_eq(owner.take_terminal_notice().reason(), CancelledByCaller)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
}

///|
test "current worker failure publishes one bounded terminal and balances KV" {
  let owner = scheduler()
  let handle = owner.admit(request(441UL), 0UL)
  let plan = next_plan(owner)
  let row = plan.prefill_row(0)
  let writer = owner.completion_writer(plan)
  writer.append_failed(
    slot=row.completion_slot(),
    request_id=row.request_id(),
    request_generation=row.request_generation(),
    failure=ExecutionFailed,
  )
  owner.complete(plan, writer.submit())
  assert_eq(owner.generated_token_count(), 0)
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_eq(terminal.reason(), WorkerFailed)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
}
