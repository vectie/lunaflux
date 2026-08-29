///|
fn request_with_stop_token(id : UInt64, stop_token : Int) -> TokenizedRequest {
  let limits = inference_limits()
  try! TokenizedRequest::new(
    request_id=try! @inference.RequestId::new(id),
    model_identity=selected_identity(),
    input_tokens=try! @inference.TokenBuffer::new([1, 2], limits),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=try! @inference.SamplingSeed::new(id),
    stop_tokens=stop_token_projection(limits, [stop_token]),
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
  ignore(owner.complete(prefill, prefill_completion))
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
  ignore(owner.complete(decode, decode_completion))
  assert_eq(owner.generated_token_count(), 4)
  assert_eq(owner.next_publication_kind(), GeneratedTokenPublication)
  try owner.take_terminal_notice() catch {
    Invalid(field=PublicationSequence, issue=OutOfOrder) => ()
    _ => fail("unexpected wrong-ring dequeue error")
  } noraise {
    _ => fail("newer terminal bypassed an older token")
  }
  let first_publication = owner.take_next_publication()
  assert_eq(first_publication.kind(), GeneratedTokenPublication)
  try first_publication.terminal_notice() catch {
    Invalid(field=PublicationSequence, issue=OutOfOrder) => ()
    _ => fail("unexpected generated-publication projection error")
  } noraise {
    _ => fail("generated publication projected as terminal")
  }
  let first_output = first_publication.generated_token_notice()
  assert_eq(first_output.publication_sequence(), 1UL)
  assert_eq(first_publication.request_slot(), first.slot())
  assert_eq(owner.next_publication_kind(), GeneratedTokenPublication)
  let second_output = owner.take_generated_token()
  assert_eq(second_output.publication_sequence(), 2UL)
  assert_eq(owner.next_publication_kind(), GeneratedTokenPublication)
  let third_output = owner.take_generated_token()
  assert_eq(third_output.publication_sequence(), 3UL)
  assert_eq(owner.next_publication_kind(), TerminalPublication)
  try owner.take_generated_token() catch {
    Invalid(field=PublicationSequence, issue=OutOfOrder) => ()
    _ => fail("unexpected reverse wrong-ring dequeue error")
  } noraise {
    _ => fail("newer token bypassed an older terminal")
  }
  let first_terminal_publication = owner.take_next_publication()
  assert_eq(first_terminal_publication.kind(), TerminalPublication)
  try first_terminal_publication.generated_token_notice() catch {
    Invalid(field=PublicationSequence, issue=OutOfOrder) => ()
    _ => fail("unexpected terminal-publication projection error")
  } noraise {
    _ => fail("terminal publication projected as generated token")
  }
  let first_terminal = first_terminal_publication.terminal_notice()
  assert_eq(first_terminal.publication_sequence(), 4UL)
  assert_eq(first_terminal_publication.request_slot(), first.slot())
  assert_eq(owner.next_publication_kind(), GeneratedTokenPublication)
  try owner.take_terminal_notice() catch {
    Invalid(field=PublicationSequence, issue=OutOfOrder) => ()
    _ => fail("unexpected interleaved dequeue error")
  } noraise {
    _ => fail("newer terminal bypassed an interleaved token")
  }
  let fourth_output = owner.take_generated_token()
  assert_eq(fourth_output.publication_sequence(), 5UL)
  assert_eq(owner.next_publication_kind(), TerminalPublication)
  let second_terminal = owner.take_terminal_notice()
  assert_eq(second_terminal.publication_sequence(), 6UL)
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
  assert_eq(first_terminal.reason(), MaximumOutputReached)
  assert_eq(second_terminal.reason(), MaximumOutputReached)
  try owner.next_publication_kind() catch {
    Invalid(field=OutputCapacity, issue=NoRunnableWork) => ()
    _ => fail("unexpected empty publication error")
  } noraise {
    _ => fail("expected empty publication head rejection")
  }
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
  ignore(owner.complete(plan, writer.submit()))
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
  ignore(owner.complete(plan, writer.submit()))
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
  ignore(owner.complete(plan, completion))
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
  ignore(owner.complete(plan, writer.submit()))
  assert_eq(owner.generated_token_count(), 0)
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_eq(terminal.reason(), WorkerFailed)
  assert_eq(owner.live_request_count(), 0)
  assert_eq(owner.free_page_count(), 8)
}

///|
test "submitted completion retries exact A and B owners in sequence" {
  let owner = scheduler()
  ignore(owner.admit(request(451UL), 0UL))
  let plan_a = next_plan(owner)
  let writer_a = owner.completion_writer(plan_a)
  append_final_result(writer_a, plan_a.prefill_row(0), 7)
  let completion_a = writer_a.submit()

  ignore(owner.admit(request(452UL), 0UL))
  let plan_b = next_plan(owner)
  let writer_b = owner.completion_writer(plan_b)
  append_final_result(writer_b, plan_b.prefill_row(0), 8)
  let completion_b = writer_b.submit()

  try owner.complete_submitted(plan_b) catch {
    Invalid(field=Completion, issue=OutOfOrder) => ()
    _ => fail("unexpected B-side ordering failure")
  } noraise {
    _ => fail("B-side completion retired before A")
  }
  assert_eq(owner.complete_submitted(plan_a), CompletionCommitted)
  assert_eq(owner.complete_submitted(plan_b), CompletionCommitted)
  assert_eq(owner.take_generated_token().token_id(), 7)
  assert_eq(owner.take_generated_token().token_id(), 8)
  try owner.complete_submitted(plan_a) catch {
    WorkerProtocol(StalePlanBufferEpoch) => ()
    _ => fail("unexpected retired-plan retry failure")
  } noraise {
    _ => fail("retired A-side plan was completed twice")
  }
  try completion_a.sequence_wire() catch {
    StaleCompletionBufferEpoch => ()
    _ => fail("unexpected retired A completion epoch failure")
  } noraise {
    _ => fail("retired A completion epoch remained current")
  }
  try completion_b.sequence_wire() catch {
    StaleCompletionBufferEpoch => ()
    _ => fail("unexpected retired B completion epoch failure")
  } noraise {
    _ => fail("retired B completion epoch remained current")
  }
}

///|
test "submitted completion rejects a foreign plan owner" {
  let owner = scheduler()
  ignore(owner.admit(request(461UL), 0UL))
  let plan = next_plan(owner)
  let writer = owner.completion_writer(plan)
  append_final_result(writer, plan.prefill_row(0), 9)
  ignore(writer.submit())

  let foreign = scheduler()
  ignore(foreign.admit(request(462UL), 0UL))
  let foreign_plan = next_plan(foreign)
  try owner.complete_submitted(foreign_plan) catch {
    WorkerProtocol(PlanBufferMismatch) => ()
    _ => fail("unexpected foreign-plan completion failure")
  } noraise {
    _ => fail("foreign plan owner selected a scheduler completion")
  }
  assert_eq(owner.complete_submitted(plan), CompletionCommitted)
}

///|
test "submitted completion backpressure preserves its exact epoch" {
  let owner = tight_scheduler()
  ignore(owner.admit(request(471UL, max_new_tokens=4), 0UL))

  let prefill = next_plan(owner)
  let prefill_writer = owner.completion_writer(prefill)
  append_final_result(prefill_writer, prefill.prefill_row(0), 10)
  ignore(owner.complete(prefill, prefill_writer.submit()))

  let first_decode = next_plan(owner)
  let first_writer = owner.completion_writer(first_decode)
  append_decode_result(first_writer, first_decode.decode_row(0), 11)
  ignore(owner.complete(first_decode, first_writer.submit()))

  let pressured = next_plan(owner)
  let pressured_writer = owner.completion_writer(pressured)
  append_decode_result(pressured_writer, pressured.decode_row(0), 12)
  let completion = pressured_writer.submit()
  assert_eq(owner.complete_submitted(pressured), CompletionBackpressured)
  assert_eq(completion.sequence_wire(), pressured.sequence_wire())
  assert_eq(owner.take_generated_token().token_id(), 10)
  assert_eq(owner.take_generated_token().token_id(), 11)
  assert_eq(owner.complete_submitted(pressured), CompletionCommitted)
  assert_eq(owner.take_generated_token().token_id(), 12)
  try completion.sequence_wire() catch {
    StaleCompletionBufferEpoch => ()
    _ => fail("unexpected pressured completion epoch failure")
  } noraise {
    _ => fail("committed pressured completion epoch remained current")
  }
}
