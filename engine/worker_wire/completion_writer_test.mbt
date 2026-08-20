///|
fn encoded_plan_frame(
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @worker_wire.ValidatedPlanFrame raise {
  let owner = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  owner.encode_plan(plan)
}

///|
test "direct completion writer emits canonical outcomes from plan rows" {
  let submitted = submitted_wire_fixture()
  let plan = encoded_plan_frame(submitted)
  let owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let writer = owner.writer_for(plan)
  writer.append_final_prefill_completed(plan.prefill_row(0), 17)
  writer.append_decode_completed(plan.decode_row(0), 18)
  let completion = writer.submit()
  completion.validate_for(submitted)
  assert_eq(completion.entry_count(), 2)
  assert_eq(completion.entry_for_slot(0).token_id(), 17)
  assert_eq(completion.entry_for_slot(1).token_id(), 18)
}

///|
test "direct completion writer carries bounded failures for exact rows" {
  let submitted = submitted_wire_fixture()
  let plan = encoded_plan_frame(submitted)
  let owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let writer = owner.writer_for(plan)
  writer.append_prefill_failed(plan.prefill_row(0), ExecutionFailed)
  writer.append_decode_failed(plan.decode_row(0), ResourceUnavailable)
  let completion = writer.submit()
  completion.validate_for(submitted)
  assert_eq(completion.entry(0).failure(), ExecutionFailed)
  assert_eq(completion.entry(1).failure(), ResourceUnavailable)
}

///|
test "direct completion writer aborts partial state and permits exact retry" {
  let submitted = submitted_wire_fixture()
  let plan = encoded_plan_frame(submitted)
  let owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let stale = owner.writer_for(plan)
  stale.append_final_prefill_completed(plan.prefill_row(0), 17)
  stale.abort()
  try stale.append_decode_completed(plan.decode_row(0), 18) catch {
    CompletionFrameWriterNotOpen => ()
    _ => fail("expected aborted writer rejection")
  } noraise {
    _ => fail("aborted writer remained live")
  }
  let retry = owner.writer_for(plan)
  retry.append_final_prefill_completed(plan.prefill_row(0), 19)
  retry.append_decode_completed(plan.decode_row(0), 20)
  let completion = retry.submit()
  assert_eq(completion.entry_for_slot(0).token_id(), 19)
  assert_eq(completion.entry_for_slot(1).token_id(), 20)
}

///|
test "direct completion writer rejects wrong owner and row order" {
  let submitted = submitted_wire_fixture()
  let plan = encoded_plan_frame(submitted)
  let foreign = encoded_plan_frame(submitted)
  let owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let writer = owner.writer_for(plan)
  try owner.writer_for(plan) catch {
    CompletionFrameWriterAlreadyOpen => ()
    _ => fail("expected duplicate-writer rejection")
  } noraise {
    _ => fail("duplicate writer opened")
  }
  try writer.append_final_prefill_completed(foreign.prefill_row(0), 17) catch {
    CompletionFrameWriterPlanMismatch => ()
    _ => fail("expected foreign-row rejection")
  } noraise {
    _ => fail("foreign row accepted")
  }
  try writer.append_decode_completed(plan.decode_row(0), 18) catch {
    CompletionFrameWriterOrderMismatch => ()
    _ => fail("expected row-order rejection")
  } noraise {
    _ => fail("out-of-order row accepted")
  }
  try writer.submit() catch {
    InvalidFrame(Counts) => ()
    _ => fail("expected incomplete-frame rejection")
  } noraise {
    _ => fail("incomplete frame submitted")
  }
  writer.abort()
}

///|
test "validated plan identity proves exact owner and epoch" {
  let submitted = submitted_wire_fixture()
  let owner = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let first = owner.encode_plan(submitted)
  let identity = first.owner_identity()
  let epoch = first.epoch_value()
  assert_true(first.matches_owner_epoch(identity, epoch))
  let foreign = encoded_plan_frame(submitted)
  assert_false(foreign.matches_owner_epoch(identity, epoch))
  let replacement = owner.encode_plan(submitted)
  try first.matches_owner_epoch(identity, epoch) catch {
    StaleFrame => ()
    _ => fail("unexpected stale-frame identity error")
  } noraise {
    _ => fail("superseded frame retained owner authority")
  }
  assert_true(
    replacement.matches_owner_epoch(
      replacement.owner_identity(),
      replacement.epoch_value(),
    ),
  )
}
