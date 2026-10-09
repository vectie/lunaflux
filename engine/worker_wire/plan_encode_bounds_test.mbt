///|
fn narrow_completion_sender() -> @worker_wire.PlanFrameBuffer raise {
  @worker_wire.PlanFrameBuffer::new(
    @worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=2,
      max_decode_rows=2,
      max_plan_rows=2,
      max_plan_tokens=32,
      max_plan_pages=32,
      max_capabilities=32,
      max_completion_slots=2,
      max_sequence_tokens=64,
      max_token_id=1024,
    ),
    wire_inference_limits(),
  )
}

///|
test "prefill slot from wider source limits is rejected before frame publication" {
  let sender = narrow_completion_sender()
  let prior = sender.encode_plan(submitted_wire_fixture())
  let foreign = submitted_wire_fixture(prefill_slot=2)
  try sender.encode_plan(foreign) catch {
    InvalidFrame(CompletionSlot) => ()
    _ => fail("unexpected prefill slot rejection")
  } noraise {
    _ => fail("foreign prefill slot exceeded destination completion capacity")
  }
  assert_eq(sender.current().epoch_value(), prior.epoch_value())
  assert_eq(prior.prefill_row(0).completion_slot(), 0)
  assert_eq(prior.decode_row(0).completion_slot(), 1)
}

///|
test "decode slot from wider source limits is rejected before frame publication" {
  let sender = narrow_completion_sender()
  let prior = sender.encode_plan(submitted_wire_fixture())
  let foreign = submitted_wire_fixture(decode_slot=2)
  try sender.encode_plan(foreign) catch {
    InvalidFrame(CompletionSlot) => ()
    _ => fail("unexpected decode slot rejection")
  } noraise {
    _ => fail("foreign decode slot exceeded destination completion capacity")
  }
  assert_eq(sender.current().epoch_value(), prior.epoch_value())
  assert_eq(prior.prefill_row(0).completion_slot(), 0)
  assert_eq(prior.decode_row(0).completion_slot(), 1)
}
