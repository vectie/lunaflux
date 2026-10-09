///|
test "retained completion slot preserves exact owner epoch and abort semantics" {
  let slot = @worker_wire.FrameCompletionSlot::new()
  let submitted = submitted_wire_fixture()
  let plan = encoded_plan_frame(submitted)
  let foreign = encoded_plan_frame(submitted)
  let owner = @worker_wire.CompletionFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let writer = owner.writer_for(plan)
  try slot.bind(foreign, writer) catch {
    CompletionFrameWriterPlanMismatch => ()
    error => raise error
  } noraise {
    _ => fail("retained foreign plan")
  }
  slot.bind(plan, writer)
  try slot.bind(plan, writer) catch {
    CompletionFrameWriterAlreadyOpen => ()
    error => raise error
  } noraise {
    _ => fail("replaced an active lease")
  }
  let retained = slot.plan()
  slot.writer().append_final_prefill_completed(retained.prefill_row(0), 17)
  slot.writer().append_decode_completed(retained.decode_row(0), 18)
  slot.submit().validate_for(submitted)
  slot.completion().validate_for(submitted)
  assert_eq(slot.completion().entry(1).token_id(), 18)
  // Reusing the underlying buffer invalidates the old retained epoch too.
  let retry = owner.writer_for(plan)
  try slot.completion() catch {
    StaleFrame => ()
    error => raise error
  } noraise {
    _ => fail("retained a superseded completion")
  }
  slot.bind(plan, retry)
  slot.close()
  slot.close()
  try retry.submit() catch {
    CompletionFrameWriterNotOpen => ()
    error => raise error
  } noraise {
    _ => fail("cancellation left a writer open")
  }
  let writer = owner.writer_for(plan)
  writer.abort()
}
