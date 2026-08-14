///|
test "completion epoch overflow leaves exact owner consumed" {
  let limits = WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=2,
    max_plan_tokens=2,
    max_plan_pages=2,
    max_capabilities=2,
    max_completion_slots=2,
    max_sequence_tokens=8,
    max_token_id=64,
  )
  let completion = CompletionBuffer::new(limits)
  let plan = SchedulePlanBuffer::new(limits)
  completion.epoch = 0xffffffffffffffffUL
  completion.state = CompletionSubmitted
  plan.epoch = 1UL
  plan.state = Submitted
  let submitted_completion = SubmittedCompletion(completion, completion.epoch)
  let submitted_plan = SubmittedSchedulePlan(plan, plan.epoch)
  let consumed = completion.consume(submitted_completion, submitted_plan)
  try completion.reset(consumed) catch {
    IdentityOverflow(CompletionBufferEpochIdentity) => ()
    _ => fail("expected completion epoch overflow")
  } noraise {
    _ => fail("exhausted completion reset succeeded")
  }
  assert_eq(completion.state(), Consumed)
  assert_eq(completion.epoch(), 0xffffffffffffffffUL)
}
