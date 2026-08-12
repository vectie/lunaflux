///|
test "completion writer is bound to exact scheduler plan owner" {
  let owner = scheduler()
  ignore(owner.admit(request(301UL), 0UL))
  let plan = next_plan(owner)
  let writer = owner.completion_writer(plan)
  let row = try! plan.prefill_row(0)
  try! writer.append_final_prefill_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    processed_tokens=try! row.tokens().length(),
    token_id=7,
  )
  let other = scheduler()
  ignore(other.admit(request(302UL), 0UL))
  let foreign_plan = next_plan(other)
  try owner.completion_writer(foreign_plan) catch {
    WorkerProtocol(_) => ()
    _ => fail("expected same-sequence foreign owner rejection")
  } noraise {
    _ => fail("foreign owner with matching sequence issued a writer")
  }
  try other.completion_writer(plan) catch {
    WorkerProtocol(_) => ()
    _ => fail("expected foreign plan owner rejection")
  } noraise {
    _ => fail("foreign scheduler issued a writer")
  }
  try! writer.abort()
}
