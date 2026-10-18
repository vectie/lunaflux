///|
test "finite oversubscribed requests finish after recompute preemption" {
  let owner = wb_owner_with_capacity(1, 2)
  ignore(owner.admit(wb_request(1801UL, max_new_tokens=4), 0UL))
  ignore(owner.admit(wb_request(1802UL, max_new_tokens=4), 0UL))
  let mut emitted = 0
  let mut retired = 0
  let mut evictions = 0
  for _ in 0..<80 {
    if owner.live_request_count() == 0 {
      break
    }
    let outcome = owner.build_next()
    if outcome.is_preempted() {
      evictions += 1
      continue
    }
    if !outcome.is_submitted() {
      continue
    }
    let plan = owner.submitted_plan(outcome)
    let writer = owner.completion_writer(plan)
    for i in 0..<plan.prefill_count() {
      let row = plan.prefill_row(i)
      if row.produces_token() {
        writer.append_final_prefill_completed(
          slot=row.completion_slot(),
          request_id=row.request_id(),
          request_generation=row.request_generation(),
          processed_tokens=row.tokens().length(),
          token_id=7,
        )
      } else {
        writer.append_prefill_completed(
          slot=row.completion_slot(),
          request_id=row.request_id(),
          request_generation=row.request_generation(),
          processed_tokens=row.tokens().length(),
        )
      }
    }
    for i in 0..<plan.decode_count() {
      let row = plan.decode_row(i)
      writer.append_decode_completed(
        slot=row.completion_slot(),
        request_id=row.request_id(),
        request_generation=row.request_generation(),
        token_id=7,
      )
    }
    ignore(owner.complete(plan, writer.submit()))
    while owner.generated_token_count() > 0 {
      ignore(owner.take_generated_token())
      emitted += 1
    }
    while owner.terminal_notice_count() > 0 {
      ignore(owner.take_terminal_notice())
      retired += 1
    }
  }
  assert_true(evictions > 0)
  assert_eq(emitted, 8)
  assert_eq(retired, 2)
  assert_eq(owner.live_request_count(), 0)
}
