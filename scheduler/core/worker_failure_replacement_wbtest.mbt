///|
fn wb_assert_terminal_failure_replacement(
  request_id : UInt64,
  exhaust_plan : Bool,
  exhaust_completion : Bool,
  backpressure_first : Bool,
) -> Unit raise {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(request_id), 0UL)
  let plan = wb_next_plan(owner)
  ignore(wb_final_completion(owner, plan, 7))
  if backpressure_first {
    owner.terminal_count = owner.terminal_request_ids.length()
    try owner.preflight_exhausted_failure_retirement(plan, 0) catch {
      Invalid(field=TerminalNoticeCapacity, issue=Backpressured) => ()
      _ => fail("unexpected exhausted replacement backpressure error")
    } noraise {
      _ => fail("exhausted replacement ignored terminal backpressure")
    }
    assert_eq(owner.completion_a.state(), CompletionSubmitted)
    assert_eq(owner.plan_a.state(), Submitted)
    assert_true(owner.status(handle.request_id()).unwrap().in_flight())
    owner.terminal_count = 0
    owner.terminal_head = 0
    owner.terminal_tail = 0
  }
  let rows = owner.preflight_exhausted_failure_retirement(plan, 0)
  owner.commit_exhausted_failure_retirement(plan, 0, rows)
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.reason(), WorkerFailed)
  assert_eq(terminal.request_id(), handle.request_id())
  assert_eq(terminal.request_generation(), handle.request_generation())
  assert_true(owner.status(handle.request_id()) is None)
  assert_eq(owner.completion_a.state(), Consumed)
  assert_eq(owner.plan_a.state(), Retired)
  assert_true(
    requires_terminal_failure_replacement(
      if exhaust_completion {
        0xfffffffffffffffeUL
      } else {
        1UL
      },
      if exhaust_plan {
        0xffffffffffffffffUL
      } else {
        1UL
      },
    ),
  )
  ignore(owner.admit(wb_request(request_id + 100UL), 0UL))
  match owner.build_next() {
    BuildSubmitted(side, _) => assert_eq(side, 1)
    _ => fail("healthy B side was not reusable after terminal A retirement")
  }
}

///|
fn wb_assert_stale_cut_terminal_replacement(
  request_id : UInt64,
  deadline : Bool,
) -> Unit raise {
  let owner = wb_owner()
  let handle = owner.admit(
    wb_request(request_id, deadline=if deadline { 10UL } else { 100UL }),
    0UL,
  )
  let plan = wb_next_plan(owner)
  ignore(wb_final_completion(owner, plan, 7))
  if deadline {
    assert_eq(owner.expire_deadlines(10UL), 1)
  } else {
    owner.cancel(handle)
  }
  assert_eq(owner.terminal_notice_count(), 1)
  let rows = owner.preflight_exhausted_failure_retirement(plan, 0)
  owner.commit_exhausted_failure_retirement(plan, 0, rows)
  assert_eq(owner.terminal_notice_count(), 1)
  let terminal = owner.take_terminal_notice()
  assert_eq(terminal.request_id(), handle.request_id())
  assert_eq(
    terminal.reason(),
    if deadline {
      DeadlineReached
    } else {
      CancelledByCaller
    },
  )
  assert_eq(owner.free_page_count(), 8)
  assert_eq(owner.allocated_block_table_count(), 0)
}

///|
test "terminal replacement preserves stale caller-cancel terminal" {
  wb_assert_stale_cut_terminal_replacement(713UL, false)
}

///|
test "terminal replacement preserves stale deadline terminal" {
  wb_assert_stale_cut_terminal_replacement(714UL, true)
}

///|
test "exhausted invalid completion retires terminally" {
  wb_assert_terminal_failure_replacement(704UL, false, true, false)
}

///|
test "exhausted invalid plan retires terminally" {
  wb_assert_terminal_failure_replacement(705UL, true, false, false)
}

///|
test "both exhausted protocol owners retire terminally after backpressure" {
  wb_assert_terminal_failure_replacement(706UL, true, true, true)
}

///|
test "failure replacement threshold includes terminal completion epochs" {
  assert_false(requires_terminal_failure_replacement(0xfffffffffffffffdUL, 1UL))
  assert_true(requires_terminal_failure_replacement(0xfffffffffffffffeUL, 1UL))
  assert_true(requires_terminal_failure_replacement(0xffffffffffffffffUL, 1UL))
  assert_true(requires_terminal_failure_replacement(1UL, 0xffffffffffffffffUL))
}

///|
test "two terminal protocol pairs report exhaustion instead of backpressure" {
  let owner = wb_owner()
  ignore(owner.admit(wb_request(707UL), 0UL))
  let plan_a = wb_next_plan(owner)
  ignore(wb_final_completion(owner, plan_a, 7))
  let rows_a = owner.preflight_exhausted_failure_retirement(plan_a, 0)
  owner.commit_exhausted_failure_retirement(plan_a, 0, rows_a)

  ignore(owner.admit(wb_request(708UL), 0UL))
  let outcome_b = owner.build_next()
  let plan_b = owner.submitted_plan(outcome_b)
  ignore(wb_final_completion(owner, plan_b, 8))
  let rows_b = owner.preflight_exhausted_failure_retirement(plan_b, 1)
  owner.commit_exhausted_failure_retirement(plan_b, 1, rows_b)

  ignore(owner.admit(wb_request(709UL), 0UL))
  try owner.build_next() catch {
    Invalid(field=Plan, issue=Exhausted) => ()
    _ => fail("unexpected terminal-pair exhaustion error")
  } noraise {
    _ => fail("terminal protocol pairs reported transient availability")
  }
}

///|
test "terminal replacement authenticates plan row before consuming owners" {
  let owner = wb_owner()
  ignore(owner.admit(wb_request(710UL), 0UL))
  let plan = wb_next_plan(owner)
  ignore(wb_final_completion(owner, plan, 7))
  let slot = owner.plan_a_slots[0]
  owner.requests[slot] = Some(wb_request(711UL))
  try owner.preflight_exhausted_failure_retirement(plan, 0) catch {
    Invalid(field=Completion, issue=Mismatch) => ()
    _ => fail("unexpected terminal replacement identity error")
  } noraise {
    _ => fail("terminal replacement accepted a foreign request row")
  }
  assert_eq(owner.completion_a.state(), CompletionSubmitted)
  assert_eq(owner.plan_a.state(), Submitted)
}

///|
test "terminal replacement retires an active decode row" {
  let owner = wb_owner()
  let handle = owner.admit(wb_request(712UL), 0UL)
  let prefill = wb_next_plan(owner)
  owner.complete(prefill, wb_final_completion(owner, prefill, 7))
  ignore(owner.take_generated_token())
  let decode = wb_next_plan(owner)
  let row = try! decode.decode_row(0)
  let writer = try! owner.completion_writer(decode)
  try! writer.append_decode_completed(
    slot=try! row.completion_slot(),
    request_id=try! row.request_id(),
    request_generation=try! row.request_generation(),
    token_id=8,
  )
  ignore(try! writer.submit())
  let rows = owner.preflight_exhausted_failure_retirement(decode, 0)
  owner.commit_exhausted_failure_retirement(decode, 0, rows)
  assert_eq(owner.take_terminal_notice().reason(), WorkerFailed)
  assert_true(owner.status(handle.request_id()) is None)
  assert_eq(owner.completion_a.state(), Consumed)
  assert_eq(owner.plan_a.state(), Retired)
}
