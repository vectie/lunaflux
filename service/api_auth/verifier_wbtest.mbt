///|
test "cooperative Luna API authentication failure remains abortable" {
  let invalid_policy : LunaApiAuthPolicy = {
    expected: [b'a'],
    expected_length: 0,
    maximum_credential_bytes: 1,
    closed: false,
  }
  let workspace : LunaApiAuthVerifierWorkspace = {
    policy: invalid_policy,
    budget: LunaApiAuthStepBudget::new(1),
    epoch: 1UL,
    phase: LUNA_API_AUTH_PADDING,
    required_input_bytes: 0,
    cursor: 0,
    difference: 1,
    failure_rule: 0,
    failure_issue: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
  let work : LunaApiAuthVerificationWork = { workspace, epoch: 1UL }
  assert_eq(work.progress(), LunaApiAuthVerificationAdvanced)
  assert_eq(work.state(), LunaApiAuthVerificationFailed)
  match work.failure() {
    LunaApiAuthFailed(rule=LunaApiAuthPolicy, issue=LunaApiAuthInvalid) => ()
    _ => fail("unexpected cooperative failure evidence")
  }
  work.abort()
  try work.state() catch {
    LunaApiAuthFailed(rule=LunaApiAuthEpoch, issue=LunaApiAuthStale) => ()
    _ => fail("unexpected aborted failure alias")
  } noraise {
    _ => fail("aborted failed work remained live")
  }
}

///|
test "cooperative Luna API authentication epoch exhaustion is pre-mutation" {
  let policy = LunaApiAuthPolicy::new(
    [b'a'],
    source_offset=0,
    length=1,
    maximum_credential_bytes=1,
  )
  let workspace = LunaApiAuthVerifierWorkspace::new(
    policy,
    LunaApiAuthStepBudget::new(1),
  )
  workspace.epoch = 0xffffffffffffffffUL
  try workspace.begin_verify(1) catch {
    LunaApiAuthFailed(rule=LunaApiAuthEpoch, issue=LunaApiAuthExhausted) => ()
    _ => fail("unexpected epoch exhaustion failure")
  } noraise {
    _ => fail("exhausted workspace admitted a writer")
  }
  assert_eq(workspace.phase, LUNA_API_AUTH_IDLE)
}

///|
test "cooperative authentication close makes outstanding epochs stale" {
  let policy = LunaApiAuthPolicy::new(
    [b'a', b'b'],
    source_offset=0,
    length=2,
    maximum_credential_bytes=4,
  )
  let workspace = LunaApiAuthVerifierWorkspace::new(
    policy,
    LunaApiAuthStepBudget::new(1),
  )
  let write = workspace.begin_verify(2)
  workspace.close()
  workspace.close()
  try write.push_byte(b'a') catch {
    LunaApiAuthFailed(rule=LunaApiAuthEpoch, issue=LunaApiAuthStale) => ()
    _ => fail("unexpected closed writer rejection")
  } noraise {
    _ => fail("stale writer survived workspace close")
  }
  try workspace.begin_verify(2) catch {
    LunaApiAuthFailed(rule=LunaApiAuthLifecycle, issue=LunaApiAuthBusy) => ()
    _ => fail("unexpected closed workspace rejection")
  } noraise {
    _ => fail("closed workspace restarted")
  }
}
