///|
fn drive_api_auth_verification(
  workspace : LunaApiAuthVerifierWorkspace,
  source : FixedArray[Byte],
  declared_length : Int,
  offered_bytes : Int,
) -> (LunaApiAuthDecision, UInt64) raise {
  let write = workspace.begin_verify(declared_length)
  for index in 0..<offered_bytes {
    let step = write.push_byte(source[index])
    assert_eq(step.work_units(), 1)
    assert_eq(step.is_complete(), index + 1 == offered_bytes)
    assert_eq(step.is_accepting(), index + 1 < offered_bytes)
  }
  let work = write.finish()
  while work.state() == LunaApiAuthVerificationPending {
    assert_eq(work.progress(), LunaApiAuthVerificationAdvanced)
    assert_true(work.last_work_units() > 0)
  }
  let total = work.total_work_units()
  (work.take_decision(), total)
}

///|
test "cooperative Luna API authentication is budget invariant" {
  let expected : FixedArray[Byte] = [b'a', b'b', b'c', b'd']
  let policy = LunaApiAuthPolicy::new(
    expected,
    source_offset=0,
    length=4,
    maximum_credential_bytes=8,
  )
  for budget in [1, 17, 65536] {
    let workspace = LunaApiAuthVerifierWorkspace::new(
      policy,
      LunaApiAuthStepBudget::new(budget),
    )
    let (accepted, accepted_total) = drive_api_auth_verification(
      workspace, expected, 4, 4,
    )
    assert_true(accepted.is_accepted())
    assert_eq(accepted_total, 5UL)
    let empty : FixedArray[Byte] = []
    let (empty_rejected, empty_total) = drive_api_auth_verification(
      workspace, empty, 0, 0,
    )
    assert_true(empty_rejected.is_rejected())
    assert_eq(empty_total, 5UL)
    let long : FixedArray[Byte] = [
      b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i',
    ]
    let (long_rejected, long_total) = drive_api_auth_verification(
      workspace, long, 9, 4,
    )
    assert_true(long_rejected.is_rejected())
    assert_eq(long_total, 5UL)
  }
}

///|
test "cooperative Luna API authentication writer is transactional" {
  let expected : FixedArray[Byte] = [b'a', b'b', b'c']
  let policy = LunaApiAuthPolicy::new(
    expected,
    source_offset=0,
    length=3,
    maximum_credential_bytes=4,
  )
  let workspace = LunaApiAuthVerifierWorkspace::new(
    policy,
    LunaApiAuthStepBudget::new(1),
  )
  let write = workspace.begin_verify(3)
  ignore(write.push_byte(b'a'))
  try write.finish() catch {
    LunaApiAuthFailed(rule=LunaApiAuthLifecycle, issue=LunaApiAuthNotReady) =>
      ()
    _ => fail("unexpected incomplete writer failure")
  } noraise {
    _ => fail("incomplete writer finished")
  }
  ignore(write.push_byte(b'b'))
  let final_push = write.push_byte(b'c')
  assert_true(final_push.is_complete())
  try write.push_byte(b'x') catch {
    LunaApiAuthFailed(rule=LunaApiAuthLifecycle, issue=LunaApiAuthNotReady) =>
      ()
    _ => fail("unexpected extra-byte failure")
  } noraise {
    _ => fail("extra byte was accepted")
  }
  let work = write.finish()
  try workspace.begin_verify(3) catch {
    LunaApiAuthFailed(rule=LunaApiAuthLifecycle, issue=LunaApiAuthBusy) => ()
    _ => fail("unexpected busy failure")
  } noraise {
    _ => fail("busy workspace admitted another writer")
  }
  while work.state() == LunaApiAuthVerificationPending {
    ignore(work.progress())
  }
  assert_true(work.take_decision().is_accepted())
  try work.state() catch {
    LunaApiAuthFailed(rule=LunaApiAuthEpoch, issue=LunaApiAuthStale) => ()
    _ => fail("unexpected stale work failure")
  } noraise {
    _ => fail("taken work remained live")
  }
  let aborted = workspace.begin_verify(3)
  aborted.abort()
  try aborted.push_byte(b'a') catch {
    LunaApiAuthFailed(rule=LunaApiAuthEpoch, issue=LunaApiAuthStale) => ()
    _ => fail("unexpected stale writer failure")
  } noraise {
    _ => fail("aborted writer remained live")
  }
  let reused = workspace.begin_verify(3)
  reused.abort()
}

///|
test "cooperative Luna API authentication validates budgets and lengths" {
  for invalid in [0, -1, 65537] {
    try LunaApiAuthStepBudget::new(invalid) catch {
      LunaApiAuthFailed(rule=LunaApiAuthWorkUnits, issue=LunaApiAuthOutOfRange) =>
        ()
      _ => fail("unexpected step-budget failure")
    } noraise {
      _ => fail("invalid step budget was accepted")
    }
  }
  let expected : FixedArray[Byte] = [b'a']
  let policy = LunaApiAuthPolicy::new(
    expected,
    source_offset=0,
    length=1,
    maximum_credential_bytes=1,
  )
  let workspace = LunaApiAuthVerifierWorkspace::new(
    policy,
    LunaApiAuthStepBudget::new(1),
  )
  try workspace.begin_verify(-1) catch {
    LunaApiAuthFailed(rule=LunaApiAuthCredential, issue=LunaApiAuthOutOfRange) =>
      ()
    _ => fail("unexpected declared-length failure")
  } noraise {
    _ => fail("negative declared length was accepted")
  }
}
