///|
test "policy close failure prevents successful publication" {
  try
    finish_policy_file(Ok(b"exact"), () => raise FileOperationFailed(Close))
  catch {
    FileOperationFailed(Close) => ()
    _ => fail("unexpected policy terminal-close failure")
  } noraise {
    _ => fail("policy bytes escaped a failed terminal close")
  }
}

///|
test "policy close preserves an earlier primary failure" {
  let closed = Ref(false)
  try
    finish_policy_file(Err(DigestMismatch), () => {
      closed.val = true
      raise FileOperationFailed(Close)
    })
  catch {
    DigestMismatch => ()
    _ => fail("policy close replaced the primary failure")
  } noraise {
    _ => fail("failed policy snapshot unexpectedly published")
  }
  assert_true(closed.val)
}
