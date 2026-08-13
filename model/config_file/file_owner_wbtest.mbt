///|
test "terminal close failure prevents successful body publication" {
  try
    finish_file_body(Ok(7), () => {
      raise FileIo(operation=CloseFile, issue=Failed)
    })
  catch {
    FileIo(operation=CloseFile, issue=Failed) => ()
    _ => fail("unexpected successful-body close failure")
  } noraise {
    _ => fail("successful body escaped a terminal close failure")
  }
}

///|
test "terminal close failure preserves an earlier body failure" {
  let closed = Ref(false)
  let outcome : Result[Int, ModelConfigFileError] = Err(DigestMismatch)
  try
    finish_file_body(outcome, () => {
      closed.val = true
      raise FileIo(operation=CloseFile, issue=Failed)
    })
  catch {
    DigestMismatch => ()
    _ => fail("terminal close replaced the primary body failure")
  } noraise {
    _ => fail("failed body unexpectedly published")
  }
  assert_true(closed.val)
}
