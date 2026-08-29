///|
test "terminal descriptor close failure prevents successful publication" {
  try
    finish_descriptor_body(Ok(7), () => raise FileOperationFailed(Close))
  catch {
    FileOperationFailed(Close) => ()
    _ => fail("unexpected descriptor close failure")
  } noraise {
    _ => fail("successful descriptor body escaped failed close")
  }
}

///|
test "terminal descriptor close preserves an earlier primary failure" {
  let close_called = Ref(false)
  let outcome : Result[Int, RuntimeDescriptorFileError] = Err(DigestMismatch)
  try
    finish_descriptor_body(outcome, () => {
      close_called.val = true
      raise FileOperationFailed(Close)
    })
  catch {
    DigestMismatch => ()
    _ => fail("descriptor close replaced primary failure")
  } noraise {
    _ => fail("failed descriptor body unexpectedly published")
  }
  assert_true(close_called.val)
}
