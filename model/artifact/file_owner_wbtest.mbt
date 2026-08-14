///|
test "terminal close failure prevents successful snapshot publication" {
  try
    finish_file_body(Ok(7), () => {
      raise FileIo(file=ModelConfig, operation=CloseFile, issue=Failed)
    })
  catch {
    FileIo(file=ModelConfig, operation=CloseFile, issue=Failed) => ()
    _ => fail("unexpected successful-body close failure")
  } noraise {
    _ => fail("successful snapshot escaped terminal close failure")
  }
}

///|
test "terminal close failure preserves the earlier snapshot failure" {
  let close_attempted = Ref(false)
  let outcome : Result[Int, ArtifactError] = Err(DigestMismatch(Tokenizer))
  try
    finish_file_body(outcome, () => {
      close_attempted.val = true
      raise FileIo(file=Tokenizer, operation=CloseFile, issue=Failed)
    })
  catch {
    DigestMismatch(Tokenizer) => ()
    _ => fail("terminal close replaced the primary artifact failure")
  } noraise {
    _ => fail("failed snapshot unexpectedly published")
  }
  assert_true(close_attempted.val)
}

///|
test "snapshot mutation truncation and aggregate limits map by file" {
  assert_eq(snapshot_error(Truncated, Weights), FileTruncated(Weights))
  assert_eq(
    snapshot_error(ChangedDuringRead, Tokenizer),
    FileChangedDuringRead(Tokenizer),
  )
  assert_eq(
    snapshot_error(FileTooLarge, ModelConfig),
    FileChangedDuringRead(ModelConfig),
  )
}
