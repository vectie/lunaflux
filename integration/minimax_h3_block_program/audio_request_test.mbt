///|
test "unprepared audio request rejects work and closes idempotently" {
  let request = @minimax_h3_block_program.AudioDecodeRequest::new()
  try request.submit() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared submission accepted")
  }
  try request.poll() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared completion accepted")
  }
  let mut exposed = false
  try request.with_output(_ => exposed = true) catch {
    _ => ()
  } noraise {
    _ => fail("unprepared output exposed")
  }
  assert_false(exposed)
  request.cancel()
  request.cancel()
  request.close()
  request.close()
  try request.submit() catch {
    _ => ()
  } noraise {
    _ => fail("closed request replayed")
  }
}
