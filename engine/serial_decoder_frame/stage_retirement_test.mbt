///|
test "partial stage retirement commits position without publishing model completion" {
  let frame = host_frame()
  let (_, first, first_writer) = request_plan(produces=true)
  frame.stage(first, first_writer)
  frame.retire_stage(execution_failed=false)
  assert_eq(frame.processed_tokens(), 2)
  try frame.completion() catch {
    @wire.WorkerWireError::StaleFrame => ()
    error => raise error
  } noraise {
    _ => fail("partial stage published a model completion")
  }
  try first_writer.submit() catch {
    _ => ()
  } noraise {
    _ => fail("partial stage left a response writer open")
  }
  let (_, second, second_writer) = request_plan(decode=true, position=2, tokens=[
    7,
  ])
  frame.stage(second, second_writer)
  assert_eq(port_value(frame, 4), 0)
  frame.retire_stage(execution_failed=false)
  assert_eq(frame.processed_tokens(), 3)
  frame.release_request()
  let (_, restart, restart_writer) = request_plan(id=8UL)
  frame.stage(restart, restart_writer)
  assert_eq(port_value(frame, 4), 1)
  frame.retire_stage(execution_failed=true)
  assert_true(frame.is_faulted())
  try frame.release_request() catch {
    @serial_decoder_frame.SerialDecoderError::Faulted => ()
    error => raise error
  } noraise {
    _ => fail("reused peer-failed retained history")
  }
  frame.close()
}
