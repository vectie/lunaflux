///|
fn verification_frame() -> @serial_decoder_frame.SerialDecoderFrame raise {
  @serial_decoder_frame.SerialDecoderFrame::new(
    maximum_rows=4,
    maximum_context=12,
    vocabulary_size=16,
    model_generation=9UL,
    with_positions=true,
    all_position_outputs=true,
  )
}

///|
test "verification selects every live row without changing ordinary last-row output" {
  let frame = verification_frame()
  defer frame.close()
  assert_eq(frame.output_capacity(), 4)
  assert_eq(frame.input_capacity(7), 16)
  let (_, prompt, writer) = request_plan()
  frame.stage(prompt, writer)
  assert_eq(frame.output_rows(), 0)
  frame.retire_stage(execution_failed=false)
  assert_eq(frame.processed_tokens(), 2)
  let (_, verify, verify_writer) = request_plan(
    position=2,
    tokens=[4, 5, 6],
    produces=true,
  )
  frame.stage(verify, verify_writer)
  assert_eq(frame.output_rows(), 3)
  assert_eq(port_value(frame, 6), 3)
  for row in 0..<3 {
    assert_eq(port_value(frame, 7, index=row), row)
    assert_eq(port_value(frame, 8, index=row), row + 2)
  }
  let copied = FixedArray::make(16, b'\xff')
  assert_eq(frame.copy_input(7, copied), 12)
  assert_eq(copied[12], b'\xff')
  try frame.complete(sampled_token=7, execution_failed=false) catch {
    @serial_decoder_frame.SerialDecoderError::UnsupportedBatch => ()
    error => raise error
  } noraise {
    _ => fail("all-position samples were collapsed to one canonical token")
  }
  assert_false(frame.is_faulted())
  frame.discard_stage()
  assert_eq(frame.processed_tokens(), 2)
  try verify_writer.submit() catch {
    _ => ()
  } noraise {
    _ => fail("discard retained the writer lease")
  }
  // Rejected suffix did not commit: replay only the accepted two input rows.
  let (_, replay, replay_writer) = request_plan(
    position=2,
    tokens=[4, 5],
    produces=false,
  )
  frame.stage(replay, replay_writer)
  assert_eq(port_value(frame, 5), 0)
  frame.retire_stage(execution_failed=false)
  assert_eq(frame.processed_tokens(), 4)
  let ordinary = host_frame()
  defer ordinary.close()
  assert_eq(ordinary.output_capacity(), 1)
  let (_, ordinary_plan, ordinary_writer) = request_plan(produces=true)
  ordinary.stage(ordinary_plan, ordinary_writer)
  assert_eq(ordinary.output_rows(), 1)
  assert_eq(port_value(ordinary, 7), 1)
}

///|
test "discard never invents a committed request and cannot revive an execution failure" {
  let frame = verification_frame()
  defer frame.close()
  let (_, prompt, writer) = request_plan()
  frame.stage(prompt, writer)
  frame.discard_stage()
  assert_eq(frame.processed_tokens(), 0)
  let (_, foreign, foreign_writer) = request_plan(id=8UL)
  frame.stage(foreign, foreign_writer)
  assert_eq(port_value(frame, 5), 1)
  frame.retire_stage(execution_failed=true)
  assert_true(frame.is_faulted())
  try frame.discard_stage() catch {
    @serial_decoder_frame.SerialDecoderError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("failed history became reusable")
  }
}
