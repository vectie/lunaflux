///|
test "absolute positions follow chunked prefill and committed decode history" {
  let host = @serial_decoder_frame.SerialDecoderFrame::new(
    maximum_rows=4,
    maximum_context=12,
    vocabulary_size=16,
    model_generation=9UL,
    with_positions=true,
  )
  assert_eq(host.input_count(), 9)
  assert_eq(host.input_capacity(8), 16)
  let (submitted, frame, writer) = request_plan()
  host.stage(frame, writer)
  assert_eq(port_value(host, 8), 0)
  assert_eq(port_value(host, 8, index=1), 1)
  host
  .complete(sampled_token=-1, execution_failed=false)
  .validate_for(submitted)
  let (submitted, frame, writer) = request_plan(
    position=2,
    tokens=[4, 5, 6],
    produces=true,
  )
  host.stage(frame, writer)
  assert_eq(port_value(host, 8), 2)
  assert_eq(port_value(host, 8, index=1), 3)
  assert_eq(port_value(host, 8, index=2), 4)
  host.complete(sampled_token=7, execution_failed=false).validate_for(submitted)
  let (submitted, frame, writer) = request_plan(
    position=5,
    tokens=[7],
    decode=true,
  )
  host.stage(frame, writer)
  let bytes = FixedArray::make(16, b'\xff')
  assert_eq(host.copy_input(8, bytes), 4)
  assert_eq(bytes[0], b'\x05')
  for i in 4..<16 {
    assert_eq(bytes[i], b'\xff')
  }
  host.complete(sampled_token=8, execution_failed=false).validate_for(submitted)
  host.release_request()
  let (_, frame, writer) = request_plan(id=8UL)
  host.stage(frame, writer)
  assert_eq(port_value(host, 8), 0)
  host.close()
}

///|
test "legacy frame retains eight-port ABI and rejects optional position access" {
  let host = host_frame()
  assert_eq(host.input_count(), 8)
  try host.input_capacity(8) catch {
    @serial_decoder_frame.SerialDecoderError::InvalidCapacity => ()
    e => raise e
  } noraise {
    _ => fail("legacy ABI unexpectedly exposed a ninth port")
  }
  host.close()
}
