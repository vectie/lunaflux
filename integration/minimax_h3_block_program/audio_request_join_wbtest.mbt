///|
test "audio joint-request receipt cannot be inferred from decoder completion or raw readiness" {
  for
    phase in (
      [Empty, Submitted, Complete, Cancelled, Failed, Closed] :
      Array[AudioRequestPhase]) {
    let state : AudioRequestState = { phase, }
    assert_false(state.permits_join())
  }
  let prepared : AudioRequestState = { phase: Prepared, }
  assert_true(prepared.permits_join())
  let request = AudioDecodeRequest::new()
  assert_true(request.input_receipt is None)
  assert_true(request.input_program is None)
  request.cancel()
  assert_false(request.state.permits_join())
  request.close()
  assert_true(request.input_receipt is None)
}
