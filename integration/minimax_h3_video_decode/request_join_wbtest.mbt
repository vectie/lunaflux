///|
test "decoder pipeline join rejects unprepared consumed cancelled and failed stages" {
  for
    phase in (
      [Empty, Submitted, Complete, Cancelled, Failed, Closed] :
      Array[VideoRequestPhase]) {
    assert_false(phase.permits_join())
  }
  assert_true(VideoRequestPhase::Prepared.permits_join())
  let request = VideoDecodeRequest::new()
  assert_true(request.input_receipt is None)
  request.cancel()
  assert_false(request.phase.permits_join())
  request.close()
  assert_true(request.input_receipt is None)
}
