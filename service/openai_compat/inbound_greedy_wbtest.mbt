///|
test "zero temperature explicitly maps to greedy independent of filter order" {
  let inference = openai_wb_fixture_limits()
  let workspace = inbound_workspace(inference, 1)
  for
    fields in [
      "\"temperature\":0,\"top_k\":3,\"top_p\":0.8", "\"top_k\":3,\"top_p\":0.8,\"temperature\":0",
      "\"temperature\":0.0", "\"temperature\":0e0",
    ] {
    let (frame, _) = inbound_encode(
      workspace,
      inbound_http_view(
        1,
        "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true," + fields + "}",
      ),
    )
    let sampling = @framed_wire.RequestFrameBuffer::new(
        @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
      )
      .load(frame, frame.length())
      .request()
      .sampling()
    assert_eq(sampling.mode(), Greedy)
    assert_eq(sampling.top_k(), None)
    assert_eq(sampling.top_p(), None)
  }
}

///|
test "disabled top k normalizes to absent while nonzero temperature remains sampled" {
  let inference = openai_wb_fixture_limits()
  for top_k in [0, -1] {
    let (frame, _) = inbound_encode(
      inbound_workspace(inference, 17),
      inbound_http_view(
        1,
        "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"temperature\":0.5,\"top_k\":\{top_k}}",
      ),
    )
    let sampling = @framed_wire.RequestFrameBuffer::new(
        @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
      )
      .load(frame, frame.length())
      .request()
      .sampling()
    assert_eq(sampling.mode(), Sample)
    assert_eq(sampling.temperature(), 0.5)
    assert_eq(sampling.top_k(), None)
  }
  for
    fields in [
      "\"temperature\":0,\"top_k\":-2", "\"temperature\":0,\"top_k\":4294967297",
      "\"temperature\":0,\"top_p\":0", "\"temperature\":-0.1",
    ] {
    inbound_expect_json_failure(
      1,
      "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true," + fields + "}",
    )
  }
}
