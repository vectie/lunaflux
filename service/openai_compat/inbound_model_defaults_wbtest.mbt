///|
test "document prefix is once per conversation and never added to raw input" {
  let inference = openai_wb_fixture_limits()
  for budget in [1, 17, 65536] {
    let body = "{\"model\":\"model-x\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\"},{\"role\":\"user\",\"content\":\"c\"}]}"
    let (frame, _) = inbound_encode(
      inbound_workspace(inference, budget, document_prefix=inbound_bytes("BOS")),
      inbound_http_view(2, body),
    )
    assert_eq(inbound_text(frame, inference), "BOS<U>a</U><A>b</A><U>c</U><A>")
    let (raw, _) = inbound_encode(
      inbound_workspace(inference, budget, document_prefix=inbound_bytes("BOS")),
      inbound_http_view(1, "{\"model\":\"model-x\",\"input\":\"a\"}"),
    )
    assert_eq(inbound_text(raw, inference), "a")
  }
}

///|
test "required EOS unions with explicit stops before typed handoff count" {
  let inference = openai_wb_fixture_limits()
  for
    (field, expected) in [
      ("", [2, 7]),
      (",\"stop_token_ids\":[]", [2, 7]),
      (",\"stop_token_ids\":[7,3]", [7, 3, 2]),
    ] {
    for budget in [1, 17, 65536] {
      let (frame, _) = inbound_encode(
        inbound_workspace(inference, budget, required_stop_tokens=[2, 7]),
        inbound_http_view(
          1,
          "{\"model\":\"model-x\",\"input\":\"a\"" + field + "}",
        ),
      )
      let request = @framed_wire.RequestFrameBuffer::new(
          @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
        )
        .load(frame, frame.length())
        .request()
      let ids = request.stops().token_ids()
      assert_eq(Array::makei(ids.length(), i => ids[i]), expected)
    }
  }
}
