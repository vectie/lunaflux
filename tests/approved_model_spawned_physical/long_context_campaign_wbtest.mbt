///|
test "actual long context requests pin the independent 8 and 9 token referee" {
  ignore(long_context_validation_contract(0))
  let frames = [
    long_context_frame(1201UL, prefix_campaign_first_input()),
    long_context_frame(1202UL, prefix_campaign_second_input()),
  ]
  let expected_inputs = [
    prefix_campaign_first_input(),
    prefix_campaign_second_input(),
  ]
  for index, frame in frames {
    let request = @framed_wire.RequestFrameBuffer::new(
        @framed_wire.FramedWireLimits::new(
          prefix_campaign_inference_limits(),
          max_frame_bytes=1048576,
        ),
      )
      .load(frame, frame.length())
      .request()
    guard request.input() is TokenIds(tokens) else {
      abort("long context request widened beyond token input")
    }
    assert_eq(tokens.length(), expected_inputs[index].length())
    for token_index, token in expected_inputs[index] {
      assert_eq(tokens.token_at(token_index), token)
    }
    assert_eq(request.max_new_tokens(), 1)
    assert_eq(request.context_ceiling(), 256)
    assert_eq(request.cache().permission(), Disabled)
  }
  assert_eq(PREFIX_REFEREE_FIRST_TOKEN, 1355)
  assert_eq(PREFIX_REFEREE_SECOND_TOKEN, 1240)
  assert_eq(
    PREFIX_REFEREE_SHA256,
    "765d4b80750e377fea131ce140a0a67931724ab06c24dc5e322c7e6b295ab8e3",
  )
}

///|
test "actual long context evidence excludes unadmitted context lengths" {
  let scope = "campaign_scope=qualification-only\n" +
    "actual_input_token_lengths=8,9\n" +
    "actual_context_tokens_including_output=9,10\n" +
    "beyond_9_token_input=not-established\n" +
    "128_or_255_token_execution=not-admitted\n" +
    "full_256_token_context=not-established\n" +
    "cache_permission=disabled\n" +
    "prefix_reuse=not-exercised\n"
  assert_true(scope.contains("actual_input_token_lengths=8,9"))
  assert_true(scope.contains("128_or_255_token_execution=not-admitted"))
  assert_false(scope.contains("full_256_token_context=pass"))
}
