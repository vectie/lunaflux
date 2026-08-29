///|
fn decode_prefix_campaign_frame(
  frame : FixedArray[Byte],
) -> @inference.GenerateRequest raise {
  @framed_wire.RequestFrameBuffer::new(
    @framed_wire.FramedWireLimits::new(
      prefix_campaign_inference_limits(),
      max_frame_bytes=1048576,
    ),
  )
  .load(frame, frame.length())
  .request()
}

///|
fn assert_prefix_campaign_input(
  request : @inference.GenerateRequest,
  expected : Array[Int],
) -> Unit raise {
  guard request.input() is TokenIds(tokens) else {
    abort("prefix campaign request did not retain token input")
  }
  assert_eq(tokens.length(), expected.length())
  for index, token in expected {
    assert_eq(tokens.token_at(index), token)
  }
}

///|
test "prefix campaign frames pin authenticated writer then reader geometry" {
  let first = decode_prefix_campaign_frame(prefix_campaign_first_frame())
  let second = decode_prefix_campaign_frame(prefix_campaign_second_frame())
  assert_eq(first.model_identity(), pinned_tiny_model_identity())
  assert_eq(second.model_identity(), pinned_tiny_model_identity())
  assert_eq(first.request_id().value(), PREFIX_CAMPAIGN_FIRST_REQUEST_ID)
  assert_eq(second.request_id().value(), PREFIX_CAMPAIGN_SECOND_REQUEST_ID)
  assert_true(first.request_id() != second.request_id())
  assert_prefix_campaign_input(first, prefix_campaign_first_input())
  assert_prefix_campaign_input(second, prefix_campaign_second_input())
  assert_eq(first.cache().permission(), ReadWrite)
  assert_eq(second.cache().permission(), ReadOnly)
  assert_eq(
    first.cache().scope().as_string(),
    second.cache().scope().as_string(),
  )
  assert_eq(first.cache().scope().as_string(), PREFIX_CAMPAIGN_SCOPE)
  assert_eq(first.max_new_tokens(), 1)
  assert_eq(second.max_new_tokens(), 1)
  assert_eq(first.context_ceiling(), CAMPAIGN_CONTEXT_TOKENS)
  assert_eq(second.context_ceiling(), CAMPAIGN_CONTEXT_TOKENS)
  assert_eq(prefix_campaign_inference_limits().max_input_tokens(), 9)
  assert_eq(prefix_campaign_first_input().length(), CAMPAIGN_TOKENS_PER_PAGE)
  assert_eq(prefix_campaign_second_input().length(), 9)
  for index, token in prefix_campaign_first_input() {
    assert_eq(prefix_campaign_second_input()[index], token)
  }
  assert_eq(prefix_campaign_second_input()[8], 114)
}

///|
test "prefix campaign refuses disabled readonly seed and cross-scope aliases" {
  let canonical = prefix_campaign_first_frame()
  let permissions : Array[@inference.CachePermission] = [Disabled, ReadOnly]
  for permission in permissions {
    let hostile = encode_prefix_campaign_request(
      PREFIX_CAMPAIGN_FIRST_REQUEST_ID,
      prefix_campaign_first_input(),
      permission,
    )
    assert_true(hostile != canonical)
    assert_eq(
      decode_prefix_campaign_frame(hostile).cache().permission(),
      permission,
    )
  }
  let reader = decode_prefix_campaign_frame(prefix_campaign_second_frame())
  let wrong_scope = @inference.CacheScope::from_ascii(
    b"spawned-prefix-other",
    prefix_campaign_inference_limits(),
  )
  assert_true(reader.cache().scope().as_string() != wrong_scope.as_string())
}

///|
test "prefix expected-token substitutions cannot rewrite either request" {
  let first = prefix_campaign_first_frame()
  let second = prefix_campaign_second_frame()
  ignore(prefix_campaign_validation_contract())
  ignore(prefix_campaign_validation_contract(first_token=1031))
  ignore(prefix_campaign_validation_contract(second_token=2185))
  assert_eq(prefix_campaign_first_frame(), first)
  assert_eq(prefix_campaign_second_frame(), second)
  assert_eq(PREFIX_REFEREE_FIRST_TOKEN, 1355)
  assert_eq(PREFIX_REFEREE_SECOND_TOKEN, 1240)
  assert_eq(
    PREFIX_REFEREE_SHA256,
    "765d4b80750e377fea131ce140a0a67931724ab06c24dc5e322c7e6b295ab8e3",
  )
}
