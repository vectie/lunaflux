///|
const LONG_CONTEXT_FRESH_CHILD_CYCLES : Int = 3

///|
const LONG_CONTEXT_FIRST_INPUT_TOKENS : Int = 8

///|
const LONG_CONTEXT_SECOND_INPUT_TOKENS : Int = 9

///|
fn long_context_frame(
  request_id : UInt64,
  input_tokens : Array[Int],
) -> FixedArray[Byte] raise {
  encode_prefix_campaign_request(request_id, input_tokens, Disabled)
}

///|
fn long_context_validation_contract(
  cycle : Int,
  first_token? : Int = PREFIX_REFEREE_FIRST_TOKEN,
  second_token? : Int = PREFIX_REFEREE_SECOND_TOKEN,
) -> @runtime_instance.RuntimeSpawnedLongContextContract raise {
  if cycle < 0 || cycle >= LONG_CONTEXT_FRESH_CHILD_CYCLES {
    raise UsageRejected
  }
  let first_input = prefix_campaign_first_input()
  let second_input = prefix_campaign_second_input()
  if first_input.length() != LONG_CONTEXT_FIRST_INPUT_TOKENS ||
    second_input.length() != LONG_CONTEXT_SECOND_INPUT_TOKENS {
    raise UsageRejected
  }
  let base = 1200UL + (cycle * 2).to_uint64()
  @runtime_instance.RuntimeSpawnedLongContextContract::new(
    long_context_frame(base + 1UL, first_input),
    long_context_frame(base + 2UL, second_input),
    pinned_tiny_model_identity(),
    first_token,
    second_token,
    CAMPAIGN_CONTEXT_TOKENS,
    CAMPAIGN_TOKENS_PER_PAGE,
    prefix_campaign_inference_limits(),
    1048576,
    1048576,
  )
}
