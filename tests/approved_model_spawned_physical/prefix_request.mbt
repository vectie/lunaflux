///|
const PREFIX_REFEREE_SCHEMA : String = "lunaflux.approved-bf16-prefix-referee.v1"

///|
const PREFIX_REFEREE_SHA256 : String = "765d4b80750e377fea131ce140a0a67931724ab06c24dc5e322c7e6b295ab8e3"

///|
const PREFIX_REFEREE_FIRST_TOKEN : Int = 1355

///|
const PREFIX_REFEREE_SECOND_TOKEN : Int = 1240

///|
const PREFIX_CAMPAIGN_FIRST_REQUEST_ID : UInt64 = 101UL

///|
const PREFIX_CAMPAIGN_SECOND_REQUEST_ID : UInt64 = 102UL

///|
const PREFIX_CAMPAIGN_SCOPE : String = "spawned-prefix-proof"

///|
fn prefix_campaign_first_input() -> Array[Int] {
  [1, 229, 153, 132, 75, 104, 111, 111]
}

///|
fn prefix_campaign_second_input() -> Array[Int] {
  [1, 229, 153, 132, 75, 104, 111, 111, 114]
}

///|
fn prefix_campaign_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=9,
    max_new_tokens=1,
    max_context_tokens=CAMPAIGN_CONTEXT_TOKENS,
    max_token_id=2999,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=32,
    max_trace_bytes=32,
    max_cache_scope_bytes=32,
    max_decoded_delta_bytes=512,
    max_deadline_millis=60000UL,
    max_top_k=3000,
    max_temperature=2.0,
  )
}

///|
fn encode_prefix_campaign_request(
  request_id : UInt64,
  input_tokens : Array[Int],
  permission : @inference.CachePermission,
) -> FixedArray[Byte] raise {
  let limits = prefix_campaign_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(request_id),
    model_identity=pinned_tiny_model_identity(),
    input=@inference.Input::from_token_ids(input_tokens, limits),
    max_new_tokens=1,
    context_ceiling=CAMPAIGN_CONTEXT_TOKENS,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(7UL),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(60000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"spawned-prefix-proof", limits),
      permission,
    ),
    trace=None,
    limits~,
  )
  let encoded = @framed_wire.RequestFrameBuffer::new(
    @framed_wire.FramedWireLimits::new(limits, max_frame_bytes=1048576),
  ).encode(request)
  let frame = FixedArray::make(encoded.length(), b'\x00')
  ignore(encoded.copy_to(frame, destination_offset=0))
  frame
}

///|
fn prefix_campaign_first_frame() -> FixedArray[Byte] raise {
  encode_prefix_campaign_request(
    PREFIX_CAMPAIGN_FIRST_REQUEST_ID,
    prefix_campaign_first_input(),
    ReadWrite,
  )
}

///|
fn prefix_campaign_second_frame() -> FixedArray[Byte] raise {
  encode_prefix_campaign_request(
    PREFIX_CAMPAIGN_SECOND_REQUEST_ID,
    prefix_campaign_second_input(),
    ReadOnly,
  )
}

///|
fn prefix_campaign_validation_contract(
  first_token? : Int = PREFIX_REFEREE_FIRST_TOKEN,
  second_token? : Int = PREFIX_REFEREE_SECOND_TOKEN,
) -> @runtime_instance.RuntimeSpawnedPrefixValidationContract raise {
  @runtime_instance.RuntimeSpawnedPrefixValidationContract::new(
    prefix_campaign_first_frame(),
    prefix_campaign_second_frame(),
    pinned_tiny_model_identity(),
    first_token,
    second_token,
    8,
    9,
    8,
    CAMPAIGN_CONTEXT_TOKENS,
    CAMPAIGN_TOKENS_PER_PAGE,
    1048576,
  )
}
