///|
const CAMPAIGN_TOKENS_PER_PAGE : Int = 8

///|
const CAMPAIGN_CONTEXT_PAGES : Int = 32

///|
const CAMPAIGN_CONTEXT_TOKENS : Int = CAMPAIGN_TOKENS_PER_PAGE *
  CAMPAIGN_CONTEXT_PAGES

///|
fn campaign_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=8,
    max_new_tokens=8,
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
fn campaign_request_frame() -> FixedArray[Byte] raise {
  let limits = campaign_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(1UL),
    model_identity=pinned_tiny_model_identity(),
    input=@inference.Input::from_token_ids([1], limits),
    max_new_tokens=2,
    context_ceiling=CAMPAIGN_CONTEXT_TOKENS,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(7UL),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(60000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"spawned-proof", limits),
      Disabled,
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
fn broad_campaign_request_frame(
  request_id : UInt64,
  seed : UInt64,
  model_identity : @spec.ModelIdentity,
  maximum_new_tokens : Int,
  scope : Bytes,
) -> FixedArray[Byte] raise {
  let limits = campaign_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(request_id),
    model_identity~,
    input=@inference.Input::from_token_ids([1], limits),
    max_new_tokens=maximum_new_tokens,
    context_ceiling=CAMPAIGN_CONTEXT_TOKENS,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(seed),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(60000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(scope, limits),
      Disabled,
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
fn broad_campaign_validation_contract(
  first_token? : Int = 1031,
  second_token? : Int = 2185,
) -> @runtime_instance.RuntimeSpawnedBroadServingValidationContract raise {
  @runtime_instance.RuntimeSpawnedBroadServingValidationContract::new(
    broad_campaign_request_frame(
      701UL,
      17UL,
      pinned_tiny_model_identity(),
      1,
      b"broad-a",
    ),
    broad_campaign_request_frame(
      702UL,
      29UL,
      pinned_tiny_model_identity(),
      1,
      b"broad-b",
    ),
    campaign_request_frame(),
    broad_campaign_request_frame(
      704UL,
      37UL,
      pinned_tiny_model_identity(),
      2,
      b"recovery",
    ),
    broad_campaign_request_frame(
      703UL,
      31UL,
      foreign_tiny_model_identity(),
      1,
      b"foreign",
    ),
    pinned_tiny_model_identity(),
    first_token,
    second_token,
    1,
    CAMPAIGN_CONTEXT_TOKENS,
    CAMPAIGN_TOKENS_PER_PAGE,
    campaign_inference_limits(),
    1048576,
    1048576,
    5000,
  )
}

///|
fn campaign_validation_contract(
  first_token? : Int = 1031,
  second_token? : Int = 2185,
) -> @runtime_instance.RuntimeSpawnedValidationContract raise {
  @runtime_instance.RuntimeSpawnedValidationContract::new(
    campaign_request_frame(),
    pinned_tiny_model_identity(),
    first_token,
    second_token,
    1,
    CAMPAIGN_CONTEXT_TOKENS,
    CAMPAIGN_TOKENS_PER_PAGE,
    1048576,
  )
}

///|
fn campaign_serving_validation_contract(
  first_token? : Int = 1031,
  second_token? : Int = 2185,
) -> @runtime_instance.RuntimeSpawnedServingValidationContract raise {
  @runtime_instance.RuntimeSpawnedServingValidationContract::new(
    campaign_request_frame(),
    pinned_tiny_model_identity(),
    first_token,
    second_token,
    1,
    CAMPAIGN_CONTEXT_TOKENS,
    CAMPAIGN_TOKENS_PER_PAGE,
    campaign_inference_limits(),
    1048576,
    1048576,
    5000,
  )
}
