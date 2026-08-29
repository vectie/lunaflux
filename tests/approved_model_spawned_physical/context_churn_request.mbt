///|
const CONTEXT_CHURN_FIRST_CEILING : Int = 8

///|
const CONTEXT_CHURN_SECOND_CEILING : Int = 32

///|
const CONTEXT_CHURN_THIRD_CEILING : Int = CAMPAIGN_CONTEXT_TOKENS

///|
const CONTEXT_CHURN_REQUESTS_PER_CHILD : Int = 3

///|
const CONTEXT_CHURN_FRESH_CHILD_CYCLES : Int = 3

///|
fn context_churn_request_frame(
  request_id : UInt64,
  context_ceiling : Int,
  scope : Bytes,
) -> FixedArray[Byte] raise {
  let limits = campaign_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(request_id),
    model_identity=pinned_tiny_model_identity(),
    input=@inference.Input::from_token_ids([1], limits),
    max_new_tokens=2,
    context_ceiling~,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(7UL),
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
fn context_churn_validation_contract(
  cycle : Int,
  first_token? : Int = 1031,
  second_token? : Int = 2185,
) -> @runtime_instance.RuntimeSpawnedContextChurnContract raise {
  if cycle < 0 || cycle >= CONTEXT_CHURN_FRESH_CHILD_CYCLES {
    raise UsageRejected
  }
  let base = 900UL + (cycle * CONTEXT_CHURN_REQUESTS_PER_CHILD).to_uint64()
  @runtime_instance.RuntimeSpawnedContextChurnContract::new(
    context_churn_request_frame(
      base + 1UL,
      CONTEXT_CHURN_FIRST_CEILING,
      b"context-small",
    ),
    CONTEXT_CHURN_FIRST_CEILING,
    context_churn_request_frame(
      base + 2UL,
      CONTEXT_CHURN_SECOND_CEILING,
      b"context-medium",
    ),
    CONTEXT_CHURN_SECOND_CEILING,
    context_churn_request_frame(
      base + 3UL,
      CONTEXT_CHURN_THIRD_CEILING,
      b"context-full",
    ),
    CONTEXT_CHURN_THIRD_CEILING,
    pinned_tiny_model_identity(),
    first_token,
    second_token,
    1,
    CAMPAIGN_TOKENS_PER_PAGE,
    campaign_inference_limits(),
    1048576,
    1048576,
  )
}
