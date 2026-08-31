///|
fn bridge_request_deadline_millis() -> UInt64 {
  60000UL
}

///|
fn bridge_internal_greedy_seed() -> UInt64 {
  // The public benchmark workload uses seed zero for cross-engine parity, but
  // LunaFlux reserves zero as an invalid internal seed. Greedy sampling does
  // not consume randomness, so normalize it to the canonical non-zero seed.
  1UL
}

///|
fn bridge_limits_for_bounds(
  max_input_tokens : Int,
  max_output_tokens : Int,
  max_token_id : Int,
  max_context_tokens : Int,
) -> @inference.InferenceLimits raise BridgeRuntimeError {
  @inference.InferenceLimits::new(
    max_text_bytes=1,
    max_input_tokens~,
    max_new_tokens=max_output_tokens,
    max_context_tokens~,
    max_token_id~,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=32,
    max_decoded_delta_bytes=4096,
    max_deadline_millis=bridge_request_deadline_millis(),
    max_top_k=max_token_id + 1,
    max_temperature=2.0,
  ) catch {
    _ => raise NativeProtocol("inference limits are invalid")
  }
}

///|
fn bridge_limits(
  config : BridgeConfig,
) -> @inference.InferenceLimits raise BridgeRuntimeError {
  bridge_limits_for_bounds(
    config.max_input_tokens,
    config.max_output_tokens,
    config.max_token_id,
    config.max_context_tokens,
  )
}

///|
fn bridge_request_frame_for_identity(
  model_content_sha256 : String,
  model_plan_sha256 : String,
  request : @bridge.Qwen3TokenIdRequest,
  request_id : UInt64,
  limits : @inference.InferenceLimits,
) -> Bytes raise BridgeRuntimeError {
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(model_content_sha256),
    @spec.PlanDigest::from_sha256(model_plan_sha256),
  ) catch {
    _ => raise NativeProtocol("model identity is invalid")
  }
  let generated = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(request_id),
    model_identity=identity,
    input=@inference.Input::from_token_ids(request.input_token_ids(), limits),
    max_new_tokens=request.max_output_tokens(),
    context_ceiling=request.input_token_ids().length() +
      request.max_output_tokens(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(bridge_internal_greedy_seed()),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(
      bridge_request_deadline_millis(),
      limits,
    ),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"qwen3-benchmark", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  ) catch {
    _ => raise NativeProtocol("native request construction failed")
  }
  let framed = @framed_wire.FramedWireLimits::new(
    limits,
    max_frame_bytes=1048576,
  ) catch {
    _ => raise NativeProtocol("native frame limits are invalid")
  }
  let encoded = @framed_wire.RequestFrameBuffer::new(framed).encode(generated) catch {
    _ => raise NativeProtocol("native request encoding failed")
  }
  let length = encoded.length() catch {
    _ => raise NativeProtocol("native request length failed")
  }
  let output = FixedArray::make(length, b'\x00')
  encoded.copy_to(output, destination_offset=0) catch {
    _ => raise NativeProtocol("native request copy failed")
  }
  Bytes::makei(length, index => output[index])
}

///|
fn bridge_request_frame(
  config : BridgeConfig,
  request : @bridge.Qwen3TokenIdRequest,
  request_id : UInt64,
  limits : @inference.InferenceLimits,
) -> Bytes raise BridgeRuntimeError {
  bridge_request_frame_for_identity(
    config.model_content_sha256,
    config.model_plan_sha256,
    request,
    request_id,
    limits,
  )
}

///|
fn bridge_event_length(prefix : FixedArray[Byte]) -> Int {
  prefix[12].to_int() |
  (prefix[13].to_int() << 8) |
  (prefix[14].to_int() << 16) |
  (prefix[15].to_int() << 24)
}

///|
async fn bridge_read_event(
  connection : @socket.Tcp,
  decoder : @framed_wire.EventFrameBuffer,
  storage : FixedArray[Byte],
) -> @inference.StreamEvent raise BridgeRuntimeError {
  let mut offset = 0
  let mut length = 16
  while offset < length {
    let count = @async.with_timeout_opt(600000, () => {
      connection.read(storage, offset~, max_len=length - offset)
    }) catch {
      _ => raise NativeProtocol("native event read failed")
    }
    let received = match count {
      None => raise NativeProtocol("native event read timed out")
      Some(0) => raise NativeProtocol("native event stream closed early")
      Some(received) => received
    }
    offset += received
    if offset == 16 {
      length = bridge_event_length(storage)
      if length < 16 || length > storage.length() {
        raise NativeProtocol("native event length is invalid")
      }
    }
  }
  decoder.load(storage, length).event() catch {
    _ => raise NativeProtocol("native event decoding failed")
  }
}
