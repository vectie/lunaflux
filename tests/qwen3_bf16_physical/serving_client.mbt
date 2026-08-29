///|
fn qwen3_serving_int(
  value : String,
  label : String,
) -> Int raise Qwen3PhysicalError {
  let parsed = @string.parse_int(value, base=10) catch {
    _ => raise InvalidInput(label + " is not numeric")
  }
  qwen3_require(parsed > 0, label + " is not positive")
  parsed
}

///|
fn qwen3_serving_limits(
  corpus : Qwen3GreedyCorpusCase,
  max_token_id : Int,
  max_context_tokens : Int,
) -> @inference.InferenceLimits raise Qwen3PhysicalError {
  qwen3_require(
    corpus.input_ids.length() + corpus.expected_ids.length() <=
    max_context_tokens,
    "greedy corpus exceeds context ceiling",
  )
  @inference.InferenceLimits::new(
    max_text_bytes=1,
    max_input_tokens=corpus.input_ids.length(),
    max_new_tokens=corpus.expected_ids.length(),
    max_context_tokens~,
    max_token_id~,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=32,
    max_decoded_delta_bytes=4096,
    max_deadline_millis=60000UL,
    max_top_k=max_token_id + 1,
    max_temperature=2.0,
  ) catch {
    _ => raise InvalidInput("serving inference limits are invalid")
  }
}

///|
fn qwen3_serving_frame(
  corpus : Qwen3GreedyCorpusCase,
  content_sha256 : String,
  plan_sha256 : String,
  limits : @inference.InferenceLimits,
) -> Bytes raise Qwen3PhysicalError {
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(content_sha256),
    @spec.PlanDigest::from_sha256(plan_sha256),
  ) catch {
    _ => raise InvalidInput("serving model identity is invalid")
  }
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(1UL),
    model_identity=identity,
    input=@inference.Input::from_token_ids(corpus.input_ids, limits),
    max_new_tokens=corpus.expected_ids.length(),
    context_ceiling=corpus.input_ids.length() + corpus.expected_ids.length(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(1UL),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(60000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"qwen3-physical", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  ) catch {
    _ => raise InvalidInput("serving request construction failed")
  }
  let framed = @framed_wire.FramedWireLimits::new(
    limits,
    max_frame_bytes=1048576,
  ) catch {
    _ => raise InvalidInput("serving frame limits are invalid")
  }
  let encoded = @framed_wire.RequestFrameBuffer::new(framed).encode(request) catch {
    _ => raise InvalidInput("serving request encoding failed")
  }
  let encoded_length = encoded.length() catch {
    _ => raise InvalidInput("serving request length failed")
  }
  let output = FixedArray::make(encoded_length, b'\x00')
  encoded.copy_to(output, destination_offset=0) catch {
    _ => raise InvalidInput("serving request copy failed")
  }
  Bytes::makei(output.length(), index => output[index])
}

///|
async fn qwen3_write_all(
  connection : @socket.Tcp,
  bytes : Bytes,
) -> Unit raise Qwen3PhysicalError {
  let mut offset = 0
  while offset < bytes.length() {
    let count = @async.with_timeout_opt(60000, () => {
      connection.write_once(bytes, offset~, len=bytes.length() - offset)
    }) catch {
      _ => raise InvalidInput("serving request write failed")
    }
    guard count is Some(written) && written > 0 else {
      raise InvalidInput("serving request write timed out")
    }
    offset += written
  }
}

///|
fn qwen3_event_length(prefix : FixedArray[Byte]) -> Int {
  prefix[12].to_int() |
  (prefix[13].to_int() << 8) |
  (prefix[14].to_int() << 16) |
  (prefix[15].to_int() << 24)
}

///|
async fn qwen3_read_event(
  connection : @socket.Tcp,
  decoder : @framed_wire.EventFrameBuffer,
) -> @inference.StreamEvent raise Qwen3PhysicalError {
  let storage = FixedArray::make(decoder.capacity(), b'\x00')
  let mut offset = 0
  let mut length = 16
  while offset < length {
    let count = @async.with_timeout_opt(60000, () => {
      connection.read(storage, offset~, max_len=length - offset)
    }) catch {
      _ => raise InvalidInput("serving event read failed")
    }
    guard count is Some(received) && received > 0 else {
      raise InvalidInput("serving event read timed out")
    }
    offset += received
    if offset == 16 {
      length = qwen3_event_length(storage)
      qwen3_require(
        length >= 16 && length <= storage.length(),
        "serving event length is invalid",
      )
    }
  }
  decoder.load(storage, length).event() catch {
    _ => raise InvalidInput("serving event decoding failed")
  }
}

///|
fn qwen3_token_sequence(values : ArrayView[Int]) -> String {
  values.map(value => value.to_string()).join(",")
}

///|
async fn qwen3_serve_check(
  args : ArrayView[String],
) -> Unit raise Qwen3PhysicalError {
  qwen3_require(args.length() == 9, "serve-check arguments are invalid")
  let corpus_bytes = qwen3_read_pinned(args[3], args[4], 1048576)
  let corpus = qwen3_greedy_corpus_case(corpus_bytes)
  let max_token_id = qwen3_serving_int(args[7], "maximum token ID")
  let max_context_tokens = qwen3_serving_int(args[8], "maximum context")
  let limits = qwen3_serving_limits(corpus, max_token_id, max_context_tokens)
  let frame = qwen3_serving_frame(corpus, args[5], args[6], limits)
  let connection = @socket.Tcp::connect(@socket.Addr::parse(args[2])) catch {
    _ => raise InvalidInput("serving endpoint connection failed")
  }
  defer connection.close()
  qwen3_write_all(connection, frame)
  let framed = @framed_wire.FramedWireLimits::new(
    limits,
    max_frame_bytes=1048576,
  ) catch {
    _ => raise InvalidInput("serving event limits are invalid")
  }
  let decoder = @framed_wire.EventFrameBuffer::new(framed)
  guard qwen3_read_event(connection, decoder) is Accepted(_) else {
    raise CorpusMismatch("serving request was not accepted")
  }
  let observed : Array[Int] = []
  for position, expected in corpus.expected_ids {
    match qwen3_read_event(connection, decoder) {
      Token(token) => {
        qwen3_require(token.position() == position, "token position differs")
        qwen3_require(token.token_id() == expected, "greedy token differs")
        observed.push(token.token_id())
      }
      _ => raise CorpusMismatch("token event is absent")
    }
  }
  match qwen3_read_event(connection, decoder) {
    Usage(usage) =>
      qwen3_require(
        usage.input_tokens() == corpus.input_ids.length() &&
        usage.output_tokens() == corpus.expected_ids.length(),
        "serving usage differs",
      )
    _ => raise CorpusMismatch("usage event is absent")
  }
  guard qwen3_read_event(connection, decoder) is Completed(_) else {
    raise CorpusMismatch("completed event is absent")
  }
  println(
    "outcome=qwen3-serving-corpus-pass first_token=" +
    observed[0].to_string() +
    " greedy_tokens=" +
    qwen3_token_sequence(observed) +
    " input_tokens=" +
    corpus.input_ids.length().to_string() +
    " output_tokens=" +
    observed.length().to_string(),
  )
}
