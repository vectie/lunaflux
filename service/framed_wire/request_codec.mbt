///|
let request_magic : UInt = 0x5152464c

///|
let request_version : UInt = 1

///|
fn request_invalid(rule : FramedWireRule) -> FramedWireError {
  InvalidFrame(frame=RequestFrame, rule~)
}

///|
fn request_room(
  cursor : Int,
  amount : Int,
  limit : Int,
) -> Unit raise FramedWireError {
  if amount < 0 || cursor < 0 || cursor > limit || amount > limit - cursor {
    raise request_invalid(Payload)
  }
}

///|
fn request_append(
  cursor : Int,
  amount : Int,
  maximum : Int,
) -> Int raise FramedWireError {
  if amount < 0 || cursor < 0 || cursor > maximum || amount > maximum - cursor {
    raise FrameTooLarge(frame=RequestFrame)
  }
  cursor + amount
}

///|
fn cache_permission_code(value : @inference.CachePermission) -> Int {
  match value {
    Disabled => 0
    ReadOnly => 1
    ReadWrite => 2
  }
}

///|
fn encode_request_payload(
  bytes : FixedArray[Byte],
  request : @inference.GenerateRequest,
  maximum : Int,
) -> (Int, UInt) raise FramedWireError {
  clear_header(bytes, REQUEST_HEADER_BYTES)
  let input_bytes : Bytes = match request.input() {
    Text(value) => @utf8.encode(value.text())
    TokenIds(_) => b""
  }
  let scope = @utf8.encode(request.cache().scope().as_string())
  let trace = match request.trace() {
    Some(value) => @utf8.encode(value.as_string())
    None => b""
  }
  let mut cursor = REQUEST_HEADER_BYTES
  match request.input() {
    Text(_) => {
      cursor = request_append(cursor, input_bytes.length(), maximum)
      ignore(write_bytes(bytes, REQUEST_HEADER_BYTES, input_bytes))
    }
    TokenIds(value) => {
      cursor = request_append(cursor, value.length() * 4, maximum)
      for index, token_id in value.token_ids() {
        write_i32(bytes, REQUEST_HEADER_BYTES + index * 4, token_id)
      }
    }
  }
  for token_id in request.stops().token_ids() {
    cursor = request_append(cursor, 4, maximum)
    write_i32(bytes, cursor - 4, token_id)
  }
  let mut stop_string_bytes = 0
  for value in request.stops().strings() {
    let encoded = @utf8.encode(value)
    cursor = request_append(cursor, 4, maximum)
    write_u32(bytes, cursor - 4, encoded.length().reinterpret_as_uint())
    cursor = request_append(cursor, encoded.length(), maximum)
    ignore(write_bytes(bytes, cursor - encoded.length(), encoded))
    stop_string_bytes += encoded.length()
  }
  cursor = request_append(cursor, scope.length(), maximum)
  ignore(write_bytes(bytes, cursor - scope.length(), scope))
  cursor = request_append(cursor, trace.length(), maximum)
  ignore(write_bytes(bytes, cursor - trace.length(), trace))
  let sampling = request.sampling()
  let mut flags : UInt = 0
  match request.input() {
    Text(_) => flags = flags | 1
    _ => ()
  }
  match sampling.mode() {
    Sample => flags = flags | 2
    _ => ()
  }
  match sampling.top_k() {
    Some(_) => flags = flags | 4
    _ => ()
  }
  match sampling.top_p() {
    Some(_) => flags = flags | 8
    _ => ()
  }
  match request.stream_preference() {
    Streaming => flags = flags | 16
    _ => ()
  }
  match request.trace() {
    Some(_) => flags = flags | 32
    _ => ()
  }
  write_u32(bytes, 0, request_magic)
  write_u32(bytes, 4, request_version)
  write_u32(bytes, 8, 1)
  write_u32(bytes, 12, cursor.reinterpret_as_uint())
  write_u32(bytes, 20, flags)
  write_i32(bytes, 24, request.protocol_version().wire_code())
  write_i32(
    bytes,
    28,
    match request.input() {
      Text(value) => value.utf8_bytes()
      TokenIds(value) => value.length()
    },
  )
  write_i32(bytes, 32, request.stops().token_ids().length())
  write_i32(bytes, 36, request.stops().strings().length())
  write_i32(bytes, 40, stop_string_bytes)
  write_i32(bytes, 44, scope.length())
  write_i32(bytes, 48, trace.length())
  write_i32(bytes, 52, request.max_new_tokens())
  write_i32(bytes, 56, request.context_ceiling())
  write_i32(
    bytes,
    60,
    match sampling.mode() {
      Greedy => 0
      Sample => 1
    },
  )
  write_i32(bytes, 64, sampling.top_k().unwrap_or(0))
  write_u64(bytes, 72, sampling.temperature().reinterpret_as_uint64())
  write_u64(bytes, 80, sampling.top_p().unwrap_or(0.0).reinterpret_as_uint64())
  write_u64(bytes, 88, request.sampling_seed().value())
  write_u64(bytes, 96, request.deadline().milliseconds())
  write_u64(bytes, 104, request.request_id().value())
  write_digest(
    bytes,
    112,
    request.model_identity().content().as_hex(),
    RequestFrame,
  )
  write_digest(
    bytes,
    176,
    request.model_identity().plan().as_hex(),
    RequestFrame,
  )
  write_i32(bytes, 240, cache_permission_code(request.cache().permission()))
  write_i32(bytes, 244, cursor - REQUEST_HEADER_BYTES)
  write_i32(bytes, 248, REQUEST_HEADER_BYTES)
  write_u32(bytes, 16, frame_checksum(bytes, cursor))
  (cursor, flags)
}

///|
fn parse_request(
  bytes : FixedArray[Byte],
  length : Int,
  limits : FramedWireLimits,
) -> @inference.GenerateRequest raise FramedWireError {
  if length < REQUEST_HEADER_BYTES ||
    length > limits.max_frame_bytes() ||
    length > bytes.length() {
    raise request_invalid(Length)
  }
  if read_u32(bytes, 0) != request_magic {
    raise request_invalid(Magic)
  }
  if read_u32(bytes, 4) != request_version {
    raise request_invalid(Version)
  }
  if read_u32(bytes, 8) != 1 {
    raise request_invalid(Kind)
  }
  if read_u32(bytes, 12).reinterpret_as_int() != length {
    raise request_invalid(Length)
  }
  if read_u32(bytes, 16) != frame_checksum(bytes, length) {
    raise request_invalid(Checksum)
  }
  let flags = read_u32(bytes, 20)
  if (flags & 0xffffffc0) != 0 {
    raise request_invalid(Flags)
  }
  if read_u32(bytes, 68) != 0 ||
    read_u32(bytes, 252) != 0 ||
    read_i32(bytes, 248) != REQUEST_HEADER_BYTES ||
    read_i32(bytes, 244) != length - REQUEST_HEADER_BYTES {
    raise request_invalid(Reserved)
  }
  let input_count = read_i32(bytes, 28)
  let stop_token_count = read_i32(bytes, 32)
  let stop_string_count = read_i32(bytes, 36)
  let declared_stop_bytes = read_i32(bytes, 40)
  let scope_length = read_i32(bytes, 44)
  let trace_length = read_i32(bytes, 48)
  if input_count < 0 ||
    stop_token_count < 0 ||
    stop_string_count < 0 ||
    declared_stop_bytes < 0 ||
    scope_length < 0 ||
    trace_length < 0 {
    raise request_invalid(Length)
  }
  let inference = limits.inference_limits()
  let input_limit = if (flags & 1) != 0 {
    inference.max_text_bytes()
  } else {
    inference.max_input_tokens()
  }
  if input_count == 0 ||
    input_count > input_limit ||
    stop_token_count > inference.max_stop_token_ids() ||
    stop_string_count > inference.max_stop_strings() ||
    declared_stop_bytes > stop_string_count * inference.max_stop_string_bytes() ||
    scope_length == 0 ||
    scope_length > inference.max_cache_scope_bytes() ||
    trace_length > inference.max_trace_bytes() {
    raise request_invalid(Length)
  }
  preflight_request_header(bytes, flags, input_count, inference)
  preflight_request_payload(
    bytes, length, flags, input_count, stop_token_count, stop_string_count, declared_stop_bytes,
    scope_length, trace_length, inference,
  )
  let mut cursor = REQUEST_HEADER_BYTES
  let input = if (flags & 1) != 0 {
    request_room(cursor, input_count, length)
    let payload = read_bytes(bytes, cursor, input_count)
    cursor += input_count
    @inference.Input::from_utf8(payload, inference) catch {
      _ => raise request_invalid(Input)
    }
  } else {
    request_room(cursor, input_count * 4, length)
    let tokens : Array[Int] = []
    for index in 0..<input_count {
      tokens.push(read_i32(bytes, cursor + index * 4))
    }
    cursor += input_count * 4
    @inference.Input::from_token_ids(tokens, inference) catch {
      _ => raise request_invalid(Input)
    }
  }
  request_room(cursor, stop_token_count * 4, length)
  let stop_tokens : Array[Int] = []
  for index in 0..<stop_token_count {
    stop_tokens.push(read_i32(bytes, cursor + index * 4))
  }
  cursor += stop_token_count * 4
  let stop_strings : Array[Bytes] = []
  let mut actual_stop_bytes = 0
  for _ in 0..<stop_string_count {
    request_room(cursor, 4, length)
    let item_length = read_u32(bytes, cursor).reinterpret_as_int()
    cursor += 4
    if item_length <= 0 || item_length > inference.max_stop_string_bytes() {
      raise request_invalid(StopTable)
    }
    request_room(cursor, item_length, length)
    stop_strings.push(read_bytes(bytes, cursor, item_length))
    cursor += item_length
    actual_stop_bytes += item_length
  }
  if actual_stop_bytes != declared_stop_bytes {
    raise request_invalid(StopTable)
  }
  request_room(cursor, scope_length, length)
  let scope_bytes = read_bytes(bytes, cursor, scope_length)
  cursor += scope_length
  request_room(cursor, trace_length, length)
  let trace_bytes = read_bytes(bytes, cursor, trace_length)
  cursor += trace_length
  if cursor != length || ((flags & 32) != 0) != (trace_length > 0) {
    raise request_invalid(CanonicalEncoding)
  }
  let protocol = @inference.ProtocolVersion::from_wire(read_i32(bytes, 24)) catch {
    _ => raise request_invalid(Version)
  }
  let request_id = @inference.RequestId::new(read_u64(bytes, 104)) catch {
    _ => raise request_invalid(Scalar)
  }
  let content = @spec.ContentDigest::from_sha256(
    read_digest(bytes, 112, RequestFrame),
  ) catch {
    _ => raise request_invalid(Identity)
  }
  let plan = @spec.PlanDigest::from_sha256(
    read_digest(bytes, 176, RequestFrame),
  ) catch {
    _ => raise request_invalid(Identity)
  }
  let sampling = if (flags & 2) == 0 {
    if read_i32(bytes, 60) != 0 ||
      (flags & 12) != 0 ||
      read_i32(bytes, 64) != 0 ||
      read_u64(bytes, 72) != 1.0.reinterpret_as_uint64() ||
      read_u64(bytes, 80) != 0UL {
      raise request_invalid(CanonicalEncoding)
    }
    @inference.SamplingParameters::greedy()
  } else {
    if read_i32(bytes, 60) != 1 ||
      ((flags & 4) != 0) != (read_i32(bytes, 64) != 0) ||
      ((flags & 8) != 0) != (read_u64(bytes, 80) != 0UL) {
      raise request_invalid(CanonicalEncoding)
    }
    @inference.SamplingParameters::sample(
      temperature=read_u64(bytes, 72).reinterpret_as_double(),
      top_k?=if (flags & 4) != 0 { Some(read_i32(bytes, 64)) } else { None },
      top_p?=if (flags & 8) != 0 {
        Some(read_u64(bytes, 80).reinterpret_as_double())
      } else {
        None
      },
      inference,
    ) catch {
      _ => raise request_invalid(Scalar)
    }
  }
  let stops = @inference.StopConditions::new(
    stop_tokens, stop_strings, inference,
  ) catch {
    _ => raise request_invalid(StopTable)
  }
  let scope = @inference.CacheScope::from_ascii(scope_bytes, inference) catch {
    _ => raise request_invalid(Payload)
  }
  let permission = match read_i32(bytes, 240) {
    0 => @inference.Disabled
    1 => ReadOnly
    2 => ReadWrite
    _ => raise request_invalid(Scalar)
  }
  let trace = if (flags & 32) != 0 {
    Some(
      @inference.TraceCorrelation::from_ascii(trace_bytes, inference) catch {
        _ => raise request_invalid(Payload)
      },
    )
  } else {
    None
  }
  @inference.GenerateRequest::new(
    protocol_version=protocol,
    request_id~,
    model_identity=@spec.ModelIdentity::new(content, plan),
    input~,
    max_new_tokens=read_i32(bytes, 52),
    context_ceiling=read_i32(bytes, 56),
    sampling~,
    sampling_seed=@inference.SamplingSeed::new(read_u64(bytes, 88)) catch {
      _ => raise request_invalid(Scalar)
    },
    stops~,
    stream_preference=if (flags & 16) != 0 { Streaming } else { Buffered },
    deadline=@inference.DeadlineBudget::from_milliseconds(
      read_u64(bytes, 96),
      inference,
    ) catch {
      _ => raise request_invalid(Scalar)
    },
    cache=@inference.CachePolicy::new(scope, permission),
    trace~,
    limits=inference,
  ) catch {
    _ => raise request_invalid(CanonicalEncoding)
  }
}

///|
pub fn RequestFrameBuffer::encode(
  self : RequestFrameBuffer,
  request : @inference.GenerateRequest,
) -> ValidatedRequestFrame raise FramedWireError {
  let (length, _) = encode_request_payload(
    self.scratch,
    request,
    self.limits.max_frame_bytes(),
  )
  let canonical = parse_request(self.scratch, length, self.limits)
  self.publish(canonical, length)
}

///|
/// Validates before replacing the current frame; rejection is transactional.
pub fn RequestFrameBuffer::load(
  self : RequestFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedRequestFrame raise FramedWireError {
  let request = parse_request(source, length, self.limits)
  copy_bytes(source, self.scratch, length)
  self.publish(request, length)
}
