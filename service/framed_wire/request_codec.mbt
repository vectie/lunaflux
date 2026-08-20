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
    Text(value) => value.utf8()
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
      for index in 0..<value.length() {
        let token_id = try value.token_at(index) catch {
          _ => raise request_invalid(Payload)
        } noraise {
          value => value
        }
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
fn drive_luna_framed_request_work(
  work : LunaFramedRequestWork,
  source : FixedArray[Byte],
  length : Int,
) -> LunaFramedRequestView raise FramedWireError {
  let mut cursor = 0
  while cursor < length {
    match work.state() {
      LunaFramedRequestFailed => {
        let error = work.failure()
        work.abort()
        raise error
      }
      LunaFramedRequestReceivingBody =>
        if work.workspace.declared_length != length {
          work.abort()
          raise request_invalid(Length)
        }
      LunaFramedRequestReady => {
        work.abort()
        raise request_invalid(Length)
      }
      LunaFramedRequestReceivingPrefix | LunaFramedRequestValidating => ()
    }
    let consumed = work.offer(
      source,
      source_offset=cursor,
      length=length - cursor,
    )
    if consumed > 0 {
      cursor += consumed
    } else {
      match work.state() {
        LunaFramedRequestFailed => {
          let error = work.failure()
          work.abort()
          raise error
        }
        LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody => {
          work.abort()
          raise request_invalid(Length)
        }
        LunaFramedRequestReady => {
          work.abort()
          raise request_invalid(Length)
        }
        LunaFramedRequestValidating => ignore(work.progress())
      }
    }
  }
  while true {
    match work.state() {
      LunaFramedRequestReady => return work.take_view()
      LunaFramedRequestFailed => {
        let error = work.failure()
        work.abort()
        raise error
      }
      LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody => {
        work.abort()
        raise request_invalid(Length)
      }
      LunaFramedRequestValidating => ignore(work.progress())
    }
  }
  abort("Luna framed request compatibility driver invariant")
}

///|
fn drive_luna_framed_request(
  workspace : LunaFramedRequestWorkspace,
  source : FixedArray[Byte],
  length : Int,
) -> LunaFramedRequestView raise FramedWireError {
  if length < REQUEST_HEADER_BYTES ||
    length > workspace.limits.max_frame_bytes() ||
    length > source.length() {
    raise request_invalid(Length)
  }
  let work = workspace.begin()
  drive_luna_framed_request_work(work, source, length) catch {
    error => {
      work.abort() catch {
        _ => ()
      }
      raise error
    }
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
  self.load(self.scratch, length)
}

///|
/// Synchronous compatibility driver over the authoritative cooperative
/// scanner. Validation finishes before the previous published frame changes.
pub fn RequestFrameBuffer::load(
  self : RequestFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedRequestFrame raise FramedWireError {
  let view = drive_luna_framed_request(self.request_workspace, source, length)
  let request = materialize_luna_framed_request(view) catch {
    error => {
      view.release()
      raise error
    }
  }
  view.release()
  self.publish(request, length)
}
