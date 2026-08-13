///|
fn preflight_request_header(
  bytes : FixedArray[Byte],
  flags : UInt,
  input_count : Int,
  inference : @inference.InferenceLimits,
) -> Unit raise FramedWireError {
  ignore(
    @inference.ProtocolVersion::from_wire(read_i32(bytes, 24)) catch {
      _ => raise request_invalid(Version)
    },
  )
  ignore(
    @inference.RequestId::new(read_u64(bytes, 104)) catch {
      _ => raise request_invalid(Scalar)
    },
  )
  ignore(
    @spec.ContentDigest::from_sha256(read_digest(bytes, 112, RequestFrame)) catch {
      _ => raise request_invalid(Identity)
    },
  )
  ignore(
    @spec.PlanDigest::from_sha256(read_digest(bytes, 176, RequestFrame)) catch {
      _ => raise request_invalid(Identity)
    },
  )
  if read_i32(bytes, 52) <= 0 ||
    read_i32(bytes, 52) > inference.max_new_tokens() ||
    read_i32(bytes, 56) <= 0 ||
    read_i32(bytes, 56) > inference.max_context_tokens() ||
    read_i32(bytes, 52) > read_i32(bytes, 56) {
    raise request_invalid(Scalar)
  }
  if (flags & 1) == 0 && input_count > read_i32(bytes, 56) - read_i32(bytes, 52) {
    raise request_invalid(Scalar)
  }
  if (flags & 2) == 0 {
    if read_i32(bytes, 60) != 0 ||
      (flags & 12) != 0 ||
      read_i32(bytes, 64) != 0 ||
      read_u64(bytes, 72) != 1.0.reinterpret_as_uint64() ||
      read_u64(bytes, 80) != 0UL {
      raise request_invalid(CanonicalEncoding)
    }
  } else {
    if read_i32(bytes, 60) != 1 ||
      ((flags & 4) != 0) != (read_i32(bytes, 64) != 0) ||
      ((flags & 8) != 0) != (read_u64(bytes, 80) != 0UL) {
      raise request_invalid(CanonicalEncoding)
    }
    ignore(
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
      },
    )
  }
  ignore(
    @inference.SamplingSeed::new(read_u64(bytes, 88)) catch {
      _ => raise request_invalid(Scalar)
    },
  )
  ignore(
    @inference.DeadlineBudget::from_milliseconds(read_u64(bytes, 96), inference) catch {
      _ => raise request_invalid(Scalar)
    },
  )
  if read_i32(bytes, 240) < 0 || read_i32(bytes, 240) > 2 {
    raise request_invalid(Scalar)
  }
}

///|
fn preflight_request_payload(
  bytes : FixedArray[Byte],
  length : Int,
  flags : UInt,
  input_count : Int,
  stop_token_count : Int,
  stop_string_count : Int,
  declared_stop_bytes : Int,
  scope_length : Int,
  trace_length : Int,
  inference : @inference.InferenceLimits,
) -> Unit raise FramedWireError {
  let mut cursor = REQUEST_HEADER_BYTES
  cursor = request_append(
    cursor,
    if (flags & 1) != 0 {
      input_count
    } else {
      input_count * 4
    },
    length,
  )
  cursor = request_append(cursor, stop_token_count * 4, length)
  let mut actual_stop_bytes = 0
  for _ in 0..<stop_string_count {
    request_room(cursor, 4, length)
    let item_length = read_u32(bytes, cursor).reinterpret_as_int()
    cursor += 4
    if item_length <= 0 || item_length > inference.max_stop_string_bytes() {
      raise request_invalid(StopTable)
    }
    cursor = request_append(cursor, item_length, length)
    actual_stop_bytes += item_length
  }
  if actual_stop_bytes != declared_stop_bytes {
    raise request_invalid(StopTable)
  }
  cursor = request_append(cursor, scope_length, length)
  cursor = request_append(cursor, trace_length, length)
  if cursor != length || ((flags & 32) != 0) != (trace_length > 0) {
    raise request_invalid(CanonicalEncoding)
  }
}
