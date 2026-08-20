///|
fn oversized_invalid_utf16(prefix_length : Int) -> String {
  let chars = Array::make(prefix_length, 'a')
  chars.push((0xd800).unsafe_to_char())
  String::from_array(chars)
}

///|
test "valid supplementary UTF-16 pairs use exact four-byte accounting" {
  let emoji = String::from_array([(0x1f600).unsafe_to_char()])
  let fits = { ..hostile_limits(), max_stop_string_bytes: 4 }
  hostile_request(
    limits=fits,
    protocol_version=ProtocolVersion::v1(),
    max_new_tokens=2,
    context_ceiling=8,
    input=hostile_valid_input(),
    sampling=SamplingParameters::greedy(),
    sampling_seed=SamplingSeed::new(1UL),
    stops=hostile_stops([], [emoji]),
    cache=hostile_valid_cache(),
    trace=None,
  )
  let too_small = { ..hostile_limits(), max_stop_string_bytes: 3 }
  try
    hostile_request(
      limits=too_small,
      protocol_version=ProtocolVersion::v1(),
      max_new_tokens=2,
      context_ceiling=8,
      input=hostile_valid_input(),
      sampling=SamplingParameters::greedy(),
      sampling_seed=SamplingSeed::new(1UL),
      stops=hostile_stops([], [emoji]),
      cache=hostile_valid_cache(),
      trace=None,
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=TooLong, index=0)",
      )
  } noraise {
    _ => fail("expected exact supplementary UTF-8 byte bound")
  }
}

///|
test "definitely oversized UTF-16 fails before malformed suffix scanning" {
  let stop = oversized_invalid_utf16(5)
  try
    hostile_request(
      limits=hostile_limits(),
      protocol_version=ProtocolVersion::v1(),
      max_new_tokens=2,
      context_ceiling=8,
      input=hostile_valid_input(),
      sampling=SamplingParameters::greedy(),
      sampling_seed=SamplingSeed::new(1UL),
      stops=hostile_stops([], [stop]),
      cache=hostile_valid_cache(),
      trace=None,
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=TooLong, index=0)",
      )
  } noraise {
    _ => fail("expected stop code-unit preflight bound")
  }
  let identifier = oversized_invalid_utf16(8)
  try
    hostile_request(
      limits=hostile_limits(),
      protocol_version=ProtocolVersion::v1(),
      max_new_tokens=2,
      context_ceiling=8,
      input=hostile_valid_input(),
      sampling=SamplingParameters::greedy(),
      sampling_seed=SamplingSeed::new(1UL),
      stops=hostile_valid_stops(),
      cache=CachePolicy::new({ value: identifier }, ReadWrite),
      trace=Some({ value: identifier }),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=CacheScope, issue=TooLong, index=0)",
      )
  } noraise {
    _ => fail("expected cache code-unit preflight bound")
  }
  try
    hostile_request(
      limits=hostile_limits(),
      protocol_version=ProtocolVersion::v1(),
      max_new_tokens=2,
      context_ceiling=8,
      input=hostile_valid_input(),
      sampling=SamplingParameters::greedy(),
      sampling_seed=SamplingSeed::new(1UL),
      stops=hostile_valid_stops(),
      cache=hostile_valid_cache(),
      trace=Some({ value: identifier }),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=TraceCorrelation, issue=TooLong, index=0)",
      )
  } noraise {
    _ => fail("expected trace code-unit preflight bound")
  }
}
