///|
fn hostile_limits() -> InferenceLimits raise {
  InferenceLimits::new(
    max_text_bytes=16,
    max_input_tokens=8,
    max_new_tokens=6,
    max_context_tokens=12,
    max_token_id=99,
    max_stop_token_ids=3,
    max_stop_strings=2,
    max_stop_string_bytes=5,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=100,
    max_temperature=4.0,
  )
}

///|
fn hostile_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn hostile_owned_input(values : Array[Int], cached_maximum : Int) -> Input {
  TokenIds({
    backing: LunaOwnedTokenBacking(ReadOnlyArray::from_array(values)),
    token_count: values.length(),
    maximum_token_id: cached_maximum,
  })
}

///|
fn hostile_stops(
  tokens : Array[Int],
  strings : Array[String],
) -> StopConditions {
  {
    token_ids: ReadOnlyArray::from_array(tokens),
    strings: ReadOnlyArray::from_array(strings),
  }
}

///|
fn hostile_request(
  limits~ : InferenceLimits,
  protocol_version~ : ProtocolVersion,
  max_new_tokens~ : Int,
  context_ceiling~ : Int,
  input~ : Input,
  sampling~ : SamplingParameters,
  sampling_seed~ : SamplingSeed,
  stops~ : StopConditions,
  cache~ : CachePolicy,
  trace~ : TraceCorrelation?,
) -> Unit raise {
  ignore(
    GenerateRequest::new(
      protocol_version~,
      request_id=RequestId::new(1UL),
      model_identity=hostile_identity(),
      input~,
      max_new_tokens~,
      context_ceiling~,
      sampling~,
      sampling_seed~,
      stops~,
      stream_preference=Buffered,
      deadline={ milliseconds: 10UL },
      cache~,
      trace~,
      limits~,
    ),
  )
}

///|
fn hostile_valid_cache() -> CachePolicy {
  CachePolicy::new({ value: "scope-1" }, ReadWrite)
}

///|
fn hostile_valid_stops() -> StopConditions {
  hostile_stops([], [])
}

///|
fn hostile_valid_input() -> Input {
  hostile_owned_input([1, 2], 2)
}

///|
fn invalid_utf16_strings() -> Array[String] {
  [
    String::from_array([(0xd800).unsafe_to_char(), 'a']),
    String::from_array(['a', (0xdc00).unsafe_to_char()]),
    String::from_array([(0xd800).unsafe_to_char()]),
  ]
}

///|
test "GenerateRequest validates forged limits before protocol authority" {
  let forged = { ..hostile_limits(), max_text_bytes: 0 }
  try
    hostile_request(
      limits=forged,
      protocol_version={ wire_code: 2 },
      max_new_tokens=0,
      context_ceiling=0,
      input=hostile_owned_input([-1], -1),
      sampling={
        mode: Sample,
        temperature: @double.not_a_number,
        top_k: Some(0),
        top_p: Some(@double.not_a_number),
      },
      sampling_seed={ value: 0UL },
      stops=hostile_stops(Array::make(4, -1), [""]),
      cache=CachePolicy::new({ value: "" }, ReadWrite),
      trace=Some({ value: "" }),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=Limit, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected forged limit precedence")
  }
}

///|
test "GenerateRequest adjacent validation precedence is fixed" {
  let cfg = hostile_limits()
  let invalid_input = hostile_owned_input([-1], -1)
  let invalid_sample = SamplingParameters::{
    mode: Sample,
    temperature: 0.0,
    top_k: Some(0),
    top_p: Some(@double.not_a_number),
  }
  let invalid_seed : SamplingSeed = { value: 0UL }
  let cases = [
    (
      0, 0, invalid_input, invalid_sample, invalid_seed, "Invalid(field=MaximumNewTokens, issue=OutOfRange, index=0)",
    ),
    (
      2, 0, invalid_input, invalid_sample, invalid_seed, "Invalid(field=ContextCeiling, issue=OutOfRange, index=0)",
    ),
    (
      6, 5, invalid_input, invalid_sample, invalid_seed, "Invalid(field=ContextCeiling, issue=Inconsistent, index=0)",
    ),
    (
      2, 8, invalid_input, invalid_sample, invalid_seed, "Invalid(field=InputToken, issue=OutOfRange, index=0)",
    ),
    (
      2,
      8,
      hostile_valid_input(),
      invalid_sample,
      invalid_seed,
      "Invalid(field=Temperature, issue=OutOfRange, index=0)",
    ),
    (
      2,
      8,
      hostile_valid_input(),
      SamplingParameters::greedy(),
      invalid_seed,
      "Invalid(field=SamplingSeed, issue=OutOfRange, index=0)",
    ),
  ]
  for case in cases {
    let (max_new, context, input, sampling, seed, expected) = case
    try
      hostile_request(
        limits=cfg,
        protocol_version=ProtocolVersion::v1(),
        max_new_tokens=max_new,
        context_ceiling=context,
        input~,
        sampling~,
        sampling_seed=seed,
        stops=hostile_valid_stops(),
        cache=hostile_valid_cache(),
        trace=None,
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected adjacent request validation rejection")
    }
  }
}

///|
test "forged Sample values retain numeric and filter precedence" {
  let cases = [
    (
      {
        mode: Sample,
        temperature: @double.not_a_number,
        top_k: Some(0),
        top_p: Some(@double.not_a_number),
      },
      "Invalid(field=Temperature, issue=NonFinite, index=0)",
    ),
    (
      { mode: Sample, temperature: @double.infinity, top_k: None, top_p: None },
      "Invalid(field=Temperature, issue=NonFinite, index=0)",
    ),
    (
      { mode: Sample, temperature: 0.0, top_k: None, top_p: None },
      "Invalid(field=Temperature, issue=OutOfRange, index=0)",
    ),
    (
      {
        mode: Sample,
        temperature: 1.0,
        top_k: Some(0),
        top_p: Some(@double.not_a_number),
      },
      "Invalid(field=TopK, issue=OutOfRange, index=0)",
    ),
    (
      {
        mode: Sample,
        temperature: 1.0,
        top_k: None,
        top_p: Some(@double.not_a_number),
      },
      "Invalid(field=TopP, issue=NonFinite, index=0)",
    ),
    (
      { mode: Sample, temperature: 1.0, top_k: None, top_p: Some(1.1) },
      "Invalid(field=TopP, issue=OutOfRange, index=0)",
    ),
  ]
  for case in cases {
    let (sampling, expected) = case
    try
      hostile_request(
        limits=hostile_limits(),
        protocol_version=ProtocolVersion::v1(),
        max_new_tokens=2,
        context_ceiling=8,
        input=hostile_valid_input(),
        sampling~,
        sampling_seed=SamplingSeed::new(1UL),
        stops=hostile_valid_stops(),
        cache=hostile_valid_cache(),
        trace=None,
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected forged Sample rejection")
    }
  }
}

///|
test "defensive collection counts precede stop and input elements" {
  let cases = [
    (
      hostile_owned_input(Array::make(9, 100), 100),
      hostile_valid_stops(),
      "Invalid(field=InputToken, issue=TooMany, index=8)",
    ),
    (
      hostile_owned_input([100], 100),
      hostile_valid_stops(),
      "Invalid(field=InputToken, issue=OutOfRange, index=0)",
    ),
    (
      hostile_valid_input(),
      hostile_stops(Array::make(4, -1), Array::make(3, "")),
      "Invalid(field=StopToken, issue=TooMany, index=3)",
    ),
    (
      hostile_valid_input(),
      hostile_stops([], Array::make(3, "")),
      "Invalid(field=StopString, issue=TooMany, index=2)",
    ),
  ]
  for case in cases {
    let (input, stops, expected) = case
    try
      hostile_request(
        limits=hostile_limits(),
        protocol_version=ProtocolVersion::v1(),
        max_new_tokens=2,
        context_ceiling=12,
        input~,
        sampling=SamplingParameters::greedy(),
        sampling_seed=SamplingSeed::new(1UL),
        stops~,
        cache=hostile_valid_cache(),
        trace=None,
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected defensive collection rejection")
    }
  }
}

///|
test "malformed UTF-16 stop and identifier strings fail typed" {
  for malformed in invalid_utf16_strings() {
    try
      hostile_request(
        limits=hostile_limits(),
        protocol_version=ProtocolVersion::v1(),
        max_new_tokens=2,
        context_ceiling=8,
        input=hostile_valid_input(),
        sampling=SamplingParameters::greedy(),
        sampling_seed=SamplingSeed::new(1UL),
        stops=hostile_stops([], [malformed]),
        cache=CachePolicy::new({ value: malformed }, ReadWrite),
        trace=Some({ value: malformed }),
      )
    catch {
      error =>
        debug_inspect(
          error,
          content="Invalid(field=StopString, issue=InvalidFormat, index=0)",
        )
    } noraise {
      _ => fail("expected malformed stop rejection")
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
        cache=CachePolicy::new({ value: malformed }, ReadWrite),
        trace=Some({ value: malformed }),
      )
    catch {
      error =>
        debug_inspect(
          error,
          content="Invalid(field=CacheScope, issue=InvalidFormat, index=0)",
        )
    } noraise {
      _ => fail("expected malformed cache scope rejection")
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
        trace=Some({ value: malformed }),
      )
    catch {
      error =>
        debug_inspect(
          error,
          content="Invalid(field=TraceCorrelation, issue=InvalidFormat, index=0)",
        )
    } noraise {
      _ => fail("expected malformed trace rejection")
    }
  }
}
