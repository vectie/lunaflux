///|
fn authority_limits() -> InferenceLimits raise {
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
fn authority_model_identity() -> @spec.ModelIdentity raise {
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
fn forged_stops(
  token_ids : Array[Int],
  strings : Array[String],
) -> StopConditions {
  {
    token_ids: ReadOnlyArray::from_array(token_ids),
    strings: ReadOnlyArray::from_array(strings),
  }
}

///|
fn validate_authority_request(
  protocol_version~ : ProtocolVersion,
  request_id~ : RequestId,
  max_new_tokens~ : Int,
  sampling~ : SamplingParameters,
  stops~ : StopConditions,
  deadline~ : DeadlineBudget,
  cache~ : CachePolicy,
  trace~ : TraceCorrelation?,
) -> Unit raise {
  let cfg = authority_limits()
  ignore(
    GenerateRequest::new(
      protocol_version~,
      request_id~,
      model_identity=authority_model_identity(),
      input=Input::from_token_ids([1, 2], cfg),
      max_new_tokens~,
      context_ceiling=8,
      sampling~,
      sampling_seed=SamplingSeed::new(7UL),
      stops~,
      stream_preference=Buffered,
      deadline~,
      cache~,
      trace~,
      limits=cfg,
    ),
  )
}

///|
fn valid_authority_stops() -> StopConditions {
  forged_stops([], [])
}

///|
fn valid_authority_cache() -> CachePolicy {
  CachePolicy::new({ value: "scope-1" }, ReadWrite)
}

///|
fn forged_owned_input(
  token_ids : Array[Int],
  token_count : Int,
  maximum_token_id : Int,
) -> Input {
  TokenIds({
    backing: LunaOwnedTokenBacking(ReadOnlyArray::from_array(token_ids)),
    token_count,
    maximum_token_id,
  })
}

///|
fn forged_text_input(text : String, utf8 : Bytes) -> Input {
  Text({ text, utf8 })
}

///|
fn validate_input_and_seed(
  input~ : Input,
  sampling_seed~ : SamplingSeed,
  stops~ : StopConditions,
) -> Unit raise {
  let cfg = authority_limits()
  ignore(
    GenerateRequest::new(
      protocol_version=ProtocolVersion::v1(),
      request_id=RequestId::new(1UL),
      model_identity=authority_model_identity(),
      input~,
      max_new_tokens=2,
      context_ceiling=8,
      sampling=SamplingParameters::greedy(),
      sampling_seed~,
      stops~,
      stream_preference=Buffered,
      deadline=DeadlineBudget::from_milliseconds(10UL, cfg),
      cache=valid_authority_cache(),
      trace=None,
      limits=cfg,
    ),
  )
}

///|
test "GenerateRequest rejects forged scalar authority in precedence order" {
  try
    validate_authority_request(
      protocol_version={ wire_code: 2 },
      request_id={ value: 0UL },
      max_new_tokens=0,
      sampling=SamplingParameters::greedy(),
      stops=valid_authority_stops(),
      deadline={ milliseconds: 10UL },
      cache=valid_authority_cache(),
      trace=None,
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=ProtocolVersion, issue=Unsupported, index=0)",
      )
  } noraise {
    _ => fail("expected protocol precedence")
  }
  try
    validate_authority_request(
      protocol_version=ProtocolVersion::v1(),
      request_id={ value: 0UL },
      max_new_tokens=0,
      sampling=SamplingParameters::greedy(),
      stops=valid_authority_stops(),
      deadline={ milliseconds: 10UL },
      cache=valid_authority_cache(),
      trace=None,
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=RequestId, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected request identity precedence")
  }
}

///|
test "GenerateRequest fully revalidates forged stop conditions" {
  let cfg = authority_limits()
  let cases = [
    (
      forged_stops([1, 1], []),
      "Invalid(field=StopToken, issue=Duplicate, index=1)",
    ),
    (
      forged_stops([100, 100], [""]),
      "Invalid(field=StopToken, issue=OutOfRange, index=0)",
    ),
    (forged_stops([], [""]), "Invalid(field=StopString, issue=Empty, index=0)"),
    (
      forged_stops([], ["ééé"]),
      "Invalid(field=StopString, issue=TooLong, index=0)",
    ),
    (
      forged_stops([], ["stop", "stop"]),
      "Invalid(field=StopString, issue=Duplicate, index=1)",
    ),
  ]
  for case in cases {
    let (forged, expected) = case
    try
      validate_authority_request(
        protocol_version=ProtocolVersion::v1(),
        request_id=RequestId::new(1UL),
        max_new_tokens=2,
        sampling=SamplingParameters::greedy(),
        stops=forged,
        deadline=DeadlineBudget::from_milliseconds(10UL, cfg),
        cache=valid_authority_cache(),
        trace=None,
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected forged stop rejection")
    }
  }
}

///|
test "GenerateRequest authenticates forged owned token structure and cache" {
  let cases = [
    (
      forged_owned_input([1], 0, 1),
      "Invalid(field=InputToken, issue=Empty, index=0)",
    ),
    (
      forged_owned_input([1], 2, 1),
      "Invalid(field=LunaTokenStorage, issue=LunaStorageStale, index=0)",
    ),
    (
      forged_owned_input([-1], 1, -1),
      "Invalid(field=InputToken, issue=OutOfRange, index=0)",
    ),
    (
      forged_owned_input([5], 1, 1),
      "Invalid(field=InputToken, issue=Inconsistent, index=0)",
    ),
  ]
  for case in cases {
    let (input, expected) = case
    try
      validate_input_and_seed(
        input~,
        sampling_seed=SamplingSeed::new(1UL),
        stops=valid_authority_stops(),
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected forged owned token rejection")
    }
  }
}

///|
test "GenerateRequest revalidates retained text bytes and text agreement" {
  let cases = [
    (
      forged_text_input("retained", b""),
      "Invalid(field=InputText, issue=Empty, index=0)",
    ),
    (
      forged_text_input("retained", Bytes::make(17, b'a')),
      "Invalid(field=InputText, issue=TooLong, index=0)",
    ),
    (
      forged_text_input("retained", Bytes::from_array([0xff])),
      "Invalid(field=InputText, issue=InvalidFormat, index=0)",
    ),
    (
      forged_text_input("retained", b"different"),
      "Invalid(field=InputText, issue=Inconsistent, index=0)",
    ),
  ]
  for case in cases {
    let (input, expected) = case
    try
      validate_input_and_seed(
        input~,
        sampling_seed=SamplingSeed::new(1UL),
        stops=valid_authority_stops(),
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected forged text rejection")
    }
  }
}

///|
test "GenerateRequest validates sampling seed before stop conditions" {
  try
    validate_input_and_seed(
      input=forged_owned_input([1], 1, 1),
      sampling_seed={ value: 0UL },
      stops=forged_stops([], [""]),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=SamplingSeed, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected sampling seed precedence")
  }
}

///|
test "GenerateRequest rejects noncanonical forged greedy sampling" {
  let cfg = authority_limits()
  let cases = [
    (
      { mode: Greedy, temperature: 0.5, top_k: Some(1), top_p: Some(0.5) },
      "Invalid(field=Temperature, issue=Inconsistent, index=0)",
    ),
    (
      { mode: Greedy, temperature: 1.0, top_k: Some(1), top_p: Some(0.5) },
      "Invalid(field=TopK, issue=Inconsistent, index=0)",
    ),
    (
      { mode: Greedy, temperature: 1.0, top_k: None, top_p: Some(0.5) },
      "Invalid(field=TopP, issue=Inconsistent, index=0)",
    ),
  ]
  for case in cases {
    let (forged, expected) = case
    try
      validate_authority_request(
        protocol_version=ProtocolVersion::v1(),
        request_id=RequestId::new(1UL),
        max_new_tokens=2,
        sampling=forged,
        stops=valid_authority_stops(),
        deadline=DeadlineBudget::from_milliseconds(10UL, cfg),
        cache=valid_authority_cache(),
        trace=None,
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected forged greedy sampling rejection")
    }
  }
}

///|
test "GenerateRequest validates deadline cache and trace by UTF-8 bytes" {
  let cfg = authority_limits()
  try
    validate_authority_request(
      protocol_version=ProtocolVersion::v1(),
      request_id=RequestId::new(1UL),
      max_new_tokens=2,
      sampling=SamplingParameters::greedy(),
      stops=forged_stops([], [""]),
      deadline={ milliseconds: 0UL },
      cache=CachePolicy::new({ value: "" }, ReadWrite),
      trace=Some({ value: "" }),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=Empty, index=0)",
      )
  } noraise {
    _ => fail("expected stops before deadline")
  }
  try
    validate_authority_request(
      protocol_version=ProtocolVersion::v1(),
      request_id=RequestId::new(1UL),
      max_new_tokens=2,
      sampling=SamplingParameters::greedy(),
      stops=valid_authority_stops(),
      deadline={ milliseconds: 0UL },
      cache=CachePolicy::new({ value: "" }, ReadWrite),
      trace=Some({ value: "" }),
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=DeadlineBudget, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected deadline before cache")
  }
  for
    case in [
      ("", "Invalid(field=CacheScope, issue=Empty, index=0)"),
      ("ééééé", "Invalid(field=CacheScope, issue=TooLong, index=0)"),
      ("bad/x", "Invalid(field=CacheScope, issue=InvalidFormat, index=0)"),
    ] {
    let (scope, expected) = case
    try
      validate_authority_request(
        protocol_version=ProtocolVersion::v1(),
        request_id=RequestId::new(1UL),
        max_new_tokens=2,
        sampling=SamplingParameters::greedy(),
        stops=valid_authority_stops(),
        deadline=DeadlineBudget::from_milliseconds(10UL, cfg),
        cache=CachePolicy::new({ value: scope }, ReadWrite),
        trace=Some({ value: "" }),
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected cache scope rejection before trace")
    }
  }
  for
    case in [
      ("", "Invalid(field=TraceCorrelation, issue=Empty, index=0)"),
      ("ééééé", "Invalid(field=TraceCorrelation, issue=TooLong, index=0)"),
      ("bad/x", "Invalid(field=TraceCorrelation, issue=InvalidFormat, index=0)"),
    ] {
    let (trace, expected) = case
    try
      validate_authority_request(
        protocol_version=ProtocolVersion::v1(),
        request_id=RequestId::new(1UL),
        max_new_tokens=2,
        sampling=SamplingParameters::greedy(),
        stops=valid_authority_stops(),
        deadline=DeadlineBudget::from_milliseconds(10UL, cfg),
        cache=valid_authority_cache(),
        trace=Some({ value: trace }),
      )
    catch {
      error => debug_inspect(error, content=expected)
    } noraise {
      _ => fail("expected trace rejection")
    }
  }
}
