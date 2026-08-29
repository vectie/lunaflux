///|
fn campaign_test_digest(unit : Char) -> String {
  unit.to_string().repeat(64)
}

///|
fn campaign_test_profile_name(index : Int) -> String {
  match index {
    0 => "latency"
    1 => "chat"
    2 => "long_prefill"
    3 => "decode_heavy"
    4 => "prefix_rich"
    5 => "prefix_cold"
    6 => "saturation"
    7 => "churn"
    _ => "mixed"
  }
}

///|
fn campaign_test_engine_name(index : Int) -> String {
  match index {
    0 => "lunaflux"
    1 => "vllm"
    _ => "sglang"
  }
}

///|
fn campaign_test_declaration(endpoint_host? : String = "127.0.0.1") -> Bytes {
  let engines = StringBuilder()
  for index in 0..<3 {
    if index > 0 {
      engines.write_string(",")
    }
    engines.write_string(
      "{\"engine\":\"" +
      campaign_test_engine_name(index) +
      "\",\"endpoint\":\"" +
      endpoint_host +
      ":" +
      (8100 + index).to_string() +
      "\",\"revision_sha256\":\"" +
      campaign_test_digest('a') +
      "\",\"image_sha256\":\"" +
      campaign_test_digest('b') +
      "\",\"configuration_sha256\":\"" +
      campaign_test_digest('c') +
      "\",\"executable_sha256\":\"" +
      campaign_test_digest('f') +
      "\"}",
    )
  }
  let profiles = StringBuilder()
  for index in 0..<9 {
    if index > 0 {
      profiles.write_string(",")
    }
    profiles.write_string(
      "{\"profile\":\"" +
      campaign_test_profile_name(index) +
      "\",\"workload_sha256\":\"" +
      campaign_test_digest('d') +
      "\",\"request_input_tokens\":[\"1\"]}",
    )
  }
  @utf8.encode(
    "{\"schema\":\"lunaflux-openai-responses-campaign-declaration.v1\"," +
    "\"credential_ingress\":\"external.inherited-descriptor.v1\"," +
    "\"credential_policy_sha256\":\"" +
    campaign_test_digest('e') +
    "\",\"hardware_sha256\":\"" +
    campaign_test_digest('1') +
    "\",\"driver_sha256\":\"" +
    campaign_test_digest('2') +
    "\",\"toolkit_sha256\":\"" +
    campaign_test_digest('3') +
    "\",\"model_sha256\":\"" +
    campaign_test_digest('4') +
    "\",\"tokenizer_sha256\":\"" +
    campaign_test_digest('5') +
    "\",\"corpus_sha256\":\"" +
    campaign_test_digest('6') +
    "\",\"protocol_sha256\":\"" +
    campaign_test_digest('7') +
    "\",\"randomization_sha256\":\"" +
    campaign_test_digest('8') +
    "\",\"engines\":[" +
    engines.to_string() +
    "],\"profiles\":[" +
    profiles.to_string() +
    "]}",
  )
}

///|
fn campaign_test_event(
  kind : String,
  timestamp : Int,
  input_tokens : String,
  status : String,
  content_type : String,
  outcome : String,
  bytes_hex : String,
) -> String {
  "{\"kind\":\"" +
  kind +
  "\",\"request_ordinal\":\"0\",\"timestamp_ns\":\"" +
  timestamp.to_string() +
  "\",\"input_tokens\":\"" +
  input_tokens +
  "\",\"status\":\"" +
  status +
  "\",\"content_type\":\"" +
  content_type +
  "\",\"outcome\":\"" +
  outcome +
  "\",\"bytes_hex\":\"" +
  bytes_hex +
  "\"}"
}

///|
fn campaign_test_capture(
  declaration_digest : String,
  index : Int,
  reversed_head? : Bool = false,
) -> Bytes {
  let profile_index = index / 9
  let ordinal = index % 9 / 3 + 1
  let engine_index = index % 3
  let base = 100 + index * 10
  let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n" +
    "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"x\"}\n\n" +
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"finish_reason\":\"stop\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n" +
    "data: [DONE]\n\n"
  let events = campaign_test_event("submit", base, "1", "0", "", "", "") +
    "," +
    campaign_test_event(
      "response_head",
      if reversed_head {
        base - 1
      } else {
        base + 1
      },
      "0",
      "200",
      "text/event-stream",
      "",
      "",
    ) +
    "," +
    campaign_test_event(
      "sse_bytes",
      base + 2,
      "0",
      "0",
      "",
      "",
      @crypto.bytes_to_hex_string(@utf8.encode(sse)),
    )
  @utf8.encode(
    "{\"schema\":\"lunaflux-openai-responses-captured-trial.v1\"," +
    "\"declaration_sha256\":\"" +
    declaration_digest +
    "\",\"trial_index\":\"" +
    index.to_string() +
    "\",\"profile\":\"" +
    campaign_test_profile_name(profile_index) +
    "\",\"engine\":\"" +
    campaign_test_engine_name(engine_index) +
    "\",\"ordinal\":\"" +
    ordinal.to_string() +
    "\",\"order_position\":\"" +
    campaign_order_position(engine_index, ordinal).to_string() +
    "\",\"events\":[" +
    events +
    "]}",
  )
}

///|
fn campaign_test_fixture() -> (OpenAIResponsesCampaignDeclaration, Array[Bytes]) {
  let bytes = campaign_test_declaration()
  let digest = campaign_sha256(bytes).as_hex()
  let declaration = try! OpenAIResponsesCampaignDeclaration::parse(
    bytes, digest,
  )
  let captures = [ for index in 0..<81 => campaign_test_capture(digest, index) ]
  (declaration, captures)
}

///|
test "exact 81-trial captured Responses campaign remains authority-free" {
  let (declaration, captures) = campaign_test_fixture()
  assert_eq(
    (try! declaration.engine_executable(0)).as_hex(),
    campaign_test_digest('f'),
  )
  let replay = try! replay_openai_responses_campaign(declaration, captures)
  assert_eq(replay.measurements().trial_count(), 81)
  let canonical = try! @utf8.decode(replay.canonical_bytes())
  assert_true(canonical.contains("timestamp_authority=external_live_runner"))
  assert_true(canonical.contains("raw_artifact_mutation=none"))
  assert_true(canonical.contains("comparison_authority=none"))
  assert_true(canonical.contains("correctness_authority=none"))
}

///|
test "declaration digest and exact loopback endpoint fail closed" {
  let bytes = campaign_test_declaration()
  try
    OpenAIResponsesCampaignDeclaration::parse(bytes, campaign_test_digest('0'))
  catch {
    CampaignInvalidDigest => ()
    _ => fail("unexpected declaration-digest rejection")
  } noraise {
    _ => fail("unauthenticated declaration was admitted")
  }
  let remote = campaign_test_declaration(endpoint_host="192.0.2.1")
  try
    OpenAIResponsesCampaignDeclaration::parse(
      remote,
      campaign_sha256(remote).as_hex(),
    )
  catch {
    CampaignInvalidEndpoint => ()
    _ => fail("unexpected non-loopback rejection")
  } noraise {
    _ => fail("non-loopback endpoint was admitted")
  }
}

///|
test "capture count and captured monotonic order fail closed" {
  let (declaration, captures) = campaign_test_fixture()
  try replay_openai_responses_campaign(declaration, captures[0:80]) catch {
    CampaignInvalidShape => ()
    _ => fail("unexpected capture-count rejection")
  } noraise {
    _ => fail("incomplete capture matrix was admitted")
  }
  let hostile = captures.copy()
  hostile[0] = campaign_test_capture(
    declaration.digest().as_hex(),
    0,
    reversed_head=true,
  )
  try replay_openai_responses_campaign(declaration, hostile) catch {
    CampaignAdapterRejected => ()
    _ => fail("unexpected captured-clock rejection")
  } noraise {
    _ => fail("reversed captured timestamp was admitted")
  }
}
