///|
fn phase4_fixture_digest(unit : Char) -> String {
  unit.to_string().repeat(64)
}

///|
fn phase4_fixture_profile_name(index : Int) -> String {
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
fn phase4_fixture_engine_name(index : Int) -> String {
  match index {
    0 => "lunaflux"
    1 => "vllm"
    _ => "sglang"
  }
}

///|
fn phase4_fixture_order_position(engine : Int, ordinal : Int) -> Int {
  match ordinal {
    1 => engine
    2 => (engine + 2) % 3
    _ => (engine + 1) % 3
  }
}

///|
fn phase4_fixture_declaration() -> Bytes {
  let engines = StringBuilder()
  for index in 0..<3 {
    if index > 0 {
      engines.write_string(",")
    }
    engines.write_string(
      "{\"engine\":\"" +
      phase4_fixture_engine_name(index) +
      "\",\"endpoint\":\"127.0.0.1:" +
      (8100 + index).to_string() +
      "\",\"revision_sha256\":\"" +
      phase4_fixture_digest('a') +
      "\",\"image_sha256\":\"" +
      phase4_fixture_digest('b') +
      "\",\"configuration_sha256\":\"" +
      phase4_fixture_digest('c') +
      "\",\"executable_sha256\":\"" +
      phase4_fixture_digest('f') +
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
      phase4_fixture_profile_name(index) +
      "\",\"workload_sha256\":\"" +
      phase4_fixture_digest('d') +
      "\",\"request_input_tokens\":[\"1\"]}",
    )
  }
  @utf8.encode(
    "{\"schema\":\"lunaflux-openai-responses-campaign-declaration.v1\"," +
    "\"credential_ingress\":\"external.inherited-descriptor.v1\"," +
    "\"credential_policy_sha256\":\"" +
    phase4_fixture_digest('e') +
    "\",\"hardware_sha256\":\"" +
    phase4_fixture_digest('1') +
    "\",\"driver_sha256\":\"" +
    phase4_fixture_digest('2') +
    "\",\"toolkit_sha256\":\"" +
    phase4_fixture_digest('3') +
    "\",\"model_sha256\":\"" +
    phase4_fixture_digest('4') +
    "\",\"tokenizer_sha256\":\"" +
    phase4_fixture_digest('5') +
    "\",\"corpus_sha256\":\"" +
    phase4_fixture_digest('6') +
    "\",\"protocol_sha256\":\"" +
    phase4_fixture_digest('7') +
    "\",\"randomization_sha256\":\"" +
    phase4_fixture_digest('8') +
    "\",\"engines\":[" +
    engines.to_string() +
    "],\"profiles\":[" +
    profiles.to_string() +
    "]}",
  )
}

///|
fn phase4_fixture_event(
  kind : String,
  timestamp : Int,
  input_tokens : String,
  status : String,
  content_type : String,
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
  "\",\"outcome\":\"\",\"bytes_hex\":\"" +
  bytes_hex +
  "\"}"
}

///|
fn phase4_fixture_elapsed(
  profile : Int,
  engine : Int,
  rich_lunaflux_ns : Int,
) -> Int {
  if profile == 4 {
    match engine {
      0 => rich_lunaflux_ns
      1 => 110
      _ => 120
    }
  } else if profile == 5 {
    match engine {
      0 => 105
      1 => 100
      _ => 110
    }
  } else {
    100 + engine
  }
}

///|
fn phase4_fixture_capture(
  declaration_sha256 : String,
  index : Int,
  rich_lunaflux_ns : Int,
) -> Bytes {
  let profile = index / 9
  let ordinal = index % 9 / 3 + 1
  let engine = index % 3
  let base = 1000 + index * 1000
  let elapsed = phase4_fixture_elapsed(profile, engine, rich_lunaflux_ns)
  let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n" +
    "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"x\"}\n\n" +
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"finish_reason\":\"stop\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n" +
    "data: [DONE]\n\n"
  let events = phase4_fixture_event("submit", base, "1", "0", "", "") +
    "," +
    phase4_fixture_event(
      "response_head",
      base + 1,
      "0",
      "200",
      "text/event-stream",
      "",
    ) +
    "," +
    phase4_fixture_event(
      "sse_bytes",
      base + elapsed,
      "0",
      "0",
      "",
      @crypto.bytes_to_hex_string(@utf8.encode(sse)),
    )
  @utf8.encode(
    "{\"schema\":\"lunaflux-openai-responses-captured-trial.v1\"," +
    "\"declaration_sha256\":\"" +
    declaration_sha256 +
    "\",\"trial_index\":\"" +
    index.to_string() +
    "\",\"profile\":\"" +
    phase4_fixture_profile_name(profile) +
    "\",\"engine\":\"" +
    phase4_fixture_engine_name(engine) +
    "\",\"ordinal\":\"" +
    ordinal.to_string() +
    "\",\"order_position\":\"" +
    phase4_fixture_order_position(engine, ordinal).to_string() +
    "\",\"events\":[" +
    events +
    "]}",
  )
}

///|
fn phase4_fixture_correctness() -> Array[@evidence.BenchmarkDigest] {
  [
    for index in 0..<PHASE4_CORRECTNESS_TRIALS => {
      let suffix = (index + 1).to_string(radix=16)
      try! @evidence.BenchmarkDigest::from_sha256(
        "0".repeat(64 - suffix.length()) + suffix,
      )
    }
  ]
}

///|
fn phase4_fixture_correctness_artifact(
  declaration_sha256 : String,
  capture_sha256 : String,
  trial_index : Int,
  passed? : String = "1",
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux.external-correctness-observation.v1\n" +
    "declaration_sha256=" +
    declaration_sha256 +
    "\ntrial_index=" +
    trial_index.to_string() +
    "\ncapture_sha256=" +
    capture_sha256 +
    "\nreference_sha256=" +
    phase4_fixture_digest('e') +
    "\ncorrectness_passed=" +
    passed +
    "\n",
  )
}

///|
fn phase4_fixture_measurements(
  rich_lunaflux_ns : Int,
) -> @openai_adapter.OpenAIResponsesComparisonMeasurementSet {
  let declaration_bytes = phase4_fixture_declaration()
  let declaration_sha256 = @crypto.bytes_to_hex_string(
    @crypto.sha256(declaration_bytes),
  )
  let declaration = try! @openai_campaign.OpenAIResponsesCampaignDeclaration::parse(
    declaration_bytes, declaration_sha256,
  )
  let captures = [
    for index in 0..<PHASE4_CORRECTNESS_TRIALS => {
      phase4_fixture_capture(declaration_sha256, index, rich_lunaflux_ns)
    }
  ]
  (try! @openai_campaign.replay_openai_responses_campaign(declaration, captures)).measurements()
}

///|
fn phase4_fixture_policy(budget? : Int = 1000) -> Phase4ExternalPolicy {
  {
    digest: phase4_fixture_digest('9'),
    declaration: phase4_fixture_digest('a'),
    hardware: phase4_fixture_digest('1'),
    driver: phase4_fixture_digest('2'),
    toolkit: phase4_fixture_digest('3'),
    model: phase4_fixture_digest('4'),
    tokenizer: phase4_fixture_digest('5'),
    corpus: phase4_fixture_digest('6'),
    protocol: phase4_fixture_digest('7'),
    randomization: phase4_fixture_digest('8'),
    prefix_rich_workload: phase4_fixture_digest('d'),
    prefix_cold_workload: phase4_fixture_digest('d'),
    engines: [
      {
        name: "lunaflux",
        revision: phase4_fixture_digest('a'),
        image: phase4_fixture_digest('b'),
        configuration: phase4_fixture_digest('c'),
        executable: phase4_fixture_digest('f'),
      },
      {
        name: "vllm",
        revision: phase4_fixture_digest('a'),
        image: phase4_fixture_digest('b'),
        configuration: phase4_fixture_digest('c'),
        executable: phase4_fixture_digest('f'),
      },
      {
        name: "sglang",
        revision: phase4_fixture_digest('a'),
        image: phase4_fixture_digest('b'),
        configuration: phase4_fixture_digest('c'),
        executable: phase4_fixture_digest('f'),
      },
    ],
    capture_inventory: phase4_fixture_digest('a'),
    correctness_inventory: phase4_fixture_digest('9'),
    campaign_tool: phase4_fixture_digest('b'),
    correctness_verifier: phase4_fixture_digest('c'),
    identity_verifier: phase4_fixture_digest('d'),
    process_supervisor: phase4_fixture_digest('e'),
    gate_executable: phase4_fixture_digest('f'),
    fd_exec_helper: phase4_fixture_digest('1'),
    cold_regression_budget_bps: budget,
  }
}

///|
test "Phase 4 rational comparison remains exact without cross multiplication" {
  assert_eq(phase4_fraction_compare(1L, 3L, 2L, 5L), -1)
  assert_eq(phase4_fraction_compare(7L, 13L, 14L, 26L), 0)
  assert_eq(
    phase4_fraction_compare(
      9223372036854775806L, 9223372036854775807L, 9223372036854775805L, 9223372036854775806L,
    ),
    1,
  )
}

///|
test "Phase 4 correctness artifacts and lexical inventories fail closed" {
  let declaration = phase4_fixture_digest('a')
  let capture = phase4_fixture_digest('b')
  let digest = phase4_correctness_digest(
    phase4_fixture_correctness_artifact(declaration, capture, 10),
    declaration,
    capture,
    10,
  )
  phase4_inventory_entry(
    [
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff  correctness/trial-1.correctness.v1",
      digest.as_hex() + "  correctness/trial-10.correctness.v1",
      "",
    ],
    "correctness/trial-10.correctness.v1",
    digest,
  )
  try
    phase4_correctness_digest(
      phase4_fixture_correctness_artifact(declaration, capture, 10, passed="0"),
      declaration,
      capture,
      10,
    )
  catch {
    Phase4InvalidCorrectness => ()
    _ => fail("unexpected Phase 4 correctness rejection")
  } noraise {
    _ => fail("failed external correctness artifact was admitted")
  }
}

///|
test "Phase 4 gate records measured prefix parity and cold budget" {
  let correctness = phase4_fixture_correctness()
  let evidence = phase4_gate_evidence(
    phase4_fixture_measurements(100),
    correctness,
    phase4_fixture_policy(),
  )
  assert_true(evidence.passed())
  let canonical = try! @utf8.decode(evidence.canonical_bytes())
  assert_true(canonical.contains("prefix_rich_lunaflux_elapsed_ns=300"))
  assert_true(canonical.contains("prefix_cold_lunaflux_elapsed_ns=315"))
  assert_true(canonical.contains("prefix_rich_parity=pass"))
  assert_true(canonical.contains("prefix_cold_regression=pass"))
  assert_true(canonical.contains("release_authority=none"))
}

///|
test "Phase 4 gate preserves losing measurements as a failed result" {
  let evidence = phase4_gate_evidence(
    phase4_fixture_measurements(130),
    phase4_fixture_correctness(),
    phase4_fixture_policy(),
  )
  assert_false(evidence.passed())
  let canonical = try! @utf8.decode(evidence.canonical_bytes())
  assert_true(canonical.contains("prefix_rich_lunaflux_elapsed_ns=390"))
  assert_true(canonical.contains("prefix_rich_parity=fail"))
  assert_true(canonical.contains("phase4_prefix_gate=fail"))
}

///|
test "Phase 4 cold regression budget cannot become vacuous" {
  try
    phase4_gate_evidence(
      phase4_fixture_measurements(100),
      phase4_fixture_correctness(),
      phase4_fixture_policy(budget=PHASE4_MAX_COLD_REGRESSION_BPS + 1),
    )
  catch {
    Phase4InvalidPolicy => ()
    _ => fail("unexpected Phase 4 policy rejection")
  } noraise {
    _ => fail("overbroad cold regression budget was admitted")
  }
}
