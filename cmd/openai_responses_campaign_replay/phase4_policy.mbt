///|
const PHASE4_POLICY_LINES : Int = 39

///|
fn phase4_policy_digest(value : String) -> String raise Phase4PrefixGateError {
  ignore(
    @evidence.BenchmarkDigest::from_sha256(value) catch {
      _ => raise Phase4InvalidPolicy
    },
  )
  value
}

///|
fn phase4_policy_engine(
  lines : ArrayView[StringView],
  base : Int,
  expected_name : String,
) -> Phase4PolicyEngine raise Phase4PrefixGateError {
  let name = phase4_require_field(
    lines,
    base,
    "engine_" + ((base - 12) / 5).to_string(),
  )
  if name != expected_name {
    raise Phase4InvalidPolicy
  }
  let prefix = "engine_" + ((base - 12) / 5).to_string() + "_"
  {
    name,
    revision: phase4_policy_digest(
      phase4_require_field(lines, base + 1, prefix + "revision_sha256"),
    ),
    image: phase4_policy_digest(
      phase4_require_field(lines, base + 2, prefix + "image_sha256"),
    ),
    configuration: phase4_policy_digest(
      phase4_require_field(lines, base + 3, prefix + "configuration_sha256"),
    ),
    executable: phase4_policy_digest(
      phase4_require_field(lines, base + 4, prefix + "executable_sha256"),
    ),
  }
}

///|
fn phase4_parse_policy(
  bytes : Bytes,
  expected_sha256 : String,
) -> Phase4ExternalPolicy raise Phase4PrefixGateError {
  let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  if actual != expected_sha256 {
    raise Phase4InvalidPolicy
  }
  let document = @utf8.decode(bytes) catch { _ => raise Phase4InvalidPolicy }
  let lines = document.split("\n").to_array()
  if lines.length() != PHASE4_POLICY_LINES + 1 ||
    lines[PHASE4_POLICY_LINES] != "" ||
    phase4_require_field(lines, 0, "schema") !=
    "lunaflux.phase4-prefix-benchmark-policy.v1" {
    raise Phase4InvalidPolicy
  }
  let budget_text = phase4_require_field(
    lines, 35, "cold_regression_budget_bps",
  )
  let budget = @string.parse_int(budget_text, base=10) catch {
    _ => raise Phase4InvalidPolicy
  }
  if budget < 0 ||
    budget > PHASE4_MAX_COLD_REGRESSION_BPS ||
    budget.to_string() != budget_text ||
    phase4_require_field(lines, 36, "comparison_metric") !=
    "aggregate_completed_requests_per_nanosecond" ||
    phase4_require_field(lines, 37, "physical_measurement_claim") != "none" ||
    phase4_require_field(lines, 38, "authority") !=
    "phase4-measurement-gate-only" {
    raise Phase4InvalidPolicy
  }
  {
    digest: phase4_policy_digest(expected_sha256),
    declaration: phase4_policy_digest(
      phase4_require_field(lines, 1, "declaration_sha256"),
    ),
    hardware: phase4_policy_digest(
      phase4_require_field(lines, 2, "hardware_sha256"),
    ),
    driver: phase4_policy_digest(
      phase4_require_field(lines, 3, "driver_sha256"),
    ),
    toolkit: phase4_policy_digest(
      phase4_require_field(lines, 4, "toolkit_sha256"),
    ),
    model: phase4_policy_digest(phase4_require_field(lines, 5, "model_sha256")),
    tokenizer: phase4_policy_digest(
      phase4_require_field(lines, 6, "tokenizer_sha256"),
    ),
    corpus: phase4_policy_digest(
      phase4_require_field(lines, 7, "corpus_sha256"),
    ),
    protocol: phase4_policy_digest(
      phase4_require_field(lines, 8, "protocol_sha256"),
    ),
    randomization: phase4_policy_digest(
      phase4_require_field(lines, 9, "randomization_sha256"),
    ),
    prefix_rich_workload: phase4_policy_digest(
      phase4_require_field(lines, 10, "prefix_rich_workload_sha256"),
    ),
    prefix_cold_workload: phase4_policy_digest(
      phase4_require_field(lines, 11, "prefix_cold_workload_sha256"),
    ),
    engines: [
      phase4_policy_engine(lines, 12, "lunaflux"),
      phase4_policy_engine(lines, 17, "vllm"),
      phase4_policy_engine(lines, 22, "sglang"),
    ],
    capture_inventory: phase4_policy_digest(
      phase4_require_field(lines, 27, "capture_inventory_sha256"),
    ),
    correctness_inventory: phase4_policy_digest(
      phase4_require_field(lines, 28, "correctness_inventory_sha256"),
    ),
    campaign_tool: phase4_policy_digest(
      phase4_require_field(lines, 29, "campaign_tool_sha256"),
    ),
    correctness_verifier: phase4_policy_digest(
      phase4_require_field(lines, 30, "correctness_verifier_sha256"),
    ),
    identity_verifier: phase4_policy_digest(
      phase4_require_field(lines, 31, "engine_identity_verifier_sha256"),
    ),
    process_supervisor: phase4_policy_digest(
      phase4_require_field(lines, 32, "process_supervisor_sha256"),
    ),
    gate_executable: phase4_policy_digest(
      phase4_require_field(lines, 33, "phase4_gate_executable_sha256"),
    ),
    fd_exec_helper: phase4_policy_digest(
      phase4_require_field(lines, 34, "phase4_fd_exec_helper_sha256"),
    ),
    cold_regression_budget_bps: budget,
  }
}

///|
fn phase4_json_string(
  object : Map[String, Json],
  key : String,
) -> String raise Phase4PrefixGateError {
  guard object.get(key) is Some(String(value)) else {
    raise Phase4InvalidPolicy
  }
  value
}

///|
fn phase4_json_array(
  object : Map[String, Json],
  key : String,
) -> Array[Json] raise Phase4PrefixGateError {
  guard object.get(key) is Some(Array(value)) else { raise Phase4InvalidPolicy }
  value
}

///|
fn phase4_join_declaration(
  policy : Phase4ExternalPolicy,
  declaration_bytes : Bytes,
) -> Unit raise Phase4PrefixGateError {
  let document = @utf8.decode(declaration_bytes) catch {
    _ => raise Phase4InvalidPolicy
  }
  let parsed = @json.parse(document) catch { _ => raise Phase4InvalidPolicy }
  guard parsed is Object(root) else { raise Phase4InvalidPolicy }
  let scalar_keys : ReadOnlyArray[String] = [
    "hardware_sha256", "driver_sha256", "toolkit_sha256", "model_sha256", "tokenizer_sha256",
    "corpus_sha256", "protocol_sha256", "randomization_sha256",
  ]
  let expected : ReadOnlyArray[String] = [
    policy.hardware,
    policy.driver,
    policy.toolkit,
    policy.model,
    policy.tokenizer,
    policy.corpus,
    policy.protocol,
    policy.randomization,
  ]
  for index, key in scalar_keys {
    if phase4_json_string(root, key) != expected[index] {
      raise Phase4InvalidPolicy
    }
  }
  let engines = phase4_json_array(root, "engines")
  if engines.length() != 3 {
    raise Phase4InvalidPolicy
  }
  for index, expected_engine in policy.engines {
    guard engines[index] is Object(engine) else { raise Phase4InvalidPolicy }
    if phase4_json_string(engine, "engine") != expected_engine.name ||
      phase4_json_string(engine, "revision_sha256") != expected_engine.revision ||
      phase4_json_string(engine, "image_sha256") != expected_engine.image ||
      phase4_json_string(engine, "configuration_sha256") !=
      expected_engine.configuration ||
      phase4_json_string(engine, "executable_sha256") !=
      expected_engine.executable {
      raise Phase4InvalidPolicy
    }
  }
  let profiles = phase4_json_array(root, "profiles")
  if profiles.length() != 9 {
    raise Phase4InvalidPolicy
  }
  guard profiles[4] is Object(rich) && profiles[5] is Object(cold) else {
    raise Phase4InvalidPolicy
  }
  if phase4_json_string(rich, "profile") != "prefix_rich" ||
    phase4_json_string(rich, "workload_sha256") != policy.prefix_rich_workload ||
    phase4_json_string(cold, "profile") != "prefix_cold" ||
    phase4_json_string(cold, "workload_sha256") != policy.prefix_cold_workload {
    raise Phase4InvalidPolicy
  }
}

///|
async fn phase4_join_handoff(
  root : String,
  policy : Phase4ExternalPolicy,
) -> Unit raise Phase4PrefixGateError {
  let bytes = read_regular_capture(root, "comparison-handoff.v1") catch {
    _ => raise Phase4InvalidPolicy
  }
  if bytes.length() <= 0 || bytes.length() > 16384 {
    raise Phase4InvalidPolicy
  }
  let document = @utf8.decode(bytes) catch { _ => raise Phase4InvalidPolicy }
  let lines = document.split("\n").to_array()
  if lines.length() != 26 ||
    lines[25] != "" ||
    phase4_require_field(lines, 0, "schema") !=
    "lunaflux.openai-comparison-handoff.v1" ||
    phase4_require_field(lines, 1, "declaration_sha256") != policy.declaration ||
    phase4_require_field(lines, 4, "campaign_tool_sha256") !=
    policy.campaign_tool ||
    phase4_require_field(lines, 6, "correctness_verifier_sha256") !=
    policy.correctness_verifier ||
    phase4_require_field(lines, 7, "engine_identity_verifier_sha256") !=
    policy.identity_verifier ||
    phase4_require_field(lines, 8, "process_supervisor_sha256") !=
    policy.process_supervisor ||
    phase4_require_field(lines, 9, "capture_inventory_sha256") !=
    policy.capture_inventory ||
    phase4_require_field(lines, 10, "correctness_inventory_sha256") !=
    policy.correctness_inventory ||
    phase4_require_field(lines, 16, "trial_count") != "81" ||
    phase4_require_field(lines, 19, "counterbalance") != "latin-square-v1" ||
    phase4_require_field(lines, 20, "protocol") != "openai.responses.sse.v1" ||
    phase4_require_field(lines, 21, "process_cleanup_complete") != "1" ||
    phase4_require_field(lines, 23, "comparison_authority") != "none" ||
    phase4_require_field(lines, 24, "physical_measurement_claim") != "none" {
    raise Phase4InvalidPolicy
  }
}
