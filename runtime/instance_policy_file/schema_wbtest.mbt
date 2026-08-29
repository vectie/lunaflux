///|
/// Canonical complete v1 policy used only by strict-schema white-box tests.
fn policy_fixture() -> String {
  (
    #|{"schema_version":"lunaflux.instance-policy.v1","runtime_descriptor_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","tokenizer":{"locator":"tokenizer.json","sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","max_file_bytes":1048576,"max_json_depth":16,"max_input_bytes":4096,"max_output_tokens":2048,"max_decoded_bytes":4096,"max_vocab_entries":32000,"max_merge_rules":64000,"max_token_bytes":256,"max_special_tokens":64},"scheduler":{"step_token_budget":64,"max_active_requests":4,"max_waiting_requests":4,"prefill_chunk_tokens":32,"emergency_decode_page_reserve":1,"waiting_age_threshold_steps":8,"output_event_capacity":64},"cache":{"tokens_per_page":16,"total_page_count":128,"block_table_pages_per_request":8,"prefix_enabled":true,"max_prefix_entries":8,"max_prefix_nodes":128,"max_prefix_tokens_per_entry":128,"max_prefix_pages":64,"max_prefix_scope_bytes":256,"max_active_references_per_page":8,"max_cached_references_per_page":8,"layout_version":1},"service":{"listen_host":"127.0.0.1","listen_port":8080,"max_request_bytes":4096,"graceful_drain_milliseconds":1000,"diagnostic_mode":"normal"},"worker_process":{"max_frame_bytes":1048576,"startup_io_timeout_millis":1000,"io_timeout_millis":1000,"shutdown_timeout_millis":1000},"restart":{"initial_backoff_millis":1,"maximum_backoff_millis":1000,"stable_after_millis":1000,"maximum_attempts":3},"transport":{"read_chunk_bytes":4096,"write_chunk_bytes":4096,"accept_timeout_millis":1000,"input_idle_timeout_millis":1000,"write_timeout_millis":1000,"reactor_transition_budget":64},"preparation":{"framed_max_frame_bytes":4096,"lane_count":8,"step_work_units":17,"total_work_units":100000,"storage_int_cells":1000000,"storage_byte_cells":1000000,"storage_reference_cells":100000,"event_work_units":17},"telemetry":{"instance_log_capacity":1024}}
  )
}

///|
fn openai_policy_fixture() -> String {
  policy_fixture()
  .replace(old="lunaflux.instance-policy.v1", new="lunaflux.instance-policy.v2")
  .replace(
    old="\"worker_process\":",
    new="\"external_protocol\":{\"mode\":\"openai_responses_v1\",\"transport_security\":\"loopback_plaintext\",\"control_authentication\":\"deployment_bearer\",\"invocation_path\":\"/v1/responses\",\"health_path\":\"/health\",\"readiness_path\":\"/ready\",\"drain_path\":\"/v1/drain\",\"drain_method\":\"POST\"},\"worker_process\":",
  )
  .replace(old="\"/health\"", new="\"/healthz\"")
  .replace(old="\"/ready\"", new="\"/readyz\"")
}

///|
fn openai_v3_policy_fixture() -> String {
  openai_policy_fixture()
  .replace(old="lunaflux.instance-policy.v2", new="lunaflux.instance-policy.v3")
  .replace(
    old="\"worker_process\":",
    new="\"openai_service\":{\"maximum_credential_bytes\":4096,\"max_head_bytes\":4096,\"max_body_bytes\":4096,\"http_step_work_units\":17,\"response_step_work_units\":31,\"inbound_step_work_units\":17,\"outbound_step_work_units\":31,\"max_messages\":8,\"system_prefix\":\"\",\"system_suffix\":\"\",\"user_prefix\":\"\",\"user_suffix\":\"\",\"assistant_prefix\":\"\",\"assistant_suffix\":\"\",\"assistant_cue\":\"\",\"max_rendered_prompt_bytes\":4096,\"model_alias\":\"tiny-bf16\",\"response_id_prefix\":\"resp_lunaflux_\",\"cache_scope_ascii\":\"openai-v3\",\"max_new_tokens\":8,\"context_ceiling\":32,\"sampling_seed\":7,\"deadline_milliseconds\":60000},\"worker_process\":",
  )
}

///|
fn policy_test_digest() -> InstancePolicyDigest raise InstancePolicyFileError {
  InstancePolicyDigest::from_sha256(
    "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn parse_policy_fixture(
  text : String,
) -> InstancePolicyAdmission raise InstancePolicyFileError {
  parse_policy(
    @utf8.encode(text),
    policy_test_digest(),
    try! InstancePolicyFileLimits::new(
      max_policy_bytes=65536L,
      max_json_depth=16,
    ),
  )
}

///|
test "instance policy v1 admits every mandatory grouped claim" {
  let policy = try! parse_policy_fixture(policy_fixture())
  assert_eq(policy.runtime_descriptor_sha256().length(), 64)
  assert_eq(policy.tokenizer().locator(), "tokenizer.json")
  assert_eq(policy.scheduler().max_active_requests(), 4)
  assert_eq(policy.cache().max_prefix_scope_bytes(), 256)
  assert_eq(policy.preparation().lane_count(), 8)
  assert_eq(policy.telemetry().instance_log_capacity(), 1024)
}

///|
test "instance policy v2 binds exact external protocol intent" {
  let policy = try! parse_policy_fixture(openai_policy_fixture())
  let protocol = policy.external_protocol()
  assert_eq(protocol.mode(), OpenAiResponsesV1)
  assert_eq(protocol.transport_security(), LoopbackPlaintext)
  assert_eq(protocol.control_authentication(), DeploymentBearer)
  assert_eq(protocol.invocation_path(), "/v1/responses")
  assert_eq(protocol.health_path(), "/healthz")
  assert_eq(protocol.readiness_path(), "/readyz")
  assert_eq(protocol.drain_path(), "/v1/drain")
  assert_eq(protocol.drain_method(), "POST")
}

///|
test "instance policy v3 parses explicit private-network plaintext intent" {
  let text = openai_v3_policy_fixture()
    .replace(old="\"loopback_plaintext\"", new="\"private_network_plaintext\"")
    .replace(
      old="\"listen_host\":\"127.0.0.1\"",
      new="\"listen_host\":\"0.0.0.0\"",
    )
  let policy = try! parse_policy_fixture(text)
  assert_eq(
    policy.external_protocol().transport_security(),
    PrivateNetworkPlaintext,
  )
}

///|
test "instance policy v3 binds complete OpenAI service construction without a secret" {
  let policy = try! parse_policy_fixture(openai_v3_policy_fixture())
  assert_eq(policy.external_protocol().mode(), OpenAiResponsesV1)
  guard policy.openai_service() is Some(openai) else {
    abort("OpenAI v3 service claims were not retained")
  }
  assert_eq(openai.maximum_credential_bytes(), 4096)
  assert_eq(openai.max_head_bytes(), 4096)
  assert_eq(openai.model_alias(), "tiny-bf16")
  assert_eq(openai.response_id_prefix(), "resp_lunaflux_")
  assert_eq(openai.cache_scope_ascii(), "openai-v3")
  assert_eq(openai.sampling_seed(), 7UL)
  assert_eq(openai.deadline_milliseconds(), 60000UL)
}

///|
test "instance policy v3 rejects missing service claims and non-loopback intent" {
  let canonical = openai_v3_policy_fixture()
  let hostile = [
    canonical.replace(
      old="\"openai_service\":{",
      new="\"openai_service_removed\":{",
    ),
    canonical.replace(old="\"loopback_plaintext\"", new="\"deployment_tls\""),
    canonical.replace(old="\"openai_responses_v1\"", new="\"native_framed_v1\""),
    canonical.replace(
      old="\"deadline_milliseconds\":60000",
      new="\"deadline_milliseconds\":60000,\"ambient_secret\":\"forbidden\"",
    ),
  ]
  for index, text in hostile {
    try parse_policy_fixture(text) catch {
      UnknownField(area=Root, ..) | MissingField(area=Root, ..) if index == 0 =>
        ()
      InvalidField(area=ExternalProtocol, ..) if index == 1 || index == 2 => ()
      UnknownField(area=OpenAiService, ..) if index == 3 => ()
      _ => abort("unexpected v3 OpenAI policy refusal")
    } noraise {
      _ => abort("hostile v3 OpenAI policy was admitted")
    }
  }
}

///|
test "instance policy v2 rejects missing malformed and ambiguous endpoints" {
  let canonical = openai_policy_fixture()
  let hostile = [
    canonical.replace(old="\"openai_responses_v1\"", new="\"native_framed_v1\""),
    canonical.replace(old="\"/v1/responses\"", new="\"/v1/chat/completions\""),
    canonical.replace(old="\"/readyz\"", new="\"/healthz\""),
    canonical.replace(old="\"/v1/drain\"", new="\"//drain\""),
    canonical.replace(old="\"POST\"", new="\"DELETE\""),
    canonical.replace(
      old="\"drain_method\":\"POST\"",
      new="\"drain_method\":\"POST\",\"ambient\":true",
    ),
    canonical.replace(
      old="\"external_protocol\":{",
      new="\"external_protocol_removed\":{",
    ),
  ]
  for index, text in hostile {
    try parse_policy_fixture(text) catch {
      InvalidField(area=ExternalProtocol, ..) if index < 5 => ()
      UnknownField(area=ExternalProtocol, ..) if index == 5 => ()
      UnknownField(area=Root, ..) | MissingField(area=Root, ..) if index == 6 =>
        ()
      _ => fail("unexpected external protocol refusal")
    } noraise {
      _ => fail("hostile external protocol was admitted")
    }
  }
}

///|
test "instance policy rejects duplicate unknown and missing claims" {
  let canonical = policy_fixture()
  let hostile = [
    canonical.replace(
      old="\"schema_version\":\"lunaflux.instance-policy.v1\"",
      new="\"schema_version\":\"lunaflux.instance-policy.v1\",\"schema_version\":\"lunaflux.instance-policy.v1\"",
    ),
    canonical.replace(
      old="\"listen_port\":8080",
      new="\"listen_port\":8080,\"ambient\":1",
    ),
    canonical.replace(
      old="\"maximum_attempts\":3",
      new="\"maximum_attempts_removed\":3",
    ),
  ]
  for index, text in hostile {
    try parse_policy_fixture(text) catch {
      DuplicateField(_) if index == 0 => ()
      UnknownField(area=Service, index=_) if index == 1 => ()
      UnknownField(area=Restart, index=_) if index == 2 => ()
      _ => fail("unexpected strict policy schema failure")
    } noraise {
      _ => fail("hostile policy schema was admitted")
    }
  }
}

///|
test "instance policy rejects malformed identities locators and UInt64" {
  let canonical = policy_fixture()
  let hostile = [
    canonical.replace(
      old="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      new="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeF",
    ),
    canonical.replace(old="tokenizer.json", new="../tokenizer.json"),
    canonical.replace(
      old="\"initial_backoff_millis\":1",
      new="\"initial_backoff_millis\":18446744073709551616",
    ),
  ]
  for text in hostile {
    try parse_policy_fixture(text) catch {
      InvalidField(area=_, issue=UnsupportedValue, index=_) => ()
      _ => fail("unexpected strict identity/locator/integer failure")
    } noraise {
      _ => fail("hostile identity, locator, or UInt64 was admitted")
    }
  }
}
