///|
fn phase4_fixture_policy_bytes(
  declaration_sha256 : String,
  hardware? : String = phase4_fixture_digest('1'),
  rich_workload? : String = phase4_fixture_digest('d'),
  lunaflux_revision? : String = phase4_fixture_digest('a'),
  campaign_tool? : String = phase4_fixture_digest('b'),
  gate_executable? : String = phase4_fixture_digest('f'),
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux.phase4-prefix-benchmark-policy.v1\n" +
    "declaration_sha256=" +
    declaration_sha256 +
    "\n" +
    "hardware_sha256=" +
    hardware +
    "\n" +
    "driver_sha256=" +
    phase4_fixture_digest('2') +
    "\n" +
    "toolkit_sha256=" +
    phase4_fixture_digest('3') +
    "\n" +
    "model_sha256=" +
    phase4_fixture_digest('4') +
    "\n" +
    "tokenizer_sha256=" +
    phase4_fixture_digest('5') +
    "\n" +
    "corpus_sha256=" +
    phase4_fixture_digest('6') +
    "\n" +
    "protocol_sha256=" +
    phase4_fixture_digest('7') +
    "\n" +
    "randomization_sha256=" +
    phase4_fixture_digest('8') +
    "\n" +
    "prefix_rich_workload_sha256=" +
    rich_workload +
    "\n" +
    "prefix_cold_workload_sha256=" +
    phase4_fixture_digest('d') +
    "\n" +
    "engine_0=lunaflux\n" +
    "engine_0_revision_sha256=" +
    lunaflux_revision +
    "\n" +
    "engine_0_image_sha256=" +
    phase4_fixture_digest('b') +
    "\n" +
    "engine_0_configuration_sha256=" +
    phase4_fixture_digest('c') +
    "\n" +
    "engine_0_executable_sha256=" +
    phase4_fixture_digest('f') +
    "\n" +
    "engine_1=vllm\n" +
    "engine_1_revision_sha256=" +
    phase4_fixture_digest('a') +
    "\n" +
    "engine_1_image_sha256=" +
    phase4_fixture_digest('b') +
    "\n" +
    "engine_1_configuration_sha256=" +
    phase4_fixture_digest('c') +
    "\n" +
    "engine_1_executable_sha256=" +
    phase4_fixture_digest('f') +
    "\n" +
    "engine_2=sglang\n" +
    "engine_2_revision_sha256=" +
    phase4_fixture_digest('a') +
    "\n" +
    "engine_2_image_sha256=" +
    phase4_fixture_digest('b') +
    "\n" +
    "engine_2_configuration_sha256=" +
    phase4_fixture_digest('c') +
    "\n" +
    "engine_2_executable_sha256=" +
    phase4_fixture_digest('f') +
    "\n" +
    "capture_inventory_sha256=" +
    phase4_fixture_digest('a') +
    "\n" +
    "correctness_inventory_sha256=" +
    phase4_fixture_digest('9') +
    "\n" +
    "campaign_tool_sha256=" +
    campaign_tool +
    "\n" +
    "correctness_verifier_sha256=" +
    phase4_fixture_digest('c') +
    "\n" +
    "engine_identity_verifier_sha256=" +
    phase4_fixture_digest('d') +
    "\n" +
    "process_supervisor_sha256=" +
    phase4_fixture_digest('e') +
    "\n" +
    "phase4_gate_executable_sha256=" +
    gate_executable +
    "\n" +
    "phase4_fd_exec_helper_sha256=" +
    phase4_fixture_digest('1') +
    "\n" +
    "cold_regression_budget_bps=1000\n" +
    "comparison_metric=aggregate_completed_requests_per_nanosecond\n" +
    "physical_measurement_claim=none\n" +
    "authority=phase4-measurement-gate-only\n",
  )
}

///|
fn phase4_parse_fixture_policy(
  bytes : Bytes,
) -> Phase4ExternalPolicy raise Phase4PrefixGateError {
  phase4_parse_policy(bytes, @crypto.bytes_to_hex_string(@crypto.sha256(bytes)))
}

///|
test "Phase 4 external policy joins the declared environment and workloads" {
  let declaration = phase4_fixture_declaration()
  let declaration_sha256 = @crypto.bytes_to_hex_string(
    @crypto.sha256(declaration),
  )
  let policy = phase4_parse_fixture_policy(
    phase4_fixture_policy_bytes(declaration_sha256),
  )
  phase4_join_declaration(policy, declaration)
  assert_eq(policy.declaration, declaration_sha256)
  assert_eq(policy.cold_regression_budget_bps, 1000)
}

///|
test "Phase 4 external policy rejects environment workload and engine substitutions" {
  let declaration = phase4_fixture_declaration()
  let declaration_sha256 = @crypto.bytes_to_hex_string(
    @crypto.sha256(declaration),
  )
  let substitutions = [
    phase4_fixture_policy_bytes(
      declaration_sha256,
      hardware=phase4_fixture_digest('9'),
    ),
    phase4_fixture_policy_bytes(
      declaration_sha256,
      rich_workload=phase4_fixture_digest('9'),
    ),
    phase4_fixture_policy_bytes(
      declaration_sha256,
      lunaflux_revision=phase4_fixture_digest('9'),
    ),
  ]
  for bytes in substitutions {
    let policy = phase4_parse_fixture_policy(bytes)
    try phase4_join_declaration(policy, declaration) catch {
      Phase4InvalidPolicy => ()
      _ => fail("unexpected Phase 4 declaration join rejection")
    } noraise {
      _ => fail("external Phase 4 declaration substitution was admitted")
    }
  }
}

///|
test "Phase 4 policy digest and authority fail closed" {
  let declaration = phase4_fixture_declaration()
  let declaration_sha256 = @crypto.bytes_to_hex_string(
    @crypto.sha256(declaration),
  )
  let bytes = phase4_fixture_policy_bytes(declaration_sha256)
  try phase4_parse_policy(bytes, phase4_fixture_digest('9')) catch {
    Phase4InvalidPolicy => ()
    _ => fail("unexpected Phase 4 policy digest rejection")
  } noraise {
    _ => fail("substituted Phase 4 policy digest was admitted")
  }
  let invalid = @utf8.encode(
    (try! @utf8.decode(bytes)).replace_all(
      old="authority=phase4-measurement-gate-only",
      new="authority=release",
    ),
  )
  try phase4_parse_fixture_policy(invalid) catch {
    Phase4InvalidPolicy => ()
    _ => fail("unexpected Phase 4 authority rejection")
  } noraise {
    _ => fail("Phase 4 release authority was admitted")
  }
}
