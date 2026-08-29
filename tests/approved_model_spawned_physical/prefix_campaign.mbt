///|
fn encode_prefix_campaign_evidence(
  release : @runtime_instance.RuntimeReleaseAdmission,
  result : @runtime_instance.RuntimeSpawnedPrefixValidationResult,
) -> String {
  let evidence = "schema=lunaflux-approved-model-prefix-reuse.v1\n" +
    "upstream_revision=" +
    PINNED_UPSTREAM_REVISION +
    "\nruntime_recipe=" +
    release.runtime_recipe() +
    "\nmodel_content_sha256=" +
    release.model_content_sha256() +
    "\nmodel_plan_sha256=" +
    release.model_plan_sha256() +
    "\ntokenizer_sha256=" +
    release.tokenizer_sha256() +
    "\nworker_executable_sha256=" +
    release.worker_executable_sha256() +
    "\nrelease_preflight_sha256=" +
    release.evidence_sha256() +
    "\nreferee_schema=" +
    PREFIX_REFEREE_SCHEMA +
    "\nreferee_sha256=" +
    PREFIX_REFEREE_SHA256 +
    "\nreferee_path=approved_scalar_bf16_reference_executor\n" +
    "first_request_id=" +
    PREFIX_CAMPAIGN_FIRST_REQUEST_ID.to_string() +
    "\nsecond_request_id=" +
    PREFIX_CAMPAIGN_SECOND_REQUEST_ID.to_string() +
    "\ncache_scope=" +
    PREFIX_CAMPAIGN_SCOPE +
    "\nfirst_cache_permission=read_write\n" +
    "second_cache_permission=read_only\n" +
    "first_input_tokens=1,229,153,132,75,104,111,111\n" +
    "second_input_tokens=1,229,153,132,75,104,111,111,114\n" +
    "independent_expected_tokens=" +
    PREFIX_REFEREE_FIRST_TOKEN.to_string() +
    "," +
    PREFIX_REFEREE_SECOND_TOKEN.to_string() +
    "\nobserved_tokens=" +
    result.first_token().to_string() +
    "," +
    result.second_token().to_string() +
    "\nfirst_cached_input_tokens=0\nsecond_cached_input_tokens=8\n" +
    "prefix_lookups=" +
    result.prefix_lookups().to_string() +
    "\nprefix_hits=" +
    result.prefix_hits().to_string() +
    "\nprefix_misses=" +
    result.prefix_misses().to_string() +
    "\nprefix_publications=" +
    result.prefix_publications().to_string() +
    "\nprefix_tokens_reused=" +
    result.reused_tokens().to_string() +
    "\nretained_prefix_pages_before_close=" +
    result.retained_prefix_pages().to_string() +
    "\nkv_pages_free_initial=" +
    result.initial_kv_pages_free().to_string() +
    "\nkv_pages_free_before_close=" +
    result.kv_pages_free().to_string() +
    "\nfirst_terminal_plan_sequence=" +
    result.first_terminal_plan_sequence().to_string() +
    "\nsecond_terminal_plan_sequence=" +
    result.second_terminal_plan_sequence().to_string() +
    "\nowner_closed=1\nchild_closed=1\ncleanup_complete=1\n" +
    "prefix_reuse_qualification=pass\n"
  evidence +
  "prefix_reuse_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_prefix_campaign(
  deployment_argument : String,
  expected_worker_sha256 : String,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  let (release, owner) = prepare_pinned_campaign_owner(
    deployment_argument, expected_worker_sha256,
  )
  let handoff_matches = try owner.require_spawned_physical_handoff() catch {
    _ => false
  } noraise {
    _ => owner.readiness() == RuntimeNotReady
  }
  if !handoff_matches {
    ignore(drain_campaign_owner(owner) catch { _ => false })
    raise PhysicalHandoffRejected
  }
  let contract = prefix_campaign_validation_contract() catch {
    _ => {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise SpawnedExecutionRejected
    }
  }
  let result = owner.validate_spawned_prefix_reuse(contract) catch {
    RuntimeSpawnedPrefixValidationFailed(rule) => {
      println("prefix_reuse_failure_rule=" + rule.code())
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise SpawnedExecutionRejected
    }
  }
  if owner.phase() != Closed ||
    owner.health() != RuntimeClosed ||
    owner.readiness() != RuntimeNotReady ||
    !owner.cleanup_complete() {
    raise CleanupIncomplete
  }
  println(encode_prefix_campaign_evidence(release, result))
}
