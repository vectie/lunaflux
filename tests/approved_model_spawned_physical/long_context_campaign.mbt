///|
suberror LongContextCampaignError {
  LongContextPhysicalHandoffRejected
  LongContextValidationRejected
  LongContextReleaseChanged
  LongContextCleanupIncomplete
} derive(Debug, Eq)

///|
fn encode_long_context_evidence(
  release : @runtime_instance.RuntimeReleaseAdmission,
  request_count : Int,
  event_count : Int,
  plan_count : Int,
  input_tokens_processed : Int,
  kv_pages_free : Int,
) -> String {
  let evidence = "schema=lunaflux-approved-model-bf16-actual-long-context-qualification.v1\n" +
    "campaign_scope=qualification-only\n" +
    "release_promotion=not-claimed\n" +
    "required_launch_envelope=max_input_tokens_9\n" +
    "actual_input_token_lengths=8,9\n" +
    "actual_context_tokens_including_output=9,10\n" +
    "physical_page_span=1,2\n" +
    "beyond_9_token_input=not-established\n" +
    "128_or_255_token_execution=not-admitted\n" +
    "full_256_token_context=not-established\n" +
    "selected_logit_correctness=not-observed\n" +
    "gpu_allocator_instrumentation=not-observed-by-this-mode\n" +
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
    "independent_expected_tokens=1355,1240\n" +
    "observed_tokens_per_cycle=1355,1240\n" +
    "cache_permission=disabled\nprefix_reuse=not-exercised\n" +
    "fresh_child_cycles=" +
    LONG_CONTEXT_FRESH_CHILD_CYCLES.to_string() +
    "\nrequest_count=" +
    request_count.to_string() +
    "\nrequests_per_cycle=2\nevents_per_cycle=8\n" +
    "prefill_plans_per_cycle=17\nplan_sequence_range_per_cycle=1-17\n" +
    "event_count=" +
    event_count.to_string() +
    "\nplan_count=" +
    plan_count.to_string() +
    "\ninput_tokens_physically_processed=" +
    input_tokens_processed.to_string() +
    "\nnetwork_observation=structural-no-listener-owner\n" +
    "listener_owner_count=0\nnetwork_accepts=0\n" +
    "network_disconnects=0\nnetwork_rejections=0\n" +
    "kv_pages_used_after_each_request=0\nkv_pages_free_after_each_request=" +
    kv_pages_free.to_string() +
    "\nchildren_spawned=" +
    LONG_CONTEXT_FRESH_CHILD_CYCLES.to_string() +
    "\nchildren_closed=" +
    LONG_CONTEXT_FRESH_CHILD_CYCLES.to_string() +
    "\ncleanup_complete=1\nactual_long_context_qualification=pass\n"
  evidence +
  "actual_long_context_qualification_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_long_context_campaign(
  deployment_argument : String,
  expected_worker_sha256 : String,
) -> Unit {
  let mut pinned_release : @runtime_instance.RuntimeReleaseAdmission? = None
  let mut total_requests = 0
  let mut total_events = 0
  let mut total_plans = 0
  let mut total_input_tokens = 0
  let mut expected_kv_pages_free = -1
  for cycle in 0..<LONG_CONTEXT_FRESH_CHILD_CYCLES {
    let (release, owner) = prepare_pinned_campaign_owner(
      deployment_argument, expected_worker_sha256,
    )
    match pinned_release {
      None => pinned_release = Some(release)
      Some(first) =>
        if first.evidence_sha256() != release.evidence_sha256() ||
          first.worker_executable_sha256() != release.worker_executable_sha256() {
          ignore(drain_campaign_owner(owner) catch { _ => false })
          raise LongContextReleaseChanged
        }
    }
    let handoff_matches = try owner.require_spawned_physical_handoff() catch {
      _ => false
    } noraise {
      _ => owner.readiness() == RuntimeNotReady
    }
    if !handoff_matches {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise LongContextPhysicalHandoffRejected
    }
    let result = owner.validate_spawned_long_context(
      long_context_validation_contract(cycle),
    ) catch {
      @runtime_instance.RuntimeSpawnedLongContextFailed(rule) => {
        println("long_context_failure_rule=" + rule.code())
        ignore(drain_campaign_owner(owner) catch { _ => false })
        raise LongContextValidationRejected
      }
      _ => {
        ignore(drain_campaign_owner(owner) catch { _ => false })
        raise LongContextValidationRejected
      }
    }
    if owner.phase() != Closed ||
      owner.health() != RuntimeClosed ||
      owner.readiness() != RuntimeNotReady ||
      !owner.cleanup_complete() ||
      result.first_token() != PREFIX_REFEREE_FIRST_TOKEN ||
      result.second_token() != PREFIX_REFEREE_SECOND_TOKEN ||
      result.request_count() != 2 ||
      result.event_count() != 8 ||
      result.plan_count() != 17 ||
      result.last_plan_sequence() != 17UL ||
      result.input_tokens_processed() != 17 ||
      result.initial_kv_pages_free() != result.final_kv_pages_free() {
      raise LongContextCleanupIncomplete
    }
    if expected_kv_pages_free < 0 {
      expected_kv_pages_free = result.final_kv_pages_free()
    } else if expected_kv_pages_free != result.final_kv_pages_free() {
      raise LongContextCleanupIncomplete
    }
    total_requests += result.request_count()
    total_events += result.event_count()
    total_plans += result.plan_count()
    total_input_tokens += result.input_tokens_processed()
  }
  guard pinned_release is Some(release) else {
    raise LongContextValidationRejected
  }
  println(
    encode_long_context_evidence(
      release, total_requests, total_events, total_plans, total_input_tokens, expected_kv_pages_free,
    ),
  )
}
