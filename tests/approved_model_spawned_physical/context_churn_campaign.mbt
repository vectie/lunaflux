///|
suberror ContextChurnCampaignError {
  ContextChurnPhysicalHandoffRejected
  ContextChurnValidationRejected
  ContextChurnReleaseChanged
  ContextChurnCleanupIncomplete
} derive(Debug, Eq)

///|
fn encode_context_churn_evidence(
  release : @runtime_instance.RuntimeReleaseAdmission,
  request_count : Int,
  event_count : Int,
  plan_count : Int,
  kv_pages_free : Int,
) -> String {
  let evidence = "schema=lunaflux-approved-model-bf16-context-churn-qualification.v1\n" +
    "campaign_scope=qualification-only\n" +
    "release_promotion=not-claimed\n" +
    "broader_context_length_coverage=not-established\n" +
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
    "\nmodel_dtype=bf16\n" +
    "input_tokens_per_request=1\n" +
    "output_tokens_per_request=2\n" +
    "actual_context_tokens_per_request=3\n" +
    "independent_expected_tokens=1031,2185\n" +
    "declared_context_ceiling_matrix=8,32,256\n" +
    "fresh_child_cycles=" +
    CONTEXT_CHURN_FRESH_CHILD_CYCLES.to_string() +
    "\nrequests_per_child=" +
    CONTEXT_CHURN_REQUESTS_PER_CHILD.to_string() +
    "\nrequest_count=" +
    request_count.to_string() +
    "\nevent_count=" +
    event_count.to_string() +
    "\nplan_count=" +
    plan_count.to_string() +
    "\nnetwork_observation=structural-no-listener-owner\n" +
    "listener_owner_count=0\nnetwork_accepts=0\n" +
    "network_disconnects=0\nnetwork_rejections=0\n" +
    "kv_pages_used_after_each_request=0\nkv_pages_free_after_each_request=" +
    kv_pages_free.to_string() +
    "\nchildren_spawned=" +
    CONTEXT_CHURN_FRESH_CHILD_CYCLES.to_string() +
    "\nchildren_closed=" +
    CONTEXT_CHURN_FRESH_CHILD_CYCLES.to_string() +
    "\ncleanup_complete=1\ncontext_churn_qualification=pass\n"
  evidence +
  "context_churn_qualification_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_context_churn_campaign(
  deployment_argument : String,
  expected_worker_sha256 : String,
) -> Unit {
  let mut pinned_release : @runtime_instance.RuntimeReleaseAdmission? = None
  let mut total_requests = 0
  let mut total_events = 0
  let mut total_plans = 0
  let mut expected_kv_pages_free = -1
  for cycle in 0..<CONTEXT_CHURN_FRESH_CHILD_CYCLES {
    let (release, owner) = prepare_pinned_campaign_owner(
      deployment_argument, expected_worker_sha256,
    )
    match pinned_release {
      None => pinned_release = Some(release)
      Some(first) =>
        if first.evidence_sha256() != release.evidence_sha256() ||
          first.worker_executable_sha256() != release.worker_executable_sha256() {
          ignore(drain_campaign_owner(owner) catch { _ => false })
          raise ContextChurnReleaseChanged
        }
    }
    let handoff_matches = try owner.require_spawned_physical_handoff() catch {
      _ => false
    } noraise {
      _ => owner.readiness() == RuntimeNotReady
    }
    if !handoff_matches {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise ContextChurnPhysicalHandoffRejected
    }
    let result = owner.validate_spawned_context_churn(
      context_churn_validation_contract(cycle),
    ) catch {
      @runtime_instance.RuntimeSpawnedContextChurnFailed(rule) => {
        println("context_churn_failure_rule=" + rule.code())
        ignore(drain_campaign_owner(owner) catch { _ => false })
        raise ContextChurnValidationRejected
      }
      _ => {
        ignore(drain_campaign_owner(owner) catch { _ => false })
        raise ContextChurnValidationRejected
      }
    }
    if owner.phase() != Closed ||
      owner.health() != RuntimeClosed ||
      owner.readiness() != RuntimeNotReady ||
      !owner.cleanup_complete() ||
      result.request_count() != CONTEXT_CHURN_REQUESTS_PER_CHILD ||
      result.event_count() != CONTEXT_CHURN_REQUESTS_PER_CHILD * 5 ||
      result.plan_count() != CONTEXT_CHURN_REQUESTS_PER_CHILD * 2 ||
      result.last_plan_sequence() !=
      (CONTEXT_CHURN_REQUESTS_PER_CHILD * 2).to_uint64() ||
      result.initial_kv_pages_free() != result.final_kv_pages_free() ||
      result.network_accepts() != 0UL ||
      result.network_disconnects() != 0UL ||
      result.network_rejections() != 0UL {
      raise ContextChurnCleanupIncomplete
    }
    if expected_kv_pages_free < 0 {
      expected_kv_pages_free = result.final_kv_pages_free()
    } else if expected_kv_pages_free != result.final_kv_pages_free() {
      raise ContextChurnCleanupIncomplete
    }
    total_requests += result.request_count()
    total_events += result.event_count()
    total_plans += result.plan_count()
  }
  guard pinned_release is Some(release) else {
    raise ContextChurnValidationRejected
  }
  println(
    encode_context_churn_evidence(
      release, total_requests, total_events, total_plans, expected_kv_pages_free,
    ),
  )
}
