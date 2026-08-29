///|
fn encode_serving_campaign_evidence(
  release : @runtime_instance.RuntimeReleaseAdmission,
  result : @runtime_instance.RuntimeSpawnedServingValidationResult,
) -> String {
  let evidence = "schema=lunaflux-approved-model-serving-readiness.v1\n" +
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
    "\nhealth_before_listener=healthy\n" +
    "readiness_before_listener=not_ready\n" +
    "health_while_serving=healthy\n" +
    "traffic_readiness_observed=1\n" +
    "native_tcp_listener=pass\nrequest_count=1\n" +
    "event_order=accepted,token,token,usage,completed\n" +
    "event_count=" +
    result.event_count().to_string() +
    "\ngenerated_tokens=" +
    result.first_token().to_string() +
    "," +
    result.second_token().to_string() +
    "\nnetwork_accepts=" +
    result.network_accepts().to_string() +
    "\nnetwork_disconnects=" +
    result.network_disconnects().to_string() +
    "\nkv_pages_used_after_request=0\nkv_pages_free_after_request=" +
    result.kv_pages_free().to_string() +
    "\nlistener_closed=1\nchild_closed=1\ncleanup_complete=1\n" +
    "tls_validation=not-run\nperformance_validation=not-run\n" +
    "serving_readiness_validation=pass\n"
  evidence +
  "serving_readiness_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_serving_campaign(
  deployment_argument : String,
  expected_worker_sha256 : String,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  let (release, owner) = prepare_pinned_campaign_owner(
    deployment_argument, expected_worker_sha256,
  )
  let handoff_matches = try owner.require_spawned_physical_handoff() catch {
    _ => false
  } noraise {
    _ =>
      owner.health() == RuntimeHealthy && owner.readiness() == RuntimeNotReady
  }
  if !handoff_matches {
    ignore(drain_campaign_owner(owner) catch { _ => false })
    raise PhysicalHandoffRejected
  }
  let contract = campaign_serving_validation_contract() catch {
    _ => {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise ServingExecutionRejected
    }
  }
  let result = owner.validate_spawned_serving_request(contract) catch {
    @runtime_instance.RuntimeSpawnedServingValidationFailed(rule) => {
      println("serving_validation_failure_rule=" + rule.code())
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise ServingExecutionRejected
    }
    _ => {
      println("serving_validation_failure_rule=unknown")
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise ServingExecutionRejected
    }
  }
  if owner.phase() != Closed ||
    owner.health() != RuntimeClosed ||
    owner.readiness() != RuntimeNotReady ||
    !owner.cleanup_complete() {
    raise CleanupIncomplete
  }
  println(encode_serving_campaign_evidence(release, result))
}
