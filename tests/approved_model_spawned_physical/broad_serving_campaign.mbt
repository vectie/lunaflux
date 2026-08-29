///|
fn encode_broad_serving_campaign_evidence(
  release : @runtime_instance.RuntimeReleaseAdmission,
  result : @runtime_instance.RuntimeSpawnedBroadServingValidationResult,
  restart : @runtime_instance.RuntimeSpawnedServingValidationResult,
) -> String {
  let evidence = "schema=lunaflux-approved-model-broad-bf16-serving-qualification.v1\n" +
    "campaign_scope=qualification-only\n" +
    "production_readiness=not-claimed\n" +
    "performance_baseline=not-established\n" +
    "benchmark_claim=not-made\n" +
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
    "physical_executor=spawned-current-source\n" +
    "mixed_concurrent_requests=pass\n" +
    "concurrent_request_count=2\n" +
    "saturation_boundary=active-plus-waiting\n" +
    "saturation_backpressure=pass\n" +
    "backpressure_count=" +
    result.backpressure_count().to_string() +
    "\noverload_rejection=not-exercised-single-connection-owner\n" +
    "cancellation_isolation=pass\n" +
    "cancellation_count=" +
    result.cancellation_count().to_string() +
    "\ntyped_foreign_model_rejection=pass\n" +
    "malformed_native_frame=fail-closed-connection-isolation-pass\n" +
    "malformed_http=not-exercised-native-framed-owner\n" +
    "post_malformed_same_owner_recovery=pass\n" +
    "pre_malformed_reconnect_after_rejection=pass\n" +
    "fresh_owner_restart=pass\n" +
    "drain_transition=pass\n" +
    "request_count=" +
    result.request_count().to_string() +
    "\ncompletion_count=" +
    result.completion_count().to_string() +
    "\nevent_count=" +
    result.event_count().to_string() +
    "\nnetwork_accepts=" +
    result.network_accepts().to_string() +
    "\nnetwork_disconnects=" +
    result.network_disconnects().to_string() +
    "\nnetwork_rejections=" +
    result.network_rejections().to_string() +
    "\nkv_pages_used_after_cases=0\nkv_pages_free_after_cases=" +
    result.kv_pages_free().to_string() +
    "\nrestart_request_event_count=" +
    restart.event_count().to_string() +
    "\nrestart_network_accepts=" +
    restart.network_accepts().to_string() +
    "\nrestart_network_disconnects=" +
    restart.network_disconnects().to_string() +
    "\nrestart_kv_pages_free=" +
    restart.kv_pages_free().to_string() +
    "\nlistener_closed=2\nchild_closed=2\ncleanup_complete=1\n" +
    "broad_bf16_serving_qualification=pass\n"
  evidence +
  "broad_bf16_serving_qualification_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_broad_serving_campaign(
  deployment_argument : String,
  expected_worker_sha256 : String,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  let (release, owner) = prepare_pinned_campaign_owner(
    deployment_argument, expected_worker_sha256,
  )
  let contract = broad_campaign_validation_contract() catch {
    _ => {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise BroadServingExecutionRejected
    }
  }
  let result = owner.validate_spawned_broad_serving(contract) catch {
    @runtime_instance.RuntimeSpawnedBroadServingValidationFailed(rule) => {
      println("broad_serving_validation_failure_rule=" + rule.code())
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise BroadServingExecutionRejected
    }
    _ => {
      println("broad_serving_validation_failure_rule=unknown")
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise BroadServingExecutionRejected
    }
  }
  if owner.phase() != Closed ||
    owner.health() != RuntimeClosed ||
    owner.readiness() != RuntimeNotReady ||
    !owner.cleanup_complete() {
    raise CleanupIncomplete
  }

  // Broad validation proves malformed-connection isolation and same-owner
  // reuse, then drains the first owner. A second authenticated child must bind
  // the same configured listener and complete the canonical request before
  // fresh-process restart is claimed.
  let (restart_release, restart_owner) = prepare_pinned_campaign_owner(
    deployment_argument, expected_worker_sha256,
  )
  if restart_release.evidence_sha256() != release.evidence_sha256() ||
    restart_release.worker_executable_sha256() !=
    release.worker_executable_sha256() {
    ignore(drain_campaign_owner(restart_owner) catch { _ => false })
    raise PinnedReleaseMismatch
  }
  let restart_contract = campaign_serving_validation_contract() catch {
    _ => {
      ignore(drain_campaign_owner(restart_owner) catch { _ => false })
      raise BroadServingExecutionRejected
    }
  }
  let restart = restart_owner.validate_spawned_serving_request(restart_contract) catch {
    _ => {
      ignore(drain_campaign_owner(restart_owner) catch { _ => false })
      raise BroadServingExecutionRejected
    }
  }
  if restart_owner.phase() != Closed ||
    restart_owner.health() != RuntimeClosed ||
    restart_owner.readiness() != RuntimeNotReady ||
    !restart_owner.cleanup_complete() {
    raise CleanupIncomplete
  }
  println(encode_broad_serving_campaign_evidence(release, result, restart))
}
