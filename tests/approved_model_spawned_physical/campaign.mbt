///|
suberror ApprovedModelSpawnedPhysicalError {
  UsageRejected
  ExpectedWorkerDigestRejected
  PinnedModelIdentityRejected
  PinnedReleaseMismatch
  ReleasePreflightRejected
  SpawnedPreparationRejected
  PhysicalHandoffRejected
  SpawnedExecutionRejected
  ServingExecutionRejected
  BroadServingExecutionRejected
  OpenAiQualificationRejected
  BenchmarkExecutionRejected
  CleanupIncomplete
  DeviceGreedyQualificationRejected
} derive(Debug, Eq)

///|
fn campaign_file_limits() -> @runtime_instance.RuntimeInstanceFileLimits raise {
  @runtime_instance.RuntimeInstanceFileLimits::new(
    launch=@launch_file.LaunchFileLimits::new(
      maximum_bytes=1048576L,
      maximum_json_depth=16,
    ),
    descriptor=@descriptor_file.RuntimeDescriptorFileLimits::new(
      max_descriptor_bytes=1048576L,
      max_json_depth=16,
    ),
    policy=@instance_policy_file.InstancePolicyFileLimits::new(
      max_policy_bytes=1048576L,
      max_json_depth=16,
    ),
    executable=@worker_executable_file.WorkerExecutableFileLimits::new(
      maximum_bytes=1073741824L,
    ),
  )
}

///|
fn prepare_pinned_campaign_owner(
  deployment_argument : String,
  expected_worker_sha256 : String,
) -> (
  @runtime_instance.RuntimeReleaseAdmission,
  @runtime_instance.RuntimeInstanceOwner,
) raise ApprovedModelSpawnedPhysicalError {
  let expected_worker = @worker_executable_file.WorkerExecutableDigest::from_sha256(
    expected_worker_sha256,
  ) catch {
    _ => raise ExpectedWorkerDigestRejected
  }
  let limits = campaign_file_limits() catch {
    _ => raise ReleasePreflightRejected
  }
  let release = @runtime_instance.preflight_release(deployment_argument, limits) catch {
    _ => raise ReleasePreflightRejected
  }
  require_pinned_release(release, expected_worker.as_hex())
  let owner = @runtime_instance.prepare(deployment_argument, limits) catch {
    RuntimeInstanceFailed(stage) => {
      println("spawned_preparation_failure_stage=" + stage.code())
      raise SpawnedPreparationRejected
    }
    RuntimeInstanceFailedAndCleanup(stage, cleanup) => {
      println("spawned_preparation_failure_stage=" + stage.code())
      println("spawned_preparation_cleanup_failure_stage=" + cleanup.code())
      raise SpawnedPreparationRejected
    }
  }
  (release, owner)
}

///|
fn campaign_drain_action(
  phase : @runtime_instance.RuntimeInstancePhase,
  cleanup_complete : Bool,
) -> Int {
  if cleanup_complete {
    return 0
  }
  match phase {
    Starting | HealthyNotReady | Ready | OpenAiQualificationReady => 1
    Draining | Closed | Failed => 2
  }
}

///|
async fn drain_campaign_owner(
  owner : @runtime_instance.RuntimeInstanceOwner,
) -> Bool {
  @async.protect_from_cancel(() => {
    for step = 0; step < 1048576; step = step + 1 {
      match campaign_drain_action(owner.phase(), owner.cleanup_complete()) {
        0 => return true
        1 => owner.begin_drain() catch { _ => () }
        _ => ()
      }
      ignore(owner.progress())
    }
    owner.cleanup_complete()
  })
}

///|
fn encode_campaign_evidence(
  release : @runtime_instance.RuntimeReleaseAdmission,
  result : @runtime_instance.RuntimeSpawnedValidationResult,
) -> String {
  let evidence = "schema=lunaflux-approved-model-spawned-execution.v2\n" +
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
    "\nspawned_child_executor_prepared=1\n" +
    "traffic_readiness=0\nplan_executed=2\nprefill_plan_executed=1\n" +
    "same_page_decode_plan_executed=1\ngenerated_tokens=" +
    result.first_token().to_string() +
    "," +
    result.second_token().to_string() +
    "\nprefill_plan_sequence=" +
    result.prefill_plan_sequence().to_string() +
    "\ndecode_plan_sequence=" +
    result.decode_plan_sequence().to_string() +
    "\nkv_pages_used_during_steps=" +
    result.kv_page_count().to_string() +
    "\ncontext_capacity_tokens=" +
    CAMPAIGN_CONTEXT_TOKENS.to_string() +
    "\nemergency_decode_page_reserve=positive-admitted" +
    "\ntoken_numerical_correctness=pinned-corpus-pass\n" +
    "selected_logit_correctness=not-observed\ncleanup_complete=1\n" +
    "spawned_execution=pass\n"
  evidence +
  "spawned_execution_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_campaign(
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
  let result = owner.validate_spawned_physical_request(
    campaign_validation_contract() catch {
      _ => {
        ignore(drain_campaign_owner(owner) catch { _ => false })
        raise SpawnedExecutionRejected
      }
    },
  ) catch {
    @runtime_instance.RuntimeSpawnedValidationFailed(rule) => {
      println("spawned_execution_failure_rule=" + rule.code())
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise SpawnedExecutionRejected
    }
    _ => {
      println("spawned_execution_failure_rule=unknown")
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise SpawnedExecutionRejected
    }
  }
  let drained = drain_campaign_owner(owner) catch { _ => false }
  if !drained || !owner.cleanup_complete() {
    raise CleanupIncomplete
  }
  let evidence = encode_campaign_evidence(release, result)
  println(evidence)
}
