///|
fn qualification_campaign_bytes(text : String) -> FixedArray[Byte] {
  let encoded = @utf8.encode(text)
  FixedArray::makei(encoded.length(), index => encoded[index])
}

///|
fn qualification_campaign_template(
  inference : @inference.InferenceLimits,
) -> @openai_compat.LunaOpenAIChatTemplate raise {
  let empty : FixedArray[Byte] = []
  @openai_compat.LunaOpenAIChatTemplate::new(
    inference,
    empty,
    empty,
    empty,
    empty,
    empty,
    empty,
    empty,
    max_rendered_prompt_bytes=4,
  )
}

///|
fn qualification_campaign_policy(
  credential : FixedArray[Byte],
) -> @runtime_instance.RuntimeOpenAiQualificationPolicy raise {
  let inference = campaign_inference_limits()
  let model_alias = qualification_campaign_bytes("tiny-bf16")
  let response_prefix = qualification_campaign_bytes("resp_physical_")
  @runtime_instance.RuntimeOpenAiQualificationPolicy::new(
    @http1.LunaHttp1Limits::new(max_head_bytes=2048, max_body_bytes=2048),
    @http1.LunaHttp1StepBudget::new(17),
    @http1.LunaHttp1ResponseStepBudget::new(31),
    @api_auth.LunaApiAuthPolicy::new(
      credential,
      source_offset=0,
      length=credential.length(),
      maximum_credential_bytes=128,
    ),
    @openai_compat.LunaOpenAICompatStepBudget::new(17),
    @openai_compat.LunaOpenAICompatStepBudget::new(31),
    @openai_compat.LunaOpenAIInboundLimits::new(max_messages=2),
    qualification_campaign_template(inference),
    model_alias,
    response_prefix,
    @inference.CacheScope::from_ascii(b"openai-qualification", inference),
    2,
    CAMPAIGN_CONTEXT_TOKENS,
    @inference.SamplingParameters::greedy(),
    @inference.SamplingSeed::new(7UL),
    @inference.DeadlineBudget::from_milliseconds(60000UL, inference),
  )
}

///|
fn encode_openai_qualification_evidence(
  release : @runtime_instance.RuntimeReleaseAdmission,
  result : @runtime_instance.RuntimeOpenAiQualificationValidationResult,
) -> String {
  let evidence = "schema=lunaflux-openai-responses-loopback-qualification.v1\n" +
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
    "\nqualification_provenance=non_routable\n" +
    "production_readiness=not_ready\n" +
    "transport=loopback_plaintext\n" +
    "bearer_auth_missing=401\n" +
    "bearer_auth_wrong=401\n" +
    "responses_status=200\n" +
    "event_order=response.created,response.output_text.delta,response.usage,response.completed,done\n" +
    "event_count=" +
    result.event_count().to_string() +
    "\nrequest_count=" +
    result.admissions().to_string() +
    "\ncompletion_count=" +
    result.completions().to_string() +
    "\nnetwork_accepts=" +
    result.network_accepts().to_string() +
    "\nnetwork_disconnects=" +
    result.network_disconnects().to_string() +
    "\nnetwork_rejections=" +
    result.network_rejections().to_string() +
    "\nkv_pages_used_after_request=0\nkv_pages_free_after_request=" +
    result.kv_pages_free().to_string() +
    "\ndrain_transition=pass\npost_drain_admission=refused\n" +
    "listener_closed=1\nchild_closed=1\ncleanup_complete=1\n" +
    "health_endpoint=loopback-200-exercised\n" +
    "readiness_endpoint=loopback-503-qualification\n" +
    "tls_validation=not-run\n" +
    "openai_responses_qualification=pass\n"
  evidence +
  "openai_responses_qualification_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_openai_qualification_campaign(
  deployment_argument : String,
  expected_worker_sha256 : String,
  credential_text : String,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  let credential = qualification_campaign_bytes(credential_text)
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
  let policy = qualification_campaign_policy(credential) catch {
    _ => {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise OpenAiQualificationRejected
    }
  }
  let contract = @runtime_instance.RuntimeOpenAiQualificationValidationContract::new(
    credential, 1048576, 5000, 65536,
  ) catch {
    _ => {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise OpenAiQualificationRejected
    }
  }
  let result = owner.validate_openai_qualification(policy, contract) catch {
    @runtime_instance.RuntimeOpenAiQualificationValidationFailed(rule) => {
      println("openai_qualification_failure_rule=" + rule.code())
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise OpenAiQualificationRejected
    }
    _ => {
      println("openai_qualification_failure_rule=unknown")
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise OpenAiQualificationRejected
    }
  }
  if owner.phase() != Closed ||
    owner.health() != RuntimeClosed ||
    owner.readiness() != RuntimeNotReady ||
    owner.openai_qualification_readiness() != OpenAiQualificationNotReady ||
    !owner.cleanup_complete() {
    raise CleanupIncomplete
  }
  println(encode_openai_qualification_evidence(release, result))
}
