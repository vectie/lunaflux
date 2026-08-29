///|
fn device_greedy_graph_path(captured : Bool) -> String {
  if captured {
    "captured"
  } else {
    "ordered-eager"
  }
}

///|
async fn run_device_greedy_route(
  deployment_argument : String,
  expected_worker_sha256 : String,
  sampling_runtime : @worker_wire.WorkerBootstrapSamplingRuntime,
  require_fused : Bool,
) -> (
  @runtime_instance.RuntimeReleaseAdmission,
  @runtime_instance.RuntimeSpawnedValidationResult,
) raise ApprovedModelSpawnedPhysicalError {
  let (release, owner) = prepare_pinned_campaign_owner(
    deployment_argument, expected_worker_sha256,
  )
  let handoff = try owner.require_spawned_physical_handoff() catch {
    _ => false
  } noraise {
    _ => owner.readiness() == RuntimeNotReady
  }
  if !handoff {
    ignore(drain_campaign_owner(owner) catch { _ => false })
    raise DeviceGreedyQualificationRejected
  }
  let expectation = match sampling_runtime {
    EmbeddedCudaGreedyV1 => @runtime_instance.EmbeddedCudaGreedyExpected
    HostSamplingRuntime => HostSamplingRuntimeExpected
  }
  let contract = campaign_validation_contract().with_execution_expectation(
    expectation, require_fused,
  ) catch {
    _ => {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise DeviceGreedyQualificationRejected
    }
  }
  let result = owner.validate_spawned_physical_request(contract) catch {
    _ => {
      ignore(drain_campaign_owner(owner) catch { _ => false })
      raise DeviceGreedyQualificationRejected
    }
  }
  let drained = drain_campaign_owner(owner) catch { _ => false }
  if !drained || !owner.cleanup_complete() {
    raise CleanupIncomplete
  }
  (release, result)
}

///|
fn encode_device_greedy_evidence(
  embedded_release : @runtime_instance.RuntimeReleaseAdmission,
  embedded : @runtime_instance.RuntimeSpawnedValidationResult,
  host_release : @runtime_instance.RuntimeReleaseAdmission,
  host : @runtime_instance.RuntimeSpawnedValidationResult,
  fused_required : Bool,
) -> String raise ApprovedModelSpawnedPhysicalError {
  if embedded.first_token() != host.first_token() ||
    embedded.second_token() != host.second_token() ||
    embedded.first_token() != 1031 ||
    embedded.second_token() != 2185 {
    raise DeviceGreedyQualificationRejected
  }
  let fused = if fused_required {
    "required-present"
  } else {
    "optional-absent"
  }
  let evidence = "schema=lunaflux-spawned-device-greedy-qualification.v1\n" +
    "embedded_release_sha256=" +
    embedded_release.evidence_sha256() +
    "\nhost_referee_release_sha256=" +
    host_release.evidence_sha256() +
    "\nworker_executable_sha256=" +
    embedded_release.worker_executable_sha256() +
    "\nmodel_content_sha256=" +
    embedded_release.model_content_sha256() +
    "\nmodel_plan_sha256=" +
    embedded_release.model_plan_sha256() +
    "\nspawn_boundary=descriptor_file,worker_wire,child_bootstrap,paged_device_executor\n" +
    "embedded_sampling_runtime=embedded_cuda_greedy_v1\n" +
    "host_referee_sampling_runtime=host_sampling\n" +
    "fused_v2_runtime=" +
    fused +
    "\nrequest_plans=2\nsampling_result_row_bytes=8\n" +
    "sampling_result_layout=token_i32,status_i32\n" +
    "sampling_readback_bytes_per_plan=8\n" +
    "sampling_readback_total_bytes=16\n" +
    "sampling_success_status=-1\n" +
    "nonfinite_status=first_nonfinite_token_id\n" +
    "tie_policy=lowest_token_id\n" +
    "generated_tokens=" +
    embedded.first_token().to_string() +
    "," +
    embedded.second_token().to_string() +
    "\nhost_oracle_tokens=" +
    host.first_token().to_string() +
    "," +
    host.second_token().to_string() +
    "\nselected_logit_correctness=pass\n" +
    "embedded_graph_path=" +
    device_greedy_graph_path(embedded.graph_captured()) +
    "\nembedded_prefill_graph_hits=" +
    embedded.prefill_graph_hits().to_string() +
    "\nembedded_prefill_graph_misses=" +
    embedded.prefill_graph_misses().to_string() +
    "\nembedded_decode_graph_hits=" +
    embedded.decode_graph_hits().to_string() +
    "\nembedded_decode_graph_misses=" +
    embedded.decode_graph_misses().to_string() +
    "\nhost_graph_path=" +
    device_greedy_graph_path(host.graph_captured()) +
    "\nhost_prefill_graph_hits=" +
    host.prefill_graph_hits().to_string() +
    "\nhost_prefill_graph_misses=" +
    host.prefill_graph_misses().to_string() +
    "\nhost_decode_graph_hits=" +
    host.decode_graph_hits().to_string() +
    "\nhost_decode_graph_misses=" +
    host.decode_graph_misses().to_string() +
    "\ngraph_policy_interaction=authenticated-and-stable\n" +
    "child_closed=2\ncleanup_complete=1\nphysical_cuda_observed=true\n" +
    "qualification_only=true\nmanifest_bindable=false\npromotion_authority=absent\n"
  evidence +
  "device_greedy_qualification_sha256=" +
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))) +
  "\n"
}

///|
async fn run_device_greedy_campaign(
  embedded_deployment : String,
  host_deployment : String,
  expected_worker_sha256 : String,
  fused_required : Bool,
) -> Unit raise ApprovedModelSpawnedPhysicalError {
  let (embedded_release, embedded) = run_device_greedy_route(
    embedded_deployment,
    expected_worker_sha256,
    EmbeddedCudaGreedyV1,
    fused_required,
  )
  let (host_release, host) = run_device_greedy_route(
    host_deployment,
    expected_worker_sha256,
    HostSamplingRuntime,
    false,
  )
  if embedded_release.worker_executable_sha256() !=
    host_release.worker_executable_sha256() ||
    embedded_release.model_content_sha256() !=
    host_release.model_content_sha256() ||
    embedded_release.model_plan_sha256() != host_release.model_plan_sha256() {
    raise DeviceGreedyQualificationRejected
  }
  println(
    encode_device_greedy_evidence(
      embedded_release, embedded, host_release, host, fused_required,
    ),
  )
}
