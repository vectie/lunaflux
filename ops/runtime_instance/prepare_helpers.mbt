///|
fn fail(stage : RuntimeInstanceStage) -> RuntimeInstanceError {
  RuntimeInstanceFailed(stage)
}

///|
fn close_root(
  root : @approved_fs.ApprovedRoot,
  primary : RuntimeInstanceStage,
) -> Unit raise RuntimeInstanceError {
  root.close() catch {
    _ => raise fail(primary)
  }
}

///|
fn close_model_kernel(
  model : @approved_fs.ApprovedRoot,
  kernel : @approved_fs.ApprovedRoot,
  primary? : RuntimeInstanceStage = Cleanup,
) -> Unit raise RuntimeInstanceError {
  let failed = Ref(false)
  kernel.close() catch {
    _ => failed.val = true
  }
  model.close() catch {
    _ => failed.val = true
  }
  if failed.val {
    raise fail(primary)
  }
}

///|
fn retain_borrowed_root_after_close(
  close_failed : Bool,
  root_closed : Bool,
) -> Bool {
  close_failed && !root_closed
}

///|
/// Captures either lower ready authority or its exact cleanup authority, then
/// closes the two borrowed roots. Both authenticated runtime recipes converge
/// here before any listener or readiness publication exists.
fn capture_prepared_owner(
  service : @online_session.LunaOnlineFramedServicePreparation,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  envelope : RuntimeServiceEnvelope,
  release_admission : RuntimeReleaseAdmission,
  startup_binding : RuntimeOpaqueStartupBinding,
  opaque_cli_requires_drain : Bool,
  clock : @monotonic_clock.MonotonicClock,
  started_at_millis : UInt64,
) -> RuntimeInstanceOwner {
  let owner : RuntimeInstanceOwner = {
    phase: HealthyNotReady,
    preparations: Array::new(capacity=1),
    cleanup: Array::new(capacity=1),
    services: Array::new(capacity=1),
    servers: Array::new(capacity=1),
    framed_pools: Array::new(capacity=1),
    openai_servers: Array::new(capacity=1),
    openai_pools: Array::new(capacity=1),
    controls: Array::new(capacity=1),
    inherited_drains: Array::new(capacity=1),
    inherited_drain_required: false,
    opaque_cli_requires_drain,
    release_admission,
    startup_binding,
    borrowed_roots: Array::new(capacity=2),
    addr: envelope.addr,
    transport: envelope.transport,
    telemetry_capacity: envelope.telemetry_capacity,
    connection_capacity: envelope.connection_capacity,
    request_work_units: envelope.request_work_units,
    event_work_units: envelope.event_work_units,
    receipt_deadline: envelope.receipt_deadline,
    loopback_openai_allowed: envelope.loopback_openai_allowed,
    openai_listener_allowed: envelope.openai_listener_allowed,
    clock,
    started_at_millis,
    cold_start_recorded: false,
    cold_start_latency_millis: -1,
    control_deferred_transitions: 0,
  }
  match service.state() {
    LunaOnlineInstanceReady => owner.preparations.push(service)
    LunaOnlineInstanceCleanupRequired => {
      let cleanup = try service.take_cleanup() catch {
        _ => {
          // Retain the typed preparation so Failed progress can retry the
          // exact cleanup outcome without abandoning its lower authority.
          owner.preparations.push(service)
          owner.phase = Failed
          None
        }
      } noraise {
        cleanup => Some(cleanup)
      }
      if cleanup is Some(failed) {
        owner.cleanup.push(failed)
      }
      owner.phase = Failed
    }
  }
  // Lower startup retained only a private duplicate pair. Once its explicit
  // authority has been captured, close the borrowed originals exactly once;
  // a still-live close failure is retained by this owner for retry.
  let kernel_close_failed = close_or_retain_borrowed_root(owner, kernel_root)
  let model_close_failed = close_or_retain_borrowed_root(owner, model_root)
  if kernel_close_failed || model_close_failed {
    if owner.preparations.length() == 1 &&
      owner.preparations[0].state() == LunaOnlineInstanceReady {
      let ready = owner.preparations[0].take_ready() catch {
        _ => {
          owner.phase = Failed
          return owner
        }
      }
      owner.preparations.clear()
      ready.begin_drain()
      owner.services.push(ready)
    }
    owner.phase = Failed
  }
  owner
}

///|
fn close_or_retain_borrowed_root(
  owner : RuntimeInstanceOwner,
  root : @approved_fs.ApprovedRoot,
) -> Bool {
  if root.is_closed() {
    return false
  }
  let failed = Ref(false)
  root.close() catch {
    _ => failed.val = true
  }
  if retain_borrowed_root_after_close(failed.val, root.is_closed()) {
    owner.borrowed_roots.push(root)
  }
  failed.val
}

///|
fn close_three_roots(
  model : @approved_fs.ApprovedRoot,
  kernel : @approved_fs.ApprovedRoot,
  policy : @approved_fs.ApprovedRoot,
) -> Unit raise RuntimeInstanceError {
  let failed = Ref(false)
  policy.close() catch {
    _ => failed.val = true
  }
  kernel.close() catch {
    _ => failed.val = true
  }
  model.close() catch {
    _ => failed.val = true
  }
  if failed.val {
    raise fail(Cleanup)
  }
}

///|
fn approval_from_launch(
  launch : @launch_file.LaunchFileAdmission,
) -> @worker_wire.WorkerBootstrapLunaApproval? raise RuntimeInstanceError {
  match launch.luna_approval() {
    None => None
    Some(approval) =>
      Some(
        @worker_wire.WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
          manifest_digest_sha256=approval.capability_manifest_sha256(),
          key_identity_sha256=approval.key_identity_sha256(),
          signature_digest_sha256=approval.signature_sha256(),
          approved_source_sha256=approval.approved_source_sha256(),
        ) catch {
          _ => raise fail(LaunchAdmission)
        },
      )
  }
}

///|
fn admit_physical_device(
  runtime : @descriptor_file.RuntimeDescriptorAdmission,
) -> Unit raise RuntimeInstanceError {
  admit_physical_target(runtime.target(), runtime.device_ordinal())
}

///|
/// I8 host admission checks the same exact ordinal/target facts before worker
/// execution. The child remains authoritative for exact memory and numeric
/// capability re-observation before it can publish executor readiness.
fn admit_i8_physical_device(
  runtime : @descriptor_file.I8RuntimeDescriptorAdmission,
) -> Unit raise RuntimeInstanceError {
  admit_physical_target(runtime.target(), runtime.device_ordinal())
}

///|
fn admit_physical_target(
  target : @catalog.DeviceTarget,
  device_ordinal : Int,
) -> Unit raise RuntimeInstanceError {
  match @hardware.probe() {
    Unavailable(_) => raise fail(PhysicalDeviceAdmission)
    Available(driver_version=_, devices~) => {
      let mut selected = None
      for device in devices {
        if device.ordinal() == device_ordinal {
          selected = Some(device)
        }
      }
      guard selected is Some(device) else {
        raise fail(PhysicalDeviceAdmission)
      }
      if !target.matches_capability(device.capability()) {
        raise fail(PhysicalDeviceAdmission)
      }
    }
  }
}

///|
fn load_tokenizer(
  model_root : @approved_fs.ApprovedRoot,
  model_root_label : String,
  policy : @instance_policy_file.InstancePolicyAdmission,
) -> @json_file.TokenizerFileAdmission raise RuntimeInstanceError {
  let claims = policy.tokenizer()
  let digest = @tokenizer.TokenizerDigest::parse(claims.sha256()) catch {
    _ => raise fail(TokenizerAdmission)
  }
  let file_limits = @json_file.TokenizerFileLimits::new(
    max_file_bytes=claims.max_file_bytes(),
    max_json_depth=claims.max_json_depth(),
  ) catch {
    _ => raise fail(TokenizerAdmission)
  }
  let tokenizer_limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: claims.max_input_bytes(),
    max_output_tokens: claims.max_output_tokens(),
    max_decoded_bytes: claims.max_decoded_bytes(),
    max_vocab_entries: claims.max_vocab_entries(),
    max_merge_rules: claims.max_merge_rules(),
    max_token_bytes: claims.max_token_bytes(),
    max_special_tokens: claims.max_special_tokens(),
  }
  @json_file.load(
    model_root,
    model_root_label,
    claims.locator(),
    digest,
    file_limits,
    tokenizer_limits,
  ) catch {
    _ => raise fail(TokenizerAdmission)
  }
}

///|
fn load_materialized_tokenizer(
  model : @approved_fs.ApprovedRootMaterializationView,
  policy : @instance_policy_file.InstancePolicyAdmission,
) -> @json_file.TokenizerFileAdmission raise RuntimeInstanceError {
  let claims = policy.tokenizer()
  let digest = @tokenizer.TokenizerDigest::parse(claims.sha256()) catch {
    _ => raise fail(TokenizerAdmission)
  }
  let file_limits = @json_file.TokenizerFileLimits::new(
    max_file_bytes=claims.max_file_bytes(),
    max_json_depth=claims.max_json_depth(),
  ) catch {
    _ => raise fail(TokenizerAdmission)
  }
  let tokenizer_limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: claims.max_input_bytes(),
    max_output_tokens: claims.max_output_tokens(),
    max_decoded_bytes: claims.max_decoded_bytes(),
    max_vocab_entries: claims.max_vocab_entries(),
    max_merge_rules: claims.max_merge_rules(),
    max_token_bytes: claims.max_token_bytes(),
    max_special_tokens: claims.max_special_tokens(),
  }
  @json_file.load_materialized(
    model,
    claims.locator(),
    digest,
    file_limits,
    tokenizer_limits,
  ) catch {
    _ => raise fail(TokenizerAdmission)
  }
}
