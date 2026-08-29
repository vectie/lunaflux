///|
test "health and readiness projections remain independent" {
  assert_eq(health_for_phase(Starting), RuntimeStarting)
  assert_eq(health_for_phase(HealthyNotReady), RuntimeHealthy)
  assert_eq(readiness_for_phase(HealthyNotReady, true), RuntimeNotReady)
  assert_eq(readiness_for_phase(Ready, false), RuntimeNotReady)
  assert_eq(readiness_for_phase(Ready, true), RuntimeReady)
  assert_eq(health_for_phase(Failed), RuntimeUnhealthy)
  assert_eq(readiness_for_phase(Failed, true), RuntimeNotReady)
  assert_eq(health_for_phase(Closed), RuntimeClosed)
  assert_true(runtime_live_health(true, true, true))
  assert_false(runtime_live_health(false, true, true))
  // Temporary traffic backpressure belongs only to readiness. It must not
  // fabricate a process-health failure while all live owners remain healthy.
  assert_eq(readiness_for_phase(Ready, false), RuntimeNotReady)
}

///|
test "root-free service protocol projection is truthful" {
  let owner = empty_owner_for_state_test(HealthyNotReady)
  assert_eq(owner.service_protocol(), RuntimeNativeFramedV1)
}

///|
test "spawned execution extensions match exact authenticated release state" {
  let host = empty_owner_for_state_test(HealthyNotReady).release_admission
  assert_true(host.matches_spawned_execution_extensions(true, false))
  assert_false(host.matches_spawned_execution_extensions(false, false))
  assert_false(host.matches_spawned_execution_extensions(true, true))
  let embedded = {
    ..host,
    sampling_runtime: EmbeddedCudaGreedyV1,
    has_fused_runtime: true,
  }
  assert_true(embedded.matches_spawned_execution_extensions(false, false))
  assert_true(embedded.matches_spawned_execution_extensions(false, true))
  assert_false(embedded.matches_spawned_execution_extensions(true, true))
}

///|
test "authenticated capacity one preserves singleton and larger capacity selects pool" {
  assert_false(runtime_uses_concurrent_ingress(1))
  assert_true(runtime_uses_concurrent_ingress(2))
  assert_true(runtime_uses_concurrent_ingress(8))
}

///|
test "startup measurement is absent before publication" {
  let owner = empty_owner_for_state_test(HealthyNotReady)
  try owner.startup_measurement() catch {
    RuntimeInstanceFailed(ColdStartTelemetry) => ()
    _ => abort("unexpected pre-publication startup measurement rejection")
  } noraise {
    _ => abort("unpublished startup measurement escaped")
  }
}

///|
test "startup measurement preserves exact duration and false readiness" {
  let owner = empty_owner_for_state_test(OpenAiQualificationReady)
  owner.cold_start_recorded = true
  owner.cold_start_latency_millis = 86400000
  let measurement = try! owner.startup_measurement()
  assert_eq(measurement.cold_start_latency_millis(), 86400000)
  assert_eq(measurement.readiness(), RuntimeNotReady)
}

///|
test "inherited drain decisions are exact and idempotent" {
  match runtime_inherited_drain_action(Ready, InheritedDrainRequested) {
    RuntimeInheritedDrainAccept => ()
    _ => abort("ready drain request was not accepted")
  }
  match runtime_inherited_drain_action(Draining, InheritedDrainRequested) {
    RuntimeInheritedDrainIdempotent => ()
    _ => abort("draining request was not idempotent")
  }
  match runtime_inherited_drain_action(Closed, InheritedDrainRequested) {
    RuntimeInheritedDrainClosedResponse => ()
    _ => abort("closed request did not select the closed response")
  }
  match runtime_inherited_drain_action(Ready, InheritedDrainChannelLost) {
    RuntimeInheritedDrainAuthorityLost => ()
    _ => abort("ready owner tolerated lost drain authority")
  }
  match runtime_inherited_drain_action(Draining, InheritedDrainChannelLost) {
    RuntimeInheritedDrainContinue => ()
    _ => abort("draining owner did not tolerate post-request channel loss")
  }
}

///|
fn empty_owner_for_state_test(
  phase : RuntimeInstancePhase,
) -> RuntimeInstanceOwner raise {
  {
    phase,
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
    opaque_cli_requires_drain: false,
    release_admission: {
      runtime_recipe: "dense_llama_paged_aot_v5",
      model_content_sha256: "a".repeat(64),
      model_plan_sha256: "b".repeat(64),
      tokenizer_sha256: "c".repeat(64),
      worker_executable_sha256: "d".repeat(64),
      sampling_runtime: HostSamplingRuntime,
      has_fused_runtime: false,
      evidence: "owner state fixture",
      evidence_sha256: "e".repeat(64),
    },
    startup_binding: RuntimeOpaqueNativeBinding,
    borrowed_roots: Array::new(capacity=2),
    addr: @socket.Addr::parse("127.0.0.1:0"),
    transport: @online_tcp.LunaOnlineTcpLimits::new(
      read_chunk_bytes=64,
      write_chunk_bytes=64,
      accept_timeout_millis=1,
      input_idle_timeout_millis=1,
      write_timeout_millis=1,
      reactor_transition_budget=1,
    ),
    telemetry_capacity: 1,
    connection_capacity: 1,
    request_work_units: 1,
    event_work_units: 1,
    receipt_deadline: @inference.DeadlineBudget::from_milliseconds(
      5000UL,
      spawned_serving_wb_limits(),
    ),
    loopback_openai_allowed: true,
    openai_listener_allowed: true,
    clock: @monotonic_clock.MonotonicClock::system(),
    started_at_millis: 0UL,
    cold_start_recorded: false,
    cold_start_latency_millis: -1,
  }
}

///|
test "borrowed root close outcomes retain only live failed authority" {
  // Busy leaves a live authority; a consumed POSIX close failure is closed and
  // must fail the instance without retrying a potentially reused descriptor.
  assert_false(retain_borrowed_root_after_close(false, true))
  assert_false(retain_borrowed_root_after_close(true, true))
  assert_true(retain_borrowed_root_after_close(true, false))
  assert_false(retain_borrowed_root_after_close(false, false))
}

///|
test "spawned physical handoff rejects every synthetic owner state" {
  for phase in [Starting, HealthyNotReady, Ready, Draining, Closed, Failed] {
    let owner = empty_owner_for_state_test(phase)
    try owner.require_spawned_physical_handoff() catch {
      RuntimeInstanceFailed(SpawnedPhysicalHandoff) => ()
      _ => abort("unexpected spawned physical handoff rejection")
    } noraise {
      _ => abort("synthetic owner fabricated a spawned physical handoff")
    }
  }
}

///|
test "spawned physical handoff scalar predicate is exact" {
  assert_true(
    spawned_physical_handoff_matches(HealthyNotReady, true, 1, 0, 0, 0, 0, 0),
  )
  assert_false(spawned_physical_handoff_matches(Ready, true, 1, 0, 0, 0, 0, 0))
  assert_false(
    spawned_physical_handoff_matches(HealthyNotReady, false, 1, 0, 0, 0, 0, 0),
  )
  assert_false(
    spawned_physical_handoff_matches(HealthyNotReady, true, 0, 0, 0, 0, 0, 0),
  )
  assert_false(
    spawned_physical_handoff_matches(HealthyNotReady, true, 1, 1, 0, 0, 0, 0),
  )
  assert_false(
    spawned_physical_handoff_matches(HealthyNotReady, true, 1, 0, 1, 0, 0, 0),
  )
  assert_false(
    spawned_physical_handoff_matches(HealthyNotReady, true, 1, 0, 0, 1, 0, 0),
  )
  assert_false(
    spawned_physical_handoff_matches(HealthyNotReady, true, 1, 0, 0, 0, 1, 0),
  )
  assert_false(
    spawned_physical_handoff_matches(HealthyNotReady, true, 1, 0, 0, 0, 0, 1),
  )
}

///|
test "OpenAI loopback host admission is exact" {
  assert_true(loopback_openai_host("127.0.0.1"))
  assert_true(loopback_openai_host("localhost"))
  assert_false(loopback_openai_host("0.0.0.0"))
  assert_false(loopback_openai_host("127.0.0.2"))
  assert_false(loopback_openai_host("example.invalid"))
}

///|
test "captured lower outcome closes both borrowed originals exactly once" {
  let owner = empty_owner_for_state_test(Failed)
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  assert_false(close_or_retain_borrowed_root(owner, kernel))
  assert_false(close_or_retain_borrowed_root(owner, model))
  assert_true(model.is_closed())
  assert_true(kernel.is_closed())
  assert_eq(owner.borrowed_roots.length(), 0)
}

///|
async test "failed owner retries only a retained live borrowed original" {
  let owner = empty_owner_for_state_test(Failed)
  let root = @approved_fs.ApprovedRoot::open_absolute("/usr")
  owner.borrowed_roots.push(root)
  assert_false(owner.cleanup_complete())
  assert_eq(owner.progress(), RuntimeFailedProgress)
  assert_true(root.is_closed())
  assert_true(owner.cleanup_complete())
}

///|
async test "drain and failed fake seams never fabricate readiness or lose cleanup" {
  let starting = empty_owner_for_state_test(Starting)
  starting.begin_drain()
  assert_eq(starting.phase(), Draining)
  assert_eq(starting.readiness(), RuntimeNotReady)
  assert_eq(starting.progress(), RuntimeClosedProgress)
  assert_true(starting.cleanup_complete())
  let failed = empty_owner_for_state_test(Failed)
  assert_eq(failed.health(), RuntimeUnhealthy)
  assert_eq(failed.readiness(), RuntimeNotReady)
  assert_eq(failed.progress(), RuntimeFailedProgress)
  assert_true(failed.cleanup_complete())
  try failed.metrics_snapshot() catch {
    RuntimeInstanceFailed(Lifecycle) => ()
    _ => abort("unexpected metrics-without-server rejection")
  } noraise {
    _ => abort("owner fabricated a metrics snapshot without server authority")
  }
}

///|
test "spawned validation owner rejection preserves cleanup authority" {
  let owner = empty_owner_for_state_test(HealthyNotReady)
  let contract = RuntimeSpawnedValidationContract::new(
    [b'a'],
    spawned_validation_wb_identity(),
    1031,
    2185,
    1,
    3,
    8,
    16,
  )
  try owner.validate_spawned_physical_request(contract) catch {
    RuntimeSpawnedValidationFailed(SpawnedValidationOwner) => ()
    _ => abort("synthetic validation owner failed at the wrong boundary")
  } noraise {
    _ => abort("synthetic owner executed a spawned validation request")
  }
  assert_true(owner.cleanup_complete())
  assert_eq(owner.phase(), HealthyNotReady)
}
