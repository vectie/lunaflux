///|
priv enum SpawnedServingValidationOutcome {
  SpawnedServingSucceeded(RuntimeSpawnedServingValidationResult)
  SpawnedServingRejectedOutcome(RuntimeSpawnedServingValidationRule)
}

///|
async fn bind_spawned_serving_listener(owner : RuntimeInstanceOwner) -> Unit {
  owner.require_spawned_physical_handoff() catch {
    _ => raise spawned_serving_error(SpawnedServingOwner)
  }
  if !spawned_serving_prebind_state_valid(
      owner.phase(),
      owner.health(),
      owner.readiness(),
    ) {
    raise spawned_serving_error(SpawnedServingReadiness)
  }
  if owner.progress() != RuntimeReadyProgress ||
    !owner.spawned_serving_ingress_shape_valid() {
    raise spawned_serving_error(SpawnedServingOwner)
  }
  if owner.health() != RuntimeHealthy {
    raise spawned_serving_error(SpawnedServingHealth)
  }
  if !spawned_serving_bound_state_valid(
      owner.phase(),
      owner.health(),
      owner.readiness(),
    ) {
    raise spawned_serving_error(SpawnedServingReadiness)
  }
}

///|
async fn accept_spawned_serving_client(
  owner : RuntimeInstanceOwner,
  turn_limit : Int,
) -> Unit {
  for turn in 0..<turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy {
      raise spawned_serving_error(SpawnedServingHealth)
    }
    if !owner.spawned_serving_ingress_live() {
      raise spawned_serving_error(SpawnedServingReadiness)
    }
    if owner.spawned_serving_ingress_active() {
      return
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise spawned_serving_error(SpawnedServingConnect)
}

///|
async fn progress_spawned_serving_request(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedServingValidationContract,
  initial_kv_pages_free : Int,
) -> SpawnedServingMetricSample {
  for turn in 0..<contract.turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy {
      raise spawned_serving_error(SpawnedServingHealth)
    }
    if !owner.spawned_serving_terminal_ingress_live() {
      raise spawned_serving_error(SpawnedServingReadiness)
    }
    let sample = capture_spawned_serving_metrics(owner)
    if spawned_serving_terminal_metrics_valid(
        sample,
        contract.input_tokens,
        initial_kv_pages_free,
        owner.spawned_serving_terminal_disconnected(),
      ) {
      return sample
    }
    if sample.failures != 0UL ||
      sample.worker_failures != 0UL ||
      sample.network_rejections != 0UL ||
      sample.cancellations != 0UL ||
      sample.deadlines != 0UL {
      raise spawned_serving_error(SpawnedServingMetrics)
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise spawned_serving_error(SpawnedServingMetrics)
}

///|
async fn retire_spawned_serving_client(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedServingValidationContract,
  initial_kv_pages_free : Int,
) -> SpawnedServingMetricSample {
  for turn in 0..<contract.turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy {
      raise spawned_serving_error(SpawnedServingHealth)
    }
    if !owner.spawned_serving_ingress_live() {
      raise spawned_serving_error(SpawnedServingReadiness)
    }
    let sample = capture_spawned_serving_metrics(owner)
    if owner.spawned_serving_ingress_listening() &&
      spawned_serving_terminal_metrics_valid(
        sample,
        contract.input_tokens,
        initial_kv_pages_free,
        true,
      ) {
      return sample
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise spawned_serving_error(SpawnedServingMetrics)
}

///|
async fn run_spawned_serving_validation(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedServingValidationContract,
) -> RuntimeSpawnedServingValidationResult {
  bind_spawned_serving_listener(owner)
  let initial = capture_spawned_serving_metrics(owner)
  if !spawned_serving_initial_metrics_valid(initial) {
    raise spawned_serving_error(SpawnedServingMetrics)
  }
  let client = connect_spawned_serving_client(owner, contract.io_timeout_millis)
  let mut client_open = true
  defer (if client_open { client.close() })
  accept_spawned_serving_client(owner, contract.turn_limit)
  write_spawned_serving_frame(
    client,
    contract.frame,
    contract.io_timeout_millis,
  )
  let decoder = @framed_wire.EventFrameBuffer::new(contract.framed_limits)
  let storage = FixedArray::make(decoder.capacity(), b'\x00')
  require_spawned_serving_accepted(
    read_spawned_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
    ),
    contract,
  )
  let first = require_spawned_serving_token(
    read_spawned_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
    ),
    contract.expected_first_token,
    0,
    SpawnedServingFirstToken,
  )
  let second = require_spawned_serving_token(
    read_spawned_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
    ),
    contract.expected_second_token,
    1,
    SpawnedServingSecondToken,
  )
  require_spawned_serving_usage(
    read_spawned_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
    ),
    contract.input_tokens,
  )
  require_spawned_serving_completed(
    read_spawned_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
    ),
    contract.input_tokens,
  )
  ignore(
    progress_spawned_serving_request(owner, contract, initial.kv_pages_free),
  )
  client.close()
  client_open = false
  let retired = retire_spawned_serving_client(
    owner,
    contract,
    initial.kv_pages_free,
  )
  {
    first_token: first,
    second_token: second,
    event_count: 5,
    kv_pages_free: retired.kv_pages_free,
    network_accepts: retired.network_accepts,
    network_disconnects: retired.network_disconnects,
  }
}

///|
async fn drain_spawned_serving_owner(
  owner : RuntimeInstanceOwner,
  turn_limit : Int,
) -> Bool {
  if owner.cleanup_complete() {
    return true
  }
  match owner.phase() {
    Starting | HealthyNotReady | Ready | OpenAiQualificationReady =>
      owner.begin_drain() catch {
        _ => ()
      }
    Draining | Failed | Closed => ()
  }
  for turn in 0..<turn_limit {
    if owner.cleanup_complete() {
      return true
    }
    ignore(owner.progress() catch { _ => RuntimeFailedProgress })
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  owner.cleanup_complete()
}

///|
/// Drives one authenticated spawned runtime through its actual native TCP
/// pipeline listener. The method owns its loopback client internally and
/// returns no listener, service, scheduler, worker, or device authority.
/// Success requires separate healthy/ready observations, the canonical five
/// events, exact request/KV/network balance, listener drain, and child reap.
pub async fn RuntimeInstanceOwner::validate_spawned_serving_request(
  self : RuntimeInstanceOwner,
  contract : RuntimeSpawnedServingValidationContract,
) -> RuntimeSpawnedServingValidationResult {
  @async.protect_from_cancel(() => {
    let outcome = try run_spawned_serving_validation(self, contract) catch {
      RuntimeSpawnedServingValidationFailed(rule) =>
        SpawnedServingRejectedOutcome(rule)
      _ => SpawnedServingRejectedOutcome(SpawnedServingOwner)
    } noraise {
      result => SpawnedServingSucceeded(result)
    }
    let cleaned = drain_spawned_serving_owner(self, contract.turn_limit)
    if !cleaned {
      raise spawned_serving_error(SpawnedServingCleanup)
    }
    match outcome {
      SpawnedServingSucceeded(result) => {
        if !spawned_serving_closed_state_valid(
            self.phase(),
            self.health(),
            self.readiness(),
            self.cleanup_complete(),
          ) {
          raise spawned_serving_error(SpawnedServingCleanup)
        }
        result
      }
      SpawnedServingRejectedOutcome(rule) => raise spawned_serving_error(rule)
    }
  })
}
