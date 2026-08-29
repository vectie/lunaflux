///|
priv enum SpawnedBroadServingOutcome {
  SpawnedBroadServingSucceeded(RuntimeSpawnedBroadServingValidationResult)
  SpawnedBroadServingRejected(RuntimeSpawnedBroadServingValidationRule)
}

///|
async fn progress_broad_serving_pair(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
) -> Unit {
  let mut saw_two_live = false
  let mut cancel_attempted = false
  for turn in 0..<contract.turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy {
      raise broad_serving_error(BroadServingHealth)
    }
    if owner.servers.length() != 1 ||
      owner.servers[0].state() != LunaOnlineTcpServerConnected ||
      !spawned_serving_live_state_valid(
        owner.phase(),
        owner.health(),
        owner.readiness(),
        owner.servers[0].state(),
      ) {
      raise broad_serving_error(BroadServingReadiness)
    }
    let sample = capture_broad_serving_metrics(owner)
    let live = sample.active_requests + sample.queue_depth
    if sample.admissions == 2UL && live == 2 {
      saw_two_live = true
    }
    if saw_two_live && !cancel_attempted {
      cancel_attempted = true
      owner.servers[0].request_cancel() catch {
        _ => raise broad_serving_error(BroadServingCancellation)
      }
    }
    if broad_serving_pair_metrics_valid(
        sample,
        contract.input_tokens,
        initial_kv_pages_free,
      ) {
      if !saw_two_live || !cancel_attempted {
        raise broad_serving_error(BroadServingConcurrency)
      }
      return
    }
    if sample.failures != 0UL ||
      sample.worker_failures != 0UL ||
      sample.deadlines != 0UL ||
      sample.network_rejections != 0UL ||
      sample.cancellations > 1UL ||
      sample.completions > 2UL {
      raise broad_serving_error(BroadServingMetrics)
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(
    if saw_two_live {
      BroadServingBackpressure
    } else {
      BroadServingConcurrency
    },
  )
}

///|
async fn require_broad_serving_pair_events(
  owner : RuntimeInstanceOwner,
  client : @socket.Tcp,
  contract : RuntimeSpawnedBroadServingValidationContract,
) -> Unit {
  let decoder = @framed_wire.EventFrameBuffer::new(contract.framed_limits)
  let storage = FixedArray::make(decoder.capacity(), b'\x00')
  require_broad_accepted(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairAccepted0,
    ),
    contract,
    1,
    BroadServingPairAccepted0,
  )
  require_broad_accepted(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairAccepted1,
    ),
    contract,
    1,
    BroadServingPairAccepted1,
  )
  let ledger = BroadServingEventLedger::new()
  for _ in 0..<10 {
    ledger.record(
      read_broad_serving_event(
        owner,
        client,
        storage,
        decoder,
        contract.io_timeout_millis,
        BroadServingPairRemainingRead,
      ),
      contract,
    )
  }
  ledger.require_exact()
}

///|
async fn run_broad_serving_pair(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
) -> Unit {
  let client = connect_broad_serving_client(owner, contract.io_timeout_millis)
  let mut client_open = true
  defer (if client_open { client.close() })
  accept_broad_serving_client(owner, contract.turn_limit)
  write_broad_serving_frame(
    client,
    broad_serving_three_frames(
      contract.first_frame,
      contract.second_frame,
      contract.saturation_tail_frame,
    ),
    contract.io_timeout_millis,
  )
  progress_broad_serving_pair(owner, contract, initial_kv_pages_free)
  require_broad_serving_pair_events(owner, client, contract)
  client.close()
  client_open = false
  retire_broad_serving_client(owner, contract.turn_limit, 1UL, 1UL, 0UL)
}

///|
async fn run_broad_serving_foreign_rejection(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
) -> Unit {
  let client = connect_broad_serving_client(owner, contract.io_timeout_millis)
  let mut client_open = true
  defer (if client_open { client.close() })
  accept_broad_serving_client(owner, contract.turn_limit)
  write_broad_serving_frame(
    client,
    contract.foreign_frame,
    contract.io_timeout_millis,
  )
  retire_broad_serving_client(owner, contract.turn_limit, 2UL, 2UL, 1UL)
  client.close()
  client_open = false
  let sample = capture_broad_serving_metrics(owner)
  if sample.admissions != 3UL ||
    sample.completions != 2UL ||
    sample.cancellations != 1UL ||
    sample.queue_depth != 0 ||
    sample.active_requests != 0 ||
    sample.kv_pages_used != 0 {
    raise broad_serving_error(BroadServingRejection)
  }
}

///|
async fn run_broad_serving_recovery(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
) -> Unit {
  let client = connect_broad_serving_client(owner, contract.io_timeout_millis)
  let mut client_open = true
  defer (if client_open { client.close() })
  accept_broad_serving_client(owner, contract.turn_limit)
  write_broad_serving_frame(
    client,
    contract.recovery_frame,
    contract.io_timeout_millis,
  )
  let decoder = @framed_wire.EventFrameBuffer::new(contract.framed_limits)
  let storage = FixedArray::make(decoder.capacity(), b'\x00')
  require_broad_accepted(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingRecoveryAccepted,
    ),
    contract,
    2,
    BroadServingRecoveryAccepted,
  )
  require_broad_token(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingRecoveryToken0,
    ),
    contract.expected_first_token,
    0,
    BroadServingRecoveryToken0,
  )
  require_broad_token(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingRecoveryToken1,
    ),
    contract.expected_second_token,
    1,
    BroadServingRecoveryToken1,
  )
  require_broad_usage(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingRecoveryUsage,
    ),
    contract.input_tokens,
    2,
    BroadServingRecoveryUsage,
  )
  require_broad_completed(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingRecoveryCompleted,
    ),
    contract.input_tokens,
    2,
    TokenLimit,
    BroadServingRecoveryCompleted,
  )
  for turn in 0..<contract.turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy {
      raise broad_serving_error(BroadServingHealth)
    }
    if owner.servers.length() != 1 ||
      owner.servers[0].state() != LunaOnlineTcpServerConnected ||
      !spawned_serving_live_state_valid(
        owner.phase(),
        owner.health(),
        owner.readiness(),
        owner.servers[0].state(),
      ) {
      raise broad_serving_error(BroadServingReadiness)
    }
    let sample = capture_broad_serving_metrics(owner)
    if sample.admissions == 4UL &&
      sample.completions == 3UL &&
      sample.cancellations == 1UL &&
      sample.queue_depth == 0 &&
      sample.active_requests == 0 &&
      sample.kv_pages_used == 0 &&
      sample.kv_pages_free == initial_kv_pages_free {
      client.close()
      client_open = false
      retire_broad_serving_client(owner, contract.turn_limit, 3UL, 3UL, 1UL)
      return
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(BroadServingRecovery)
}

///|
async fn run_spawned_broad_serving_validation(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
) -> RuntimeSpawnedBroadServingValidationResult {
  bind_spawned_serving_listener(owner) catch {
    _ => raise broad_serving_error(BroadServingOwner)
  }
  let initial = capture_broad_serving_metrics(owner)
  if !broad_serving_initial_metrics_valid(initial) {
    raise broad_serving_error(BroadServingMetrics)
  }
  run_broad_serving_pair(owner, contract, initial.kv_pages_free)
  run_broad_serving_foreign_rejection(owner, contract)
  run_broad_serving_recovery(owner, contract, initial.kv_pages_free)
  run_broad_serving_malformed_fail_closed(
    owner,
    contract,
    initial.kv_pages_free,
  )
}

///|
/// Drives one authenticated spawned runtime through a bounded, qualification-
/// only native campaign. No listener, worker, scheduler, device, or mutable
/// service authority escapes this method. A malformed frame must retire only
/// its connection; a canonical request then proves same-owner reuse before the
/// validation wrapper drains every retained authority.
pub async fn RuntimeInstanceOwner::validate_spawned_broad_serving(
  self : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
) -> RuntimeSpawnedBroadServingValidationResult {
  @async.protect_from_cancel(() => {
    let outcome = try
      run_spawned_broad_serving_validation(self, contract)
    catch {
      RuntimeSpawnedBroadServingValidationFailed(rule) =>
        SpawnedBroadServingRejected(rule)
      _ => SpawnedBroadServingRejected(BroadServingOwner)
    } noraise {
      result => SpawnedBroadServingSucceeded(result)
    }
    let cleaned = drain_spawned_serving_owner(self, contract.turn_limit) catch {
      _ => false
    }
    if !cleaned {
      raise broad_serving_error(BroadServingCleanup)
    }
    match outcome {
      SpawnedBroadServingSucceeded(result) => {
        if !spawned_serving_closed_state_valid(
            self.phase(),
            self.health(),
            self.readiness(),
            self.cleanup_complete(),
          ) {
          raise broad_serving_error(BroadServingCleanup)
        }
        result
      }
      SpawnedBroadServingRejected(rule) => raise broad_serving_error(rule)
    }
  })
}
