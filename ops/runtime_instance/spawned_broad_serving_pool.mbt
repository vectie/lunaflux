///|
fn RuntimeInstanceOwner::broad_serving_uses_framed_pool(
  self : RuntimeInstanceOwner,
) -> Bool {
  self.servers.length() == 0 && self.framed_pools.length() == 1
}

///|
async fn await_broad_serving_pool_connections(
  owner : RuntimeInstanceOwner,
  expected_connections : Int,
  turn_limit : Int,
) -> Unit {
  for turn in 0..<turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy {
      raise broad_serving_error(BroadServingHealth)
    }
    if !owner.broad_serving_uses_framed_pool() ||
      !owner.spawned_serving_ingress_live() {
      raise broad_serving_error(BroadServingReadiness)
    }
    let pool = owner.framed_pools[0]
    if pool.active_connection_count() == expected_connections {
      return
    }
    if pool.active_connection_count() > expected_connections {
      raise broad_serving_error(BroadServingConcurrency)
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(BroadServingConcurrency)
}

///|
async fn progress_broad_serving_pool_pair(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
) -> Unit {
  for turn in 0..<contract.turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy {
      raise broad_serving_error(BroadServingHealth)
    }
    if !owner.broad_serving_uses_framed_pool() ||
      !owner.spawned_serving_ingress_live() {
      raise broad_serving_error(BroadServingReadiness)
    }
    let sample = capture_broad_serving_metrics(owner)
    if broad_serving_pair_metrics_valid(
        sample,
        contract.input_tokens,
        initial_kv_pages_free,
        pooled=true,
      ) {
      return
    }
    if sample.failures != 0UL ||
      sample.worker_failures != 0UL ||
      sample.deadlines != 0UL ||
      sample.network_rejections != 0UL ||
      sample.cancellations > 1UL ||
      sample.completions > 2UL ||
      sample.network_accepts > 3UL ||
      sample.network_disconnects > 3UL {
      raise broad_serving_error(BroadServingMetrics)
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(BroadServingBackpressure)
}

///|
async fn require_broad_serving_pool_cancel_boundary(
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
    2,
    BroadServingPairAccepted0,
  )
  let sample = capture_broad_serving_metrics(owner)
  if !broad_serving_pool_cancel_boundary_valid(sample) {
    raise broad_serving_error(BroadServingConcurrency)
  }
}

///|
async fn await_broad_serving_pool_cancelled_lane(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
) -> Unit {
  for turn in 0..<contract.turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy ||
      !owner.broad_serving_uses_framed_pool() ||
      !owner.spawned_serving_ingress_live() {
      raise broad_serving_error(BroadServingReadiness)
    }
    let sample = capture_broad_serving_metrics(owner)
    if sample.admissions == 2UL &&
      sample.cancellations == 1UL &&
      sample.completions <= 1UL &&
      sample.generated_tokens <= 2UL &&
      owner.framed_pools[0].active_connection_count() < 2 {
      return
    }
    if !broad_serving_pool_cancel_progress_valid(sample) {
      raise broad_serving_error(BroadServingCancellation)
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(BroadServingCancellation)
}

///|
async fn await_broad_serving_pool_accept_count(
  owner : RuntimeInstanceOwner,
  expected_accepts : UInt64,
  turn_limit : Int,
) -> Unit {
  for turn in 0..<turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy ||
      !owner.broad_serving_uses_framed_pool() ||
      !owner.spawned_serving_ingress_live() {
      raise broad_serving_error(BroadServingReadiness)
    }
    let sample = capture_broad_serving_metrics(owner)
    if sample.network_accepts == expected_accepts {
      return
    }
    if sample.network_accepts > expected_accepts {
      raise broad_serving_error(BroadServingConcurrency)
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(BroadServingConnect)
}

///|
async fn require_broad_serving_pool_survivor_events(
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
      BroadServingPairAccepted1,
    ),
    contract,
    2,
    BroadServingPairAccepted1,
  )
  require_broad_token(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairRemainingRead,
    ),
    contract.expected_first_token,
    0,
    BroadServingPairRemainingToken0Shape,
  )
  require_broad_token(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairRemainingRead,
    ),
    contract.expected_second_token,
    1,
    BroadServingPairRemainingToken1Shape,
  )
  require_broad_usage(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairRemainingRead,
    ),
    contract.input_tokens,
    2,
    BroadServingPairRemainingUsageShape,
  )
  require_broad_completed(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairRemainingRead,
    ),
    contract.input_tokens,
    2,
    TokenLimit,
    BroadServingPairRemainingCompletedShape,
  )
}

///|
async fn require_broad_serving_pool_reuse_events(
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
      BroadServingPairRemainingRead,
    ),
    contract,
    1,
    BroadServingPairRemainingAcceptedShape,
  )
  require_broad_token(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairRemainingRead,
    ),
    contract.expected_first_token,
    0,
    BroadServingPairRemainingToken0Shape,
  )
  require_broad_usage(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairRemainingRead,
    ),
    contract.input_tokens,
    1,
    BroadServingPairRemainingUsageShape,
  )
  require_broad_completed(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingPairRemainingRead,
    ),
    contract.input_tokens,
    1,
    TokenLimit,
    BroadServingPairRemainingCompletedShape,
  )
}

///|
async fn run_broad_serving_pool_pair(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
) -> Unit {
  let cancelled = connect_broad_serving_client(
    owner,
    contract.io_timeout_millis,
  )
  let mut cancelled_open = true
  defer (if cancelled_open { cancelled.close() })
  let survivor = connect_broad_serving_client(owner, contract.io_timeout_millis)
  let mut survivor_open = true
  defer (if survivor_open { survivor.close() })
  await_broad_serving_pool_connections(owner, 2, contract.turn_limit)
  write_broad_serving_frame(
    cancelled,
    contract.saturation_tail_frame,
    contract.io_timeout_millis,
  )
  write_broad_serving_frame(
    survivor,
    contract.recovery_frame,
    contract.io_timeout_millis,
  )
  require_broad_serving_pool_cancel_boundary(owner, cancelled, contract)
  cancelled.close()
  cancelled_open = false
  await_broad_serving_pool_cancelled_lane(owner, contract)
  let reuse = connect_broad_serving_client(owner, contract.io_timeout_millis)
  let mut reuse_open = true
  defer (if reuse_open { reuse.close() })
  await_broad_serving_pool_accept_count(owner, 3UL, contract.turn_limit)
  write_broad_serving_frame(
    reuse,
    contract.first_frame,
    contract.io_timeout_millis,
  )
  progress_broad_serving_pool_pair(owner, contract, initial_kv_pages_free)
  require_broad_serving_pool_survivor_events(owner, survivor, contract)
  require_broad_serving_pool_reuse_events(owner, reuse, contract)
  survivor.close()
  survivor_open = false
  reuse.close()
  reuse_open = false
  retire_broad_serving_client(owner, contract.turn_limit, 3UL, 3UL, 0UL)
}
