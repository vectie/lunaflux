///|
fn require_broad_serving_malformed_metrics(
  sample : BroadServingMetricSample,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
  pooled? : Bool = false,
) -> Unit raise RuntimeSpawnedBroadServingValidationError {
  if !broad_serving_malformed_request_metrics_valid(
      sample,
      contract.input_tokens,
      pooled~,
    ) {
    raise broad_serving_error(BroadServingMalformedRequestMetrics)
  }
  if !broad_serving_malformed_network_metrics_valid(sample, pooled~) {
    raise broad_serving_error(BroadServingMalformedNetworkMetrics)
  }
  if !broad_serving_resource_metrics_valid(sample, initial_kv_pages_free) {
    raise broad_serving_error(BroadServingMalformedResourceMetrics)
  }
}

///|
fn require_broad_serving_final_metrics(
  sample : BroadServingMetricSample,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
  pooled? : Bool = false,
) -> Unit raise RuntimeSpawnedBroadServingValidationError {
  if !broad_serving_final_request_metrics_valid(
      sample,
      contract.input_tokens,
      pooled~,
    ) {
    raise broad_serving_error(BroadServingMalformedReuseRequestMetrics)
  }
  if !broad_serving_final_network_metrics_valid(sample, pooled~) {
    raise broad_serving_error(BroadServingMalformedReuseNetworkMetrics)
  }
  if !broad_serving_resource_metrics_valid(sample, initial_kv_pages_free) {
    raise broad_serving_error(BroadServingMalformedReuseResourceMetrics)
  }
}

///|
async fn await_broad_serving_malformed_retirement(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
) -> Unit {
  let pooled = owner.broad_serving_uses_framed_pool()
  for turn in 0..<contract.turn_limit {
    ignore(owner.progress())
    if owner.phase() != Ready ||
      owner.health() != RuntimeHealthy ||
      !owner.spawned_serving_ingress_live() {
      raise broad_serving_error(BroadServingMalformedUnexpectedOwnerFailure)
    }
    if owner.spawned_serving_ingress_listening() {
      require_broad_serving_malformed_metrics(
        capture_broad_serving_metrics(owner),
        contract,
        initial_kv_pages_free,
        pooled~,
      )
      return
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(BroadServingMalformedTransition)
}

///|
async fn run_broad_serving_post_malformed_recovery(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
) -> RuntimeSpawnedBroadServingValidationResult {
  let pooled = owner.broad_serving_uses_framed_pool()
  let expected_accepts = if pooled { 7UL } else { 5UL }
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
      BroadServingMalformedReuseAccepted,
    ),
    contract,
    2,
    BroadServingMalformedReuseAccepted,
  )
  require_broad_token(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingMalformedReuseToken0,
    ),
    contract.expected_first_token,
    0,
    BroadServingMalformedReuseToken0,
  )
  require_broad_token(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingMalformedReuseToken1,
    ),
    contract.expected_second_token,
    1,
    BroadServingMalformedReuseToken1,
  )
  require_broad_usage(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingMalformedReuseUsage,
    ),
    contract.input_tokens,
    2,
    BroadServingMalformedReuseUsage,
  )
  require_broad_completed(
    read_broad_serving_event(
      owner,
      client,
      storage,
      decoder,
      contract.io_timeout_millis,
      BroadServingMalformedReuseCompleted,
    ),
    contract.input_tokens,
    2,
    TokenLimit,
    BroadServingMalformedReuseCompleted,
  )
  for turn in 0..<contract.turn_limit {
    ignore(owner.progress())
    let sample = capture_broad_serving_metrics(owner)
    if sample.admissions == 5UL &&
      sample.completions == 4UL &&
      sample.cancellations == 1UL &&
      sample.queue_depth == 0 &&
      sample.active_requests == 0 &&
      sample.kv_pages_used == 0 &&
      sample.kv_pages_free == initial_kv_pages_free {
      client.close()
      client_open = false
      retire_broad_serving_client(
        owner,
        contract.turn_limit,
        expected_accepts,
        expected_accepts,
        2UL,
      ) catch {
        RuntimeSpawnedBroadServingValidationFailed(
          BroadServingHealth
          | BroadServingReadiness
        ) =>
          raise broad_serving_error(BroadServingMalformedUnexpectedOwnerFailure)
        _ => raise broad_serving_error(BroadServingMalformedReuseNetworkMetrics)
      }
      let final_sample = capture_broad_serving_metrics(owner)
      require_broad_serving_final_metrics(
        final_sample,
        contract,
        initial_kv_pages_free,
        pooled~,
      )
      return {
        ingress_owner: if pooled {
          BroadServingFramedConnectionPool
        } else {
          BroadServingSingletonServer
        },
        request_count: final_sample.admissions,
        completion_count: final_sample.completions,
        cancellation_count: final_sample.cancellations,
        event_count: if pooled {
          20
        } else {
          22
        },
        backpressure_count: final_sample.backpressure,
        network_accepts: final_sample.network_accepts,
        network_disconnects: final_sample.network_disconnects,
        network_rejections: final_sample.network_rejections,
        kv_pages_free: final_sample.kv_pages_free,
      }
    }
    if owner.phase() != Ready ||
      owner.health() != RuntimeHealthy ||
      !owner.spawned_serving_terminal_ingress_live() {
      raise broad_serving_error(BroadServingMalformedUnexpectedOwnerFailure)
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(BroadServingMalformedReuseRequestMetrics)
}

///|
async fn run_broad_serving_malformed_fail_closed(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBroadServingValidationContract,
  initial_kv_pages_free : Int,
) -> RuntimeSpawnedBroadServingValidationResult {
  let malformed = broad_serving_copy_frame(contract.first_frame)
  malformed[0] = malformed[0] ^ b'\x01'
  let client = connect_broad_serving_client(owner, contract.io_timeout_millis)
  let mut client_open = true
  defer (if client_open { client.close() })
  accept_broad_serving_client(owner, contract.turn_limit)
  write_broad_serving_frame(client, malformed, contract.io_timeout_millis)
  await_broad_serving_malformed_retirement(
    owner, contract, initial_kv_pages_free,
  )
  client.close()
  client_open = false
  run_broad_serving_post_malformed_recovery(
    owner, contract, initial_kv_pages_free,
  )
}
