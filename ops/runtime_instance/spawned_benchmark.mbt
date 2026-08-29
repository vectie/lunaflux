///|
priv enum SpawnedBenchmarkOutcome {
  SpawnedBenchmarkSucceeded(RuntimeSpawnedBenchmarkResult)
  SpawnedBenchmarkRejected(RuntimeSpawnedBenchmarkRule)
}

///|
fn observe_spawned_benchmark_token(
  event : @inference.StreamEvent,
  expected_position : Int,
  maximum_token_id : Int,
) -> Int raise RuntimeSpawnedBenchmarkError {
  match event {
    Token(token) if token.position() == expected_position &&
      token.token_id() >= 0 &&
      token.token_id() <= maximum_token_id => token.token_id()
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
}

///|
async fn run_spawned_benchmark(
  owner : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBenchmarkContract,
) -> RuntimeSpawnedBenchmarkResult {
  let serving = contract.serving
  bind_spawned_serving_listener(owner) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  let startup = owner.startup_measurement() catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  if startup.readiness() != RuntimeReady {
    raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  let initial = capture_spawned_serving_metrics(owner) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  if !spawned_serving_initial_metrics_valid(initial) {
    raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  let collector = @benchmark_runner.BenchmarkTrialCollector::new(
    engine=LunaFlux,
    profile=contract.profile,
    ordinal=contract.ordinal,
    order_position=contract.order_position,
    request_capacity=1,
  ) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkCollector)
  }
  let client = connect_spawned_serving_client(owner, serving.io_timeout_millis) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  let mut client_open = true
  defer (if client_open { client.close() })
  accept_spawned_serving_client(owner, serving.turn_limit) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  collector.submit(
    request_ordinal=0,
    input_tokens=Int64::from_int(serving.input_tokens),
  ) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkCollector)
  }
  write_spawned_serving_frame(client, serving.frame, serving.io_timeout_millis) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  let decoder = @framed_wire.EventFrameBuffer::new(serving.framed_limits)
  let storage = FixedArray::make(decoder.capacity(), b'\x00')
  require_spawned_serving_accepted(
    read_spawned_serving_event(
      owner,
      client,
      storage,
      decoder,
      serving.io_timeout_millis,
    ) catch {
      _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
    },
    serving,
  ) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  collector.admit(request_ordinal=0) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkCollector)
  }
  let first_event = read_spawned_serving_event(
    owner,
    client,
    storage,
    decoder,
    serving.io_timeout_millis,
  ) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  let first = if contract.verify_expected_tokens {
    require_spawned_serving_token(
      first_event,
      serving.expected_first_token,
      0,
      SpawnedServingFirstToken,
    ) catch {
      _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
    }
  } else {
    observe_spawned_benchmark_token(
      first_event,
      0,
      serving.framed_limits.inference_limits().max_token_id(),
    ) catch {
      _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
    }
  }
  collector.first_token(request_ordinal=0) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkCollector)
  }
  let second_event = read_spawned_serving_event(
    owner,
    client,
    storage,
    decoder,
    serving.io_timeout_millis,
  ) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  let second = if contract.verify_expected_tokens {
    require_spawned_serving_token(
      second_event,
      serving.expected_second_token,
      1,
      SpawnedServingSecondToken,
    ) catch {
      _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
    }
  } else {
    observe_spawned_benchmark_token(
      second_event,
      1,
      serving.framed_limits.inference_limits().max_token_id(),
    ) catch {
      _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
    }
  }
  require_spawned_serving_usage(
    read_spawned_serving_event(
      owner,
      client,
      storage,
      decoder,
      serving.io_timeout_millis,
    ) catch {
      _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
    },
    serving.input_tokens,
  ) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  require_spawned_serving_completed(
    read_spawned_serving_event(
      owner,
      client,
      storage,
      decoder,
      serving.io_timeout_millis,
    ) catch {
      _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
    },
    serving.input_tokens,
  ) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  collector.terminate(request_ordinal=0, outcome=Completed, generated_tokens=2L) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkCollector)
  }
  ignore(
    progress_spawned_serving_request(owner, serving, initial.kv_pages_free) catch {
      _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
    },
  )
  client.close()
  client_open = false
  let retired = retire_spawned_serving_client(
    owner,
    serving,
    initial.kv_pages_free,
  ) catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkServing)
  }
  let measurements = collector.finish() catch {
    _ => raise spawned_benchmark_error(SpawnedBenchmarkCollector)
  }
  {
    measurements,
    startup,
    serving: {
      first_token: first,
      second_token: second,
      event_count: 5,
      kv_pages_free: retired.kv_pages_free,
      network_accepts: retired.network_accepts,
      network_disconnects: retired.network_disconnects,
    },
  }
}

///|
/// Measures one authenticated request through the actual spawned native TCP
/// listener. Lifecycle timestamps are sampled by the bounded benchmark
/// collector at the external protocol observations; no synthetic duration is
/// accepted. Evidence is returned only after listener, service, child,
/// scheduler, and KV ownership close cleanly.
pub async fn RuntimeInstanceOwner::benchmark_spawned_serving_request(
  self : RuntimeInstanceOwner,
  contract : RuntimeSpawnedBenchmarkContract,
) -> RuntimeSpawnedBenchmarkResult {
  @async.protect_from_cancel(() => {
    let outcome = try run_spawned_benchmark(self, contract) catch {
      RuntimeSpawnedBenchmarkFailed(rule) => SpawnedBenchmarkRejected(rule)
      _ => SpawnedBenchmarkRejected(SpawnedBenchmarkServing)
    } noraise {
      result => SpawnedBenchmarkSucceeded(result)
    }
    let cleaned = drain_spawned_serving_owner(self, contract.serving.turn_limit)
    if !cleaned {
      raise spawned_benchmark_error(SpawnedBenchmarkCleanup)
    }
    match outcome {
      SpawnedBenchmarkSucceeded(result) => {
        if !spawned_serving_closed_state_valid(
            self.phase(),
            self.health(),
            self.readiness(),
            self.cleanup_complete(),
          ) {
          raise spawned_benchmark_error(SpawnedBenchmarkCleanup)
        }
        result
      }
      SpawnedBenchmarkRejected(rule) => raise spawned_benchmark_error(rule)
    }
  })
}
