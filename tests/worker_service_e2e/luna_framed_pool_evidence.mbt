///|
fn prepare_framed_pool_e2e(
  executable : Bytes,
  tcp : @online_tcp.LunaOnlineTcpLimits,
  fixture_root : String,
) -> @online_tcp.LunaOnlineFramedConnectionPoolPreparation raise {
  @online_tcp.prepare_luna_online_framed_connection_pool(
    prepare_tcp_service(executable, session_tokenizer(), fixture_root~),
    tcp,
    @online_tcp.LunaOnlineFramedConnectionPoolLimits::new(
      connection_capacity=2,
      io_poll_timeout_millis=1,
      request_work_units=17,
      event_work_units=17,
    ),
    @inference.DeadlineBudget::from_milliseconds(
      10000UL,
      tcp_inference_limits(),
    ),
  )
}

///|
async fn progress_framed_pool_to_connections(
  pool : @online_tcp.LunaOnlineFramedConnectionPool,
  expected : Int,
  stage : String,
) -> Unit {
  for turn in 0..<5000 {
    if pool.active_connection_count() == expected {
      return
    }
    match pool.progress_on_reactor() {
      LunaOnlineFramedConnectionPoolFailedProgress =>
        raise Failure("framed pool failed during \{stage}")
      _ => ()
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise Failure(
    "framed pool connection count did not converge during \{stage} active=\{pool.active_connection_count()} accepts=\{pool.network_accepts()} disconnects=\{pool.network_disconnects()} events=\{pool.delivered_events()}",
  )
}

///|
fn require_framed_pool_sequence(
  first : @inference.StreamEvent,
  second : @inference.StreamEvent,
  third : @inference.StreamEvent,
  fourth : @inference.StreamEvent,
) -> Unit raise {
  guard first is Accepted(accepted) else {
    raise Failure("framed pool sequence did not start with Accepted")
  }
  assert_eq(accepted.model_identity(), identity())
  guard second is Token(token) else {
    raise Failure("framed pool sequence did not publish Token second")
  }
  assert_eq(token.position(), 0)
  guard third is Usage(usage) else {
    raise Failure("framed pool sequence did not publish Usage third")
  }
  assert_eq(usage.input_tokens(), 2)
  assert_eq(usage.output_tokens(), 1)
  guard fourth is Completed(completed) else {
    raise Failure("framed pool sequence did not finish with Completed")
  }
  assert_eq(completed.finish_reason(), TokenLimit)
  assert_eq(completed.final_usage().input_tokens(), 2)
  assert_eq(completed.final_usage().output_tokens(), 1)
}

///|
async fn drain_framed_pool(
  pool : @online_tcp.LunaOnlineFramedConnectionPool,
) -> Unit {
  pool.begin_drain()
  assert_eq(pool.readiness(), LunaOnlineFramedConnectionPoolNotReady)
  for turn in 0..<5000 {
    if pool.state() == LunaOnlineFramedConnectionPoolClosed {
      return
    }
    ignore(pool.progress_on_reactor())
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise Failure("framed pool drain did not close every lower owner")
}

///|
/// Two sockets remain live together. The slow socket contributes only one
/// byte, while the complete fast request crosses the one shared service and
/// retires. Only then is the slow frame completed and admitted.
async fn run_luna_framed_pool_evidence(
  executable : Bytes,
  fixture_root : String,
) -> Unit {
  let tcp = @online_tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes=17,
    write_chunk_bytes=17,
    accept_timeout_millis=5000,
    input_idle_timeout_millis=5000,
    write_timeout_millis=5000,
    reactor_transition_budget=1,
  )
  let pool = @online_tcp.bind_luna_online_framed_connection_pool(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_framed_pool_e2e(executable, tcp, fixture_root),
    tcp,
    log_capacity=64,
  )
  assert_eq(pool.readiness(), LunaOnlineFramedConnectionPoolReady)
  let slow_frame = tcp_request_frame(1200UL)
  let fast_frame = tcp_request_frame(1201UL)
  let slow = @socket.Tcp::connect(pool.local_addr())
  progress_framed_pool_to_connections(pool, 1, "slow accept")
  tcp_write_fixed(slow, slow_frame, source_offset=0, length=1)
  let fast = @socket.Tcp::connect(pool.local_addr())
  progress_framed_pool_to_connections(pool, 2, "fast accept")
  tcp_write_fixed(fast, fast_frame, source_offset=0, length=fast_frame.length())
  progress_framed_pool_to_connections(pool, 1, "fast retirement")
  if pool.delivered_events() != 4UL {
    raise Failure(
      "fast route retired without four delivered events: \{pool.delivered_events()}",
    )
  }
  require_framed_pool_sequence(
    tcp_read_event(fast),
    tcp_read_event(fast),
    tcp_read_event(fast),
    tcp_read_event(fast),
  )
  fast.close()

  tcp_write_fixed(
    slow,
    slow_frame,
    source_offset=1,
    length=slow_frame.length() - 1,
  )
  progress_framed_pool_to_connections(pool, 0, "slow retirement")
  require_framed_pool_sequence(
    tcp_read_event(slow),
    tcp_read_event(slow),
    tcp_read_event(slow),
    tcp_read_event(slow),
  )
  slow.close()
  assert_eq(pool.network_accepts(), 2UL)
  assert_eq(pool.network_disconnects(), 2UL)
  assert_eq(pool.delivered_events(), 8UL)
  assert_eq(pool.state(), LunaOnlineFramedConnectionPoolListening)
  let metrics = pool.metrics_snapshot()
  assert_eq(metrics.counter_value(Admissions), 2UL)
  assert_eq(metrics.counter_value(Completions), 2UL)
  assert_eq(metrics.counter_value(NetworkAccepts), 2UL)
  assert_eq(metrics.counter_value(NetworkDisconnects), 2UL)
  assert_eq(metrics.gauge_value(QueueDepth), 0)
  assert_eq(metrics.gauge_value(ActiveRequests), 0)
  drain_framed_pool(pool)
  assert_eq(pool.state(), LunaOnlineFramedConnectionPoolClosed)
}
