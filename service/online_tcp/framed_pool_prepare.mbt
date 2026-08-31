///|
fn new_luna_online_framed_connection_slot(
  framed : @framed_wire.FramedWireLimits,
  receipt_deadline : @inference.DeadlineBudget,
  pool_limits : LunaOnlineFramedConnectionPoolLimits,
) -> LunaOnlineFramedConnectionSlot raise LunaOnlineTcpError {
  let receipt_workspace = @request_admission.LunaRequestReceiptWorkspace::new(
    @monotonic_clock.MonotonicClock::system(),
    receipt_deadline,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  {
    receipt_workspace,
    request_workspace: @framed_wire.LunaFramedRequestWorkspace::new(
      framed,
      @framed_wire.LunaFramedRequestStepBudget::new(
        pool_limits.request_work_units,
      ) catch {
        _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
      },
    ),
    event_workspace: @framed_wire.LunaFramedEventWorkspace::new(
      framed,
      @framed_wire.LunaFramedEventStepBudget::new(pool_limits.event_work_units) catch {
        _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
      },
    ) catch {
      _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    },
    connections: Array::new(capacity=1),
    receipts: Array::new(capacity=1),
    request_works: Array::new(capacity=1),
    request_views: Array::new(capacity=1),
    event_works: Array::new(capacity=1),
    event_views: Array::new(capacity=1),
    semantic_events: Array::new(capacity=1),
    routes: Array::new(capacity=1),
    input: FixedArray::make(framed.max_frame_bytes(), b'\x00'),
    output: LunaOnlineTcpOutputScratch::new(framed.max_frame_bytes()) catch {
      _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    },
    output_writes: Array::new(capacity=1),
    flights: Array::new(capacity=1),
    phase: LUNA_FRAMED_POOL_FREE,
    generation: 0UL,
    input_cursor: 0,
    input_tail_start: 0,
    input_tail_length: 0,
    feed_cursor: 0,
    event_copy_cursor: 0,
    output_length: 0,
    output_cursor: 0,
    event_terminal: false,
    terminal_delivered: false,
    cancel_requested: false,
  }
}

///|
/// Preallocates one independent framed parser and event cell per connection.
/// `receipt_deadline` is the exact deadline admitted by this listener; request
/// frames with a different deadline fail in the canonical lower admission.
pub fn prepare_luna_online_framed_connection_pool(
  service : @online_session.LunaOnlineFramedServicePreparation,
  tcp_limits : LunaOnlineTcpLimits,
  pool_limits : LunaOnlineFramedConnectionPoolLimits,
  receipt_deadline : @inference.DeadlineBudget,
) -> LunaOnlineFramedConnectionPoolPreparation raise LunaOnlineTcpError {
  let framed = service.framed_limits() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let maximum_wait_millis = service.maximum_transport_wait_millis() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  tcp_limits.require_fits_transport_wait(maximum_wait_millis)
  if pool_limits.io_poll_timeout_millis >
    Int::min(
      tcp_limits.input_idle_timeout_millis,
      tcp_limits.write_timeout_millis,
    ) {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let slots = FixedArray::makei(pool_limits.connection_capacity, _ => {
    new_luna_online_framed_connection_slot(
      framed, receipt_deadline, pool_limits,
    )
  })
  { service, slots, limits: pool_limits, maximum_wait_millis, consumed: false }
}

///|
pub async fn bind_luna_online_framed_connection_pool(
  addr : @socket.Addr,
  preparation : LunaOnlineFramedConnectionPoolPreparation,
  tcp_limits : LunaOnlineTcpLimits,
  log_capacity~ : Int,
) -> LunaOnlineFramedConnectionPool {
  if preparation.consumed {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  tcp_limits.require_fits_transport_wait(preparation.maximum_wait_millis)
  let initial = preparation.service.luna_telemetry() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let initial_plan = preparation.service.luna_plan_telemetry() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let telemetry = LunaServerTelemetryBridge::new(log_capacity)
  if !telemetry.refresh(initial, initial_plan) {
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let listener = bind_luna_online_tcp_listener(addr)
  let service = preparation.service.take_ready() catch {
    _ => {
      listener.close()
      raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    }
  }
  let opened = service.open_semantic_stream()
  if opened.kind() != LunaOnlineFramedOpened {
    listener.close()
    service.begin_drain()
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let stream = service.take_open_stream(opened) catch {
    _ => {
      listener.close()
      service.begin_drain()
      raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    }
  }
  preparation.consumed = true
  telemetry.record_log(LunaInstanceReady, LunaLogNoReason)
  {
    tcp_limits,
    pool_limits: preparation.limits,
    bound_addr: listener.addr,
    listeners: [listener],
    services: [service],
    streams: [stream],
    slots: preparation.slots,
    telemetry,
    active_connections: 0,
    ingress_slot: -1,
    io_cursor: 0,
    accept_turn: true,
    service_waiting: false,
    service_probe_cooldown: 0,
    drain_requested: false,
    failed: false,
    stream_disconnect_started: false,
    service_drain_started: false,
    closed: false,
    network_accepts: 0UL,
    network_disconnects: 0UL,
    delivered_events: 0UL,
  }
}
