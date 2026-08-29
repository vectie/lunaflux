///|
fn wb_server_graph_report(
  sequence : UInt64,
  hits : UInt64,
  misses : UInt64,
) -> @worker_wire.WorkerGraphTelemetry raise {
  let path = if hits > 0UL {
    @worker_wire.WorkerGraphCaptured
  } else {
    WorkerGraphOrderedEager
  }
  @worker_wire.WorkerGraphTelemetry::without_shape(
    sequence~,
    path~,
    graph_hits=hits,
    graph_misses=misses,
  )
}

///|
test "server metrics consume graph deltas once and reject regression" {
  let bridge = try! LunaServerTelemetryBridge::new(4)
  assert_true(
    bridge.refresh_graph_runtime(wb_server_graph_report(1UL, 1UL, 0UL)),
  )
  assert_true(
    bridge.refresh_graph_runtime(wb_server_graph_report(2UL, 3UL, 0UL)),
  )
  assert_true(
    bridge.refresh_graph_runtime(wb_server_graph_report(2UL, 3UL, 0UL)),
  )
  assert_eq(bridge.metrics.counter_value(GraphHits), 3UL)
  assert_eq(bridge.metrics.counter_value(GraphMisses), 0UL)
  assert_false(
    bridge.refresh_graph_runtime(wb_server_graph_report(3UL, 2UL, 0UL)),
  )
  assert_true(bridge.degraded)
}
