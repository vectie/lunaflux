///|
fn graph_test_report(
  sequence : UInt64,
  misses : UInt64,
) -> @worker_wire.WorkerGraphTelemetry raise {
  @worker_wire.WorkerGraphTelemetry::without_shape(
    sequence~,
    path=WorkerGraphOrderedEager,
    graph_hits=0UL,
    graph_misses=misses,
  )
}

///|
fn graph_test_shape(
  batch_rows : Int,
) -> @worker_wire.WorkerGraphShapeClass raise {
  @worker_wire.WorkerGraphShapeClass::new(
    phase=1,
    batch_rows~,
    query_tokens_per_row=1,
    context_tokens=8,
    token_rows=batch_rows.to_int64(),
  )
}

///|
test "all equal rank reports select one logical eager miss, not rank sum" {
  let selected = @worker_wire.WorkerGraphTelemetrySlot::new()
  let previous = @worker_wire.WorkerGraphTelemetrySlot::new()
  let report = graph_test_report(1UL, 1UL)
  assert_true(admit_rank_graph_report(report, 1UL, selected, previous))
  assert_true(admit_rank_graph_report(report, 1UL, selected, previous))
  let logical = selected.current()
  assert_eq(logical.sequence_value(), 1UL)
  assert_eq(logical.selected_path(), WorkerGraphOrderedEager)
  assert_false(logical.has_selected_shape_class())
  assert_eq(logical.graph_hits(), 0UL)
  assert_eq(logical.graph_misses(), 1UL)
}

///|
test "rank path, shape, counter, and sequence substitution is rejected" {
  let selected = @worker_wire.WorkerGraphTelemetrySlot::new()
  let previous = @worker_wire.WorkerGraphTelemetrySlot::new()
  let first = @worker_wire.WorkerGraphTelemetry::with_shape(
    sequence=1UL,
    path=WorkerGraphOrderedEager,
    shape=graph_test_shape(1),
    graph_hits=0UL,
    graph_misses=1UL,
  )
  assert_true(admit_rank_graph_report(first, 1UL, selected, previous))
  let shape_substitution = @worker_wire.WorkerGraphTelemetry::with_shape(
    sequence=1UL,
    path=WorkerGraphOrderedEager,
    shape=graph_test_shape(2),
    graph_hits=0UL,
    graph_misses=1UL,
  )
  assert_false(
    admit_rank_graph_report(shape_substitution, 1UL, selected, previous),
  )
  let path_substitution = @worker_wire.WorkerGraphTelemetry::without_shape(
    sequence=1UL,
    path=WorkerGraphCaptured,
    graph_hits=1UL,
    graph_misses=0UL,
  )
  assert_false(
    admit_rank_graph_report(path_substitution, 1UL, selected, previous),
  )
  assert_false(
    admit_rank_graph_report(
      graph_test_report(1UL, 2UL),
      1UL,
      selected,
      previous,
    ),
  )
  assert_false(
    admit_rank_graph_report(
      graph_test_report(2UL, 1UL),
      1UL,
      selected,
      previous,
    ),
  )
}

///|
test "partial and corrupt graph sidecars fail fixed-wire decoding" {
  let encoded = FixedArray::make(
    @worker_wire.worker_graph_telemetry_frame_capacity(),
    b'\x00',
  )
  let length = @worker_wire.encode_worker_graph_telemetry(
    graph_test_report(1UL, 1UL),
    encoded,
  )
  try @worker_wire.decode_worker_graph_telemetry(encoded, length - 1) catch {
    InvalidFrame(_) => ()
    _ => fail("unexpected partial TP graph sidecar failure")
  } noraise {
    _ => fail("partial TP graph sidecar was accepted")
  }
  encoded[8] = encoded[8] ^ b'\x01'
  try @worker_wire.decode_worker_graph_telemetry(encoded, length) catch {
    InvalidFrame(_) => ()
    _ => fail("unexpected corrupt TP graph sidecar failure")
  } noraise {
    _ => fail("corrupt TP graph sidecar was accepted")
  }
}

///|
test "group graph counters require one monotonic step and allow saturation" {
  let previous = graph_test_report(7UL, 7UL)
  assert_true(valid_next_graph_report(graph_test_report(8UL, 8UL), previous))
  assert_false(valid_next_graph_report(graph_test_report(8UL, 7UL), previous))
  assert_false(valid_next_graph_report(graph_test_report(8UL, 9UL), previous))
  let saturated = graph_test_report(9UL, 0xffffffffffffffffUL)
  assert_true(valid_next_graph_report(saturated, saturated))
}
