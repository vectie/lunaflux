///|
fn graph_wire_report(
  path : @worker_wire.WorkerGraphExecutionPath,
  hits : UInt64,
  misses : UInt64,
) -> @worker_wire.WorkerGraphTelemetry raise {
  @worker_wire.WorkerGraphTelemetry::with_shape(
    sequence=9UL,
    path~,
    shape=@worker_wire.WorkerGraphShapeClass::new(
      phase=1,
      batch_rows=3,
      query_tokens_per_row=5,
      context_tokens=5,
      token_rows=15L,
    ),
    graph_hits=hits,
    graph_misses=misses,
  )
}

///|
test "graph telemetry sidecar round trips exact scalar report" {
  let output = FixedArray::make(
    @worker_wire.worker_graph_telemetry_frame_capacity(),
    b'\x00',
  )
  let length = @worker_wire.encode_worker_graph_telemetry(
    graph_wire_report(WorkerGraphCaptured, 7UL, 2UL),
    output,
  )
  let decoded = @worker_wire.decode_worker_graph_telemetry(output, length)
  assert_eq(decoded.sequence_value(), 9UL)
  assert_eq(decoded.selected_path(), WorkerGraphCaptured)
  assert_eq(decoded.graph_hits(), 7UL)
  assert_eq(decoded.graph_misses(), 2UL)
  assert_true(decoded.has_selected_shape_class())
  let shape = decoded.selected_shape_class()
  assert_eq(shape.phase(), 1)
  assert_eq(shape.batch_rows(), 3)
  assert_eq(shape.query_tokens_per_row(), 5)
  assert_eq(shape.context_tokens(), 5)
  assert_eq(shape.token_rows(), 15L)
}

///|
test "graph telemetry sidecar rejects corruption and invalid shape" {
  let output = FixedArray::make(
    @worker_wire.worker_graph_telemetry_frame_capacity(),
    b'\x00',
  )
  let length = @worker_wire.encode_worker_graph_telemetry(
    graph_wire_report(WorkerGraphOrderedEager, 0UL, 9UL),
    output,
  )
  output[40] = output[40] ^ b'\x01'
  try @worker_wire.decode_worker_graph_telemetry(output, length) catch {
    InvalidFrame(Checksum) => ()
    _ => fail("unexpected graph telemetry corruption error")
  } noraise {
    _ => fail("corrupt graph telemetry sidecar was admitted")
  }
  try
    @worker_wire.WorkerGraphShapeClass::new(
      phase=1,
      batch_rows=3,
      query_tokens_per_row=5,
      context_tokens=5,
      token_rows=14L,
    )
  catch {
    InvalidFrame(Counts) => ()
    _ => fail("unexpected graph shape rejection")
  } noraise {
    _ => fail("inconsistent graph telemetry shape was admitted")
  }
}

///|
test "graph telemetry shape packing preserves every maximum-width field" {
  let maximum = 0x1fffff
  let token_rows = Int64::from_int(maximum) * Int64::from_int(maximum)
  let report = @worker_wire.WorkerGraphTelemetry::with_shape(
    sequence=1UL,
    path=WorkerGraphCaptured,
    shape=@worker_wire.WorkerGraphShapeClass::new(
      phase=1,
      batch_rows=maximum,
      query_tokens_per_row=maximum,
      context_tokens=maximum,
      token_rows~,
    ),
    graph_hits=1UL,
    graph_misses=0UL,
  )
  let output = FixedArray::make(
    @worker_wire.worker_graph_telemetry_frame_capacity(),
    b'\x00',
  )
  let length = @worker_wire.encode_worker_graph_telemetry(report, output)
  let decoded = @worker_wire.decode_worker_graph_telemetry(output, length)
  let shape = decoded.selected_shape_class()
  assert_eq(shape.phase(), 1)
  assert_eq(shape.batch_rows(), maximum)
  assert_eq(shape.query_tokens_per_row(), maximum)
  assert_eq(shape.context_tokens(), maximum)
  assert_eq(shape.token_rows(), token_rows)
}
