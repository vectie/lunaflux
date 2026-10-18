///|
fn encoded_initial_graph_attack(
  path : @worker_wire.WorkerGraphExecutionPath,
  hits : UInt64,
  misses : UInt64,
  destination : FixedArray[Byte],
) -> Int raise {
  @worker_wire.encode_worker_graph_telemetry(
    @worker_wire.WorkerGraphTelemetry::without_shape(
      sequence=1UL,
      path~,
      graph_hits=hits,
      graph_misses=misses,
    ),
    destination,
  )
}

///|
fn expect_initial_graph_attack_rejected(
  path : @worker_wire.WorkerGraphExecutionPath,
  hits : UInt64,
  misses : UInt64,
) -> Unit raise {
  let owner = test_supervisor()
  let plan = supervisor_plan(1UL)
  ignore(owner.submit(plan))
  let completion_length = encoded_completion(plan, owner.rx)
  owner.accept_completion_bytes_in_place(0, plan, completion_length)
  let graph_length = encoded_initial_graph_attack(
    path,
    hits,
    misses,
    owner.graph_rx,
  )
  try owner.accept_graph_telemetry_bytes_in_place(0, graph_length) catch {
    Invalid(Completion) => ()
    _ => fail("unexpected hostile initial graph counter error")
  } noraise {
    _ => fail("hostile initial graph counters were published")
  }
  assert_eq(owner.lifecycle(), Failed)
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "completion is unpublished until exact graph sidecar authenticates" {
  let owner = test_supervisor()
  let plan = supervisor_plan(1UL)
  ignore(owner.submit(plan))
  let completion_length = encoded_completion(plan, owner.rx)
  owner.accept_completion_bytes_in_place(0, plan, completion_length)
  try owner.received_for_sequence(1UL) catch {
    Invalid(Completion) => ()
    _ => fail("unexpected sidecar publication guard error")
  } noraise {
    _ => fail("completion was published before graph telemetry")
  }
  let graph_length = encoded_graph_telemetry(2UL, 1UL, owner.graph_rx)
  try owner.accept_graph_telemetry_bytes_in_place(0, graph_length) catch {
    Invalid(Completion) => ()
    _ => fail("unexpected foreign graph sequence error")
  } noraise {
    _ => fail("foreign graph sequence was attached to completion")
  }
  assert_eq(owner.lifecycle(), Failed)
  owner.close()
}

///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "initial graph counters reject overflow and selected-path mismatch" {
  expect_initial_graph_attack_rejected(
    WorkerGraphCaptured,
    0xffffffffffffffffUL,
    2UL,
  )
  expect_initial_graph_attack_rejected(WorkerGraphOrderedEager, 1UL, 0UL)
}

///|
test "graph reports follow eager prefill and captured decode across shapes" {
  let prefill = @worker_wire.WorkerGraphTelemetry::without_shape(
    sequence=1UL,
    path=WorkerGraphOrderedEager,
    graph_hits=0UL,
    graph_misses=1UL,
  )
  let decode = @worker_wire.WorkerGraphTelemetry::with_shape(
    sequence=2UL,
    path=WorkerGraphCaptured,
    shape=@worker_wire.WorkerGraphShapeClass::new(
      phase=1,
      batch_rows=1,
      query_tokens_per_row=1,
      context_tokens=32,
      token_rows=1L,
    ),
    graph_hits=1UL,
    graph_misses=1UL,
  )
  let mixed = @worker_wire.WorkerGraphTelemetry::without_shape(
    sequence=64UL,
    path=WorkerGraphOrderedEager,
    graph_hits=58UL,
    graph_misses=6UL,
  )
  assert_true(valid_initial_graph_counters(prefill))
  assert_true(valid_graph_counter_step(decode, prefill))
  assert_true(valid_graph_counter_step(mixed, decode))
  assert_false(valid_graph_counter_step(decode, mixed))
  assert_false(valid_graph_counter_step(mixed, mixed))
}
