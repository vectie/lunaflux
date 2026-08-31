///|
fn wb_graph_report(
  sequence : UInt64,
  hits : UInt64,
  misses : UInt64,
  batch_rows? : Int = 1,
) -> @worker_wire.WorkerGraphTelemetry raise {
  let shape = @worker_wire.WorkerGraphShapeClass::new(
    phase=1,
    batch_rows~,
    query_tokens_per_row=1,
    context_tokens=1,
    token_rows=Int64::from_int(batch_rows),
  )
  let path = if hits > 0UL {
    @worker_wire.WorkerGraphCaptured
  } else {
    WorkerGraphOrderedEager
  }
  @worker_wire.WorkerGraphTelemetry::with_shape(
    sequence~,
    path~,
    shape~,
    graph_hits=hits,
    graph_misses=misses,
  )
}

///|
test "service graph telemetry aggregates authenticated replacement counters" {
  let owner = new_worker_service_graph_telemetry()
  owner.record_child_report(1UL, wb_graph_report(1UL, 1UL, 0UL))
  let first = owner.aggregate.current()
  assert_eq(first.graph_hits(), 1UL)
  owner.record_child_report(2UL, wb_graph_report(2UL, 2UL, 0UL))
  owner.begin_replacement_child()
  owner.record_child_report(3UL, wb_graph_report(3UL, 1UL, 0UL))
  let replaced = owner.aggregate.current()
  assert_eq(replaced.sequence_value(), 3UL)
  assert_eq(replaced.graph_hits(), 3UL)
  assert_eq(replaced.graph_misses(), 0UL)
  assert_eq(replaced.selected_path(), WorkerGraphCaptured)
  assert_eq(replaced.selected_shape_class().batch_rows(), 1)
}

///|
test "service graph telemetry admits a cumulative sparse sample" {
  let owner = new_worker_service_graph_telemetry()
  owner.record_child_report(4UL, wb_graph_report(4UL, 4UL, 0UL))
  owner.record_child_report(64UL, wb_graph_report(64UL, 64UL, 0UL))
  let aggregate = owner.aggregate.current()
  assert_eq(aggregate.sequence_value(), 64UL)
  assert_eq(aggregate.graph_hits(), 64UL)
  assert_eq(aggregate.graph_misses(), 0UL)
}

///|
test "service graph telemetry rejects regressions and replacement substitution" {
  let regressed = new_worker_service_graph_telemetry()
  regressed.record_child_report(1UL, wb_graph_report(1UL, 1UL, 0UL))
  try regressed.record_child_report(2UL, wb_graph_report(2UL, 1UL, 0UL)) catch {
    Invalid(WorkerOutput) => ()
    _ => fail("unexpected graph counter regression error")
  } noraise {
    _ => fail("non-advancing graph counters were published")
  }

  let substituted = new_worker_service_graph_telemetry()
  substituted.record_child_report(1UL, wb_graph_report(1UL, 1UL, 0UL))
  substituted.begin_replacement_child()
  try
    substituted.record_child_report(
      2UL,
      wb_graph_report(2UL, 1UL, 0UL, batch_rows=2),
    )
  catch {
    Invalid(WorkerOutput) => ()
    _ => fail("unexpected graph shape substitution error")
  } noraise {
    _ => fail("replacement changed the selected graph shape")
  }
  let stable = substituted.aggregate.current()
  assert_eq(stable.sequence_value(), 1UL)
  assert_eq(stable.selected_shape_class().batch_rows(), 1)
}

///|
test "service graph telemetry accepts only a genuinely saturated child step" {
  let owner = new_worker_service_graph_telemetry()
  owner.record_child_report(1UL, wb_graph_report(1UL, 1UL, 0UL))
  owner.aggregate.store(wb_graph_report(1UL, 0xffffffffffffffffUL, 0UL))
  owner.child_hits = 0xffffffffffffffffUL
  owner.child_misses = 0UL
  owner.child_initialized = true
  owner.record_child_report(
    2UL,
    wb_graph_report(2UL, 0xffffffffffffffffUL, 0UL),
  )
  let saturated = owner.aggregate.current()
  assert_eq(saturated.sequence_value(), 2UL)
  assert_eq(saturated.graph_hits(), 0xffffffffffffffffUL)
  assert_eq(saturated.graph_misses(), 0UL)

  let false_saturation = new_worker_service_graph_telemetry()
  false_saturation.record_child_report(1UL, wb_graph_report(1UL, 1UL, 0UL))
  try
    false_saturation.record_child_report(2UL, wb_graph_report(2UL, 1UL, 0UL))
  catch {
    Invalid(WorkerOutput) => ()
    _ => fail("unexpected false graph saturation error")
  } noraise {
    _ => fail("non-maximum unchanged graph counter was accepted")
  }
}
