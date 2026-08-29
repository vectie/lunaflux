///|
fn graph_telemetry_shape(
  batch_rows : Int,
  sequence_tokens : Int,
) -> @luna_capability_manifest.LunaGraphShapeClass raise {
  @luna_capability_manifest.LunaGraphShapeClass::from_execution_shape(
    @device_profile.ExactExecutionShape::full_recompute(
      batch_rows~,
      sequence_tokens~,
    ),
  )
}

///|
test "actual captured mode records only graph hits with authenticated shape" {
  let telemetry = new_paged_graph_telemetry(
    OrderedKernelCaptured,
    Some(graph_telemetry_shape(3, 5)),
  )
  let startup = telemetry.snapshot()
  assert_eq(startup.selected_path(), PagedGraphCaptured)
  assert_eq(startup.graph_hits(), 0UL)
  assert_eq(startup.graph_misses(), 0UL)
  guard startup.selected_shape_class() is Some(shape) else {
    fail("authenticated selected shape was dropped")
  }
  assert_eq(shape.phase(), FullRecompute)
  assert_eq(shape.batch_rows(), 3)
  assert_eq(shape.query_tokens_per_row(), 5)
  assert_eq(shape.context_tokens(), 5)
  assert_eq(shape.token_rows(), 15L)
  telemetry.record_completed()
  telemetry.record_completed()
  let completed = telemetry.snapshot()
  assert_eq(completed.graph_hits(), 2UL)
  assert_eq(completed.graph_misses(), 0UL)
  // A previously returned scalar report is immutable and carries no owner.
  assert_eq(startup.graph_hits(), 0UL)
}

///|
test "actual eager mode records misses and never fabricates shape evidence" {
  let telemetry = new_paged_graph_telemetry(OrderedKernelEager, None)
  telemetry.record_completed()
  let report = telemetry.snapshot()
  assert_eq(report.selected_path(), PagedGraphOrderedEager)
  assert_true(report.selected_shape_class() is None)
  assert_eq(report.graph_hits(), 0UL)
  assert_eq(report.graph_misses(), 1UL)
}

///|
test "graph runtime counters saturate without wrapping or changing selection" {
  let hit = new_paged_graph_telemetry(
    OrderedKernelCaptured,
    Some(graph_telemetry_shape(1, 1)),
  )
  hit.graph_hits = 0xffffffffffffffffUL
  hit.record_completed()
  assert_eq(hit.snapshot().graph_hits(), 0xffffffffffffffffUL)
  assert_eq(hit.snapshot().selected_path(), PagedGraphCaptured)
  let miss = new_paged_graph_telemetry(OrderedKernelEager, None)
  miss.graph_misses = 0xffffffffffffffffUL
  miss.record_completed()
  assert_eq(miss.snapshot().graph_misses(), 0xffffffffffffffffUL)
  assert_eq(miss.snapshot().selected_path(), PagedGraphOrderedEager)
}

///|
fn paged_graph_telemetry_allocation_positive_control() -> FixedArray[Byte] {
  FixedArray::make(23, b'G')
}

///|
test "graph telemetry allocation gate positive control is linked" {
  assert_eq(paged_graph_telemetry_allocation_positive_control().length(), 23)
}
