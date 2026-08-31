///|
fn new_worker_service_graph_telemetry() -> WorkerServiceGraphTelemetryOwner {
  {
    aggregate: @worker_wire.WorkerGraphTelemetrySlot::new(),
    child_hits: 0UL,
    child_misses: 0UL,
    child_initialized: false,
  }
}

///|
fn worker_service_graph_shape_equal(
  left : @worker_wire.WorkerGraphTelemetry,
  right : @worker_wire.WorkerGraphTelemetry,
) -> Bool raise WorkerServiceError {
  if left.has_selected_shape_class() != right.has_selected_shape_class() {
    return false
  }
  if !left.has_selected_shape_class() {
    return true
  }
  let left_shape = left.selected_shape_class() catch {
    _ => raise Invalid(WorkerOutput)
  }
  let right_shape = right.selected_shape_class() catch {
    _ => raise Invalid(WorkerOutput)
  }
  left_shape.phase() == right_shape.phase() &&
  left_shape.batch_rows() == right_shape.batch_rows() &&
  left_shape.query_tokens_per_row() == right_shape.query_tokens_per_row() &&
  left_shape.context_tokens() == right_shape.context_tokens() &&
  left_shape.token_rows() == right_shape.token_rows()
}

///|
fn saturated_graph_sum(left : UInt64, right : UInt64) -> UInt64 {
  if 0xffffffffffffffffUL - left < right {
    0xffffffffffffffffUL
  } else {
    left + right
  }
}

///|
fn aggregate_graph_report(
  sequence : UInt64,
  selected : @worker_wire.WorkerGraphTelemetry,
  hits : UInt64,
  misses : UInt64,
) -> @worker_wire.WorkerGraphTelemetry raise WorkerServiceError {
  let path = selected.selected_path()
  if selected.has_selected_shape_class() {
    let shape = selected.selected_shape_class() catch {
      _ => raise Invalid(WorkerOutput)
    }
    @worker_wire.WorkerGraphTelemetry::with_shape(
      sequence~,
      path~,
      shape~,
      graph_hits=hits,
      graph_misses=misses,
    ) catch {
      _ => raise Invalid(WorkerOutput)
    }
  } else {
    @worker_wire.WorkerGraphTelemetry::without_shape(
      sequence~,
      path~,
      graph_hits=hits,
      graph_misses=misses,
    ) catch {
      _ => raise Invalid(WorkerOutput)
    }
  }
}

///|
fn WorkerServiceGraphTelemetryOwner::record_child_report(
  self : WorkerServiceGraphTelemetryOwner,
  sequence : UInt64,
  report : @worker_wire.WorkerGraphTelemetry,
) -> Unit raise WorkerServiceError {
  if report.sequence_value() != sequence {
    raise Invalid(Sequence)
  }
  let hits = report.graph_hits()
  let misses = report.graph_misses()
  let (delta_hits, delta_misses) = if self.child_initialized {
    if hits < self.child_hits || misses < self.child_misses {
      raise Invalid(WorkerOutput)
    }
    (hits - self.child_hits, misses - self.child_misses)
  } else {
    (hits, misses)
  }
  let saturated_step = self.child_initialized &&
    delta_hits == 0UL &&
    delta_misses == 0UL &&
    (match report.selected_path() {
      WorkerGraphCaptured =>
        self.child_hits == 0xffffffffffffffffUL && self.child_misses == misses
      WorkerGraphOrderedEager =>
        self.child_misses == 0xffffffffffffffffUL && self.child_hits == hits
    })
  if !saturated_step &&
    !((delta_hits > 0UL && delta_misses == 0UL) ||
    (delta_hits == 0UL && delta_misses > 0UL)) {
    raise Invalid(WorkerOutput)
  }
  let (aggregate_hits, aggregate_misses) = if self.aggregate.is_available() {
    let previous = self.aggregate.current() catch {
      _ => raise Invalid(WorkerOutput)
    }
    if previous.selected_path() != report.selected_path() ||
      !worker_service_graph_shape_equal(previous, report) {
      raise Invalid(WorkerOutput)
    }
    (
      saturated_graph_sum(previous.graph_hits(), delta_hits),
      saturated_graph_sum(previous.graph_misses(), delta_misses),
    )
  } else {
    (delta_hits, delta_misses)
  }
  self.aggregate.store(
    aggregate_graph_report(sequence, report, aggregate_hits, aggregate_misses),
  )
  self.child_hits = hits
  self.child_misses = misses
  self.child_initialized = true
}

///|
fn WorkerServiceGraphTelemetryOwner::begin_replacement_child(
  self : WorkerServiceGraphTelemetryOwner,
) -> Unit {
  self.child_hits = 0UL
  self.child_misses = 0UL
  self.child_initialized = false
}

///|
pub fn OnlineWorkerLease::luna_graph_runtime_telemetry(
  self : OnlineWorkerLease,
) -> @worker_wire.WorkerGraphTelemetry raise WorkerServiceError {
  self.require_current().graph_telemetry.aggregate.current() catch {
    _ => raise Invalid(WorkerOutput)
  }
}

///|
pub fn OnlineWorkerLease::has_luna_graph_runtime_telemetry(
  self : OnlineWorkerLease,
) -> Bool raise WorkerServiceError {
  self.require_current().graph_telemetry.aggregate.is_available()
}
