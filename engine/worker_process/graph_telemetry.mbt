///|
fn same_graph_shape(
  left : @worker_wire.WorkerGraphTelemetry,
  right : @worker_wire.WorkerGraphTelemetry,
) -> Bool raise WorkerProcessError {
  if left.has_selected_shape_class() != right.has_selected_shape_class() {
    return false
  }
  if !left.has_selected_shape_class() {
    return true
  }
  let left_shape = left.selected_shape_class() catch {
    error => raise Wire(error)
  }
  let right_shape = right.selected_shape_class() catch {
    error => raise Wire(error)
  }
  left_shape.phase() == right_shape.phase() &&
  left_shape.batch_rows() == right_shape.batch_rows() &&
  left_shape.query_tokens_per_row() == right_shape.query_tokens_per_row() &&
  left_shape.context_tokens() == right_shape.context_tokens() &&
  left_shape.token_rows() == right_shape.token_rows()
}

///|
fn valid_graph_counter_step(
  current : @worker_wire.WorkerGraphTelemetry,
  previous : @worker_wire.WorkerGraphTelemetry,
) -> Bool raise WorkerProcessError {
  let hits = current.graph_hits()
  let misses = current.graph_misses()
  match current.selected_path() {
    WorkerGraphCaptured if misses != 0UL || hits == 0UL => return false
    WorkerGraphOrderedEager if hits != 0UL || misses == 0UL => return false
    _ => ()
  }
  if current.selected_path() != previous.selected_path() ||
    !same_graph_shape(current, previous) {
    return false
  }
  let old_hits = previous.graph_hits()
  let old_misses = previous.graph_misses()
  if hits < old_hits || misses < old_misses {
    return false
  }
  match current.selected_path() {
    WorkerGraphCaptured =>
      misses == old_misses &&
      (if old_hits == 0xffffffffffffffffUL {
        hits == old_hits
      } else {
        hits == old_hits + 1UL
      })
    WorkerGraphOrderedEager =>
      hits == old_hits &&
      (if old_misses == 0xffffffffffffffffUL {
        misses == old_misses
      } else {
        misses == old_misses + 1UL
      })
  }
}

///|
fn valid_initial_graph_counters(
  report : @worker_wire.WorkerGraphTelemetry,
) -> Bool {
  match report.selected_path() {
    WorkerGraphCaptured =>
      report.graph_hits() == 1UL && report.graph_misses() == 0UL
    WorkerGraphOrderedEager =>
      report.graph_hits() == 0UL && report.graph_misses() == 1UL
  }
}

///|
fn WorkerProcessSupervisor::accept_graph_telemetry_bytes_in_place(
  self : WorkerProcessSupervisor,
  side : Int,
  length : Int,
) -> Unit raise WorkerProcessError {
  let expected_sequence = if side == 0 {
    self.sequence_a
  } else if side == 1 {
    self.sequence_b
  } else {
    self.lifecycle = Failed
    raise Invalid(Completion)
  }
  let report = @worker_wire.decode_worker_graph_telemetry(self.graph_rx, length) catch {
    error => {
      self.lifecycle = Failed
      raise Wire(error)
    }
  }
  let valid_counters = if self.last_graph_telemetry.is_available() {
    let previous = self.last_graph_telemetry.current() catch {
      error => raise Wire(error)
    }
    valid_graph_counter_step(report, previous)
  } else {
    valid_initial_graph_counters(report)
  }
  if report.sequence_value() != expected_sequence || !valid_counters {
    self.lifecycle = Failed
    raise Invalid(Completion)
  }
  if side == 0 {
    if self.state_a != 1 {
      self.lifecycle = Failed
      raise Invalid(Completion)
    }
    self.graph_telemetry_a.store(report)
    self.state_a = 2
  } else {
    if self.state_b != 1 {
      self.lifecycle = Failed
      raise Invalid(Completion)
    }
    self.graph_telemetry_b.store(report)
    self.state_b = 2
  }
  self.last_graph_telemetry.store(report)
}

///|
pub fn ReceivedWorkerCompletion::graph_telemetry(
  self : ReceivedWorkerCompletion,
  owner : WorkerProcessSupervisor,
) -> @worker_wire.WorkerGraphTelemetry raise WorkerProcessError {
  ignore(self.frame(owner))
  let slot = if self.1 == 0 {
    owner.graph_telemetry_a
  } else {
    owner.graph_telemetry_b
  }
  let report = slot.current() catch { error => raise Wire(error) }
  if report.sequence_value() != self.2 {
    raise Invalid(Completion)
  }
  report
}
