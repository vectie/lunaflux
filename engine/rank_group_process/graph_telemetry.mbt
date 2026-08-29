///|
fn same_graph_shape(
  left : @worker_wire.WorkerGraphTelemetry,
  right : @worker_wire.WorkerGraphTelemetry,
) -> Bool raise RankGroupProcessError {
  if left.has_selected_shape_class() != right.has_selected_shape_class() {
    return false
  }
  if !left.has_selected_shape_class() {
    return true
  }
  let left_shape = left.selected_shape_class() catch {
    error => raise Failed(WorkerWire(error))
  }
  let right_shape = right.selected_shape_class() catch {
    error => raise Failed(WorkerWire(error))
  }
  left_shape.phase() == right_shape.phase() &&
  left_shape.batch_rows() == right_shape.batch_rows() &&
  left_shape.query_tokens_per_row() == right_shape.query_tokens_per_row() &&
  left_shape.context_tokens() == right_shape.context_tokens() &&
  left_shape.token_rows() == right_shape.token_rows()
}

///|
fn same_graph_report(
  left : @worker_wire.WorkerGraphTelemetry,
  right : @worker_wire.WorkerGraphTelemetry,
) -> Bool raise RankGroupProcessError {
  left.sequence_value() == right.sequence_value() &&
  left.selected_path() == right.selected_path() &&
  same_graph_shape(left, right) &&
  left.graph_hits() == right.graph_hits() &&
  left.graph_misses() == right.graph_misses()
}

///|
fn valid_initial_graph_report(
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
fn valid_next_graph_report(
  current : @worker_wire.WorkerGraphTelemetry,
  previous : @worker_wire.WorkerGraphTelemetry,
) -> Bool raise RankGroupProcessError {
  if current.selected_path() != previous.selected_path() ||
    !same_graph_shape(current, previous) {
    return false
  }
  let hits = current.graph_hits()
  let misses = current.graph_misses()
  let previous_hits = previous.graph_hits()
  let previous_misses = previous.graph_misses()
  if hits < previous_hits || misses < previous_misses {
    return false
  }
  match current.selected_path() {
    WorkerGraphCaptured =>
      misses == previous_misses &&
      (if previous_hits == 0xffffffffffffffffUL {
        hits == previous_hits
      } else {
        hits == previous_hits + 1UL
      })
    WorkerGraphOrderedEager =>
      hits == previous_hits &&
      (if previous_misses == 0xffffffffffffffffUL {
        misses == previous_misses
      } else {
        misses == previous_misses + 1UL
      })
  }
}

///|
fn admit_rank_graph_report(
  report : @worker_wire.WorkerGraphTelemetry,
  expected_sequence : UInt64,
  selected : @worker_wire.WorkerGraphTelemetrySlot,
  previous : @worker_wire.WorkerGraphTelemetrySlot,
) -> Bool raise RankGroupProcessError {
  if report.sequence_value() != expected_sequence {
    return false
  }
  if selected.is_available() {
    let expected = selected.current() catch {
      error => raise Failed(WorkerWire(error))
    }
    return same_graph_report(report, expected)
  }
  let counters_valid = if previous.is_available() {
    let prior = previous.current() catch {
      error => raise Failed(WorkerWire(error))
    }
    valid_next_graph_report(report, prior)
  } else {
    valid_initial_graph_report(report)
  }
  if counters_valid {
    selected.store(report)
  }
  counters_valid
}

///|
fn[T] RankGroupProcessSupervisor::fail_graph_telemetry(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
) -> T raise RankGroupProcessError {
  self.fail_active(plan, rank, GraphTelemetry(rank~))
}

///|
fn RankGroupProcessSupervisor::accept_graph_telemetry_bytes_in_place(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
  length : Int,
) -> Unit raise RankGroupProcessError {
  if rank < 0 ||
    rank >= self.rank_done.length() ||
    !self.rank_sidecar_pending[rank] ||
    self.rank_done[rank] {
    self.fail_graph_telemetry(plan, rank)
  }
  let report = @worker_wire.decode_worker_graph_telemetry(self.rx, length) catch {
    error => self.fail_active(plan, rank, WorkerWire(error))
  }
  let admitted = admit_rank_graph_report(
    report,
    self.active_sequence_value,
    self.graph_telemetry,
    self.last_graph_telemetry,
  ) catch {
    _ => self.fail_graph_telemetry(plan, rank)
  }
  if !admitted {
    self.fail_graph_telemetry(plan, rank)
  }
  self.rank_sidecar_pending[rank] = false
  self.rank_done[rank] = true
  if self.all_rank_results_ready() {
    let selected = self.graph_telemetry.current() catch {
      error => self.fail_active(plan, rank, WorkerWire(error))
    }
    self.last_graph_telemetry.store(selected)
  }
}

///|
pub fn ReceivedRankGroupCompletion::graph_telemetry(
  self : ReceivedRankGroupCompletion,
  owner : RankGroupProcessSupervisor,
) -> @worker_wire.WorkerGraphTelemetry raise RankGroupProcessError {
  let actual = self.owner()
  if !physical_equal(actual, owner) {
    raise Invalid(Owner)
  }
  let report = actual.graph_telemetry.current() catch {
    error => actual.fail(WorkerWire(error))
  }
  if report.sequence_value() != actual.active_sequence_value {
    actual.fail(GraphTelemetry(rank=-1))
  }
  report
}
