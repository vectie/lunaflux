///|
fn valid_graph_counter_step(
  current : @worker_wire.WorkerGraphTelemetry,
  previous : @worker_wire.WorkerGraphTelemetry,
) -> Bool {
  current.counters_advance_from(previous.graph_hits(), previous.graph_misses())
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
fn WorkerProcessSupervisor::accept_unsampled_graph_telemetry(
  self : WorkerProcessSupervisor,
  side : Int,
) -> Unit raise WorkerProcessError {
  if side == 0 {
    if self.state_a != 1 {
      self.lifecycle = Failed
      raise Invalid(Completion)
    }
    self.graph_telemetry_a.clear()
    self.state_a = 2
  } else if side == 1 {
    if self.state_b != 1 {
      self.lifecycle = Failed
      raise Invalid(Completion)
    }
    self.graph_telemetry_b.clear()
    self.state_b = 2
  } else {
    self.lifecycle = Failed
    raise Invalid(Completion)
  }
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
