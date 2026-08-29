///|
/// Startup-owned scalar slot for one report. Mutating or reading the slot does
/// not allocate and carries no execution/admission authority.
pub struct WorkerGraphTelemetrySlot {
  priv mut valid : Bool
  priv mut sequence : UInt64
  priv mut path : Int
  priv mut shape_code : UInt64
  priv mut graph_hits : UInt64
  priv mut graph_misses : UInt64
}

///|
pub fn WorkerGraphTelemetrySlot::new() -> WorkerGraphTelemetrySlot {
  {
    valid: false,
    sequence: 0UL,
    path: 0,
    shape_code: 0UL,
    graph_hits: 0UL,
    graph_misses: 0UL,
  }
}

///|
pub fn WorkerGraphTelemetrySlot::is_available(
  self : WorkerGraphTelemetrySlot,
) -> Bool {
  self.valid
}

///|
pub fn WorkerGraphTelemetrySlot::clear(self : WorkerGraphTelemetrySlot) -> Unit {
  self.valid = false
  self.sequence = 0UL
  self.path = 0
  self.shape_code = 0UL
  self.graph_hits = 0UL
  self.graph_misses = 0UL
}

///|
pub fn WorkerGraphTelemetrySlot::store(
  self : WorkerGraphTelemetrySlot,
  report : WorkerGraphTelemetry,
) -> Unit {
  self.sequence = report.sequence
  self.path = report.path
  self.shape_code = report.shape_code
  self.graph_hits = report.graph_hits
  self.graph_misses = report.graph_misses
  self.valid = true
}

///|
pub fn WorkerGraphTelemetrySlot::current(
  self : WorkerGraphTelemetrySlot,
) -> WorkerGraphTelemetry raise WorkerWireError {
  if !self.valid {
    raise InvalidFrame(Counts)
  }
  {
    sequence: self.sequence,
    path: self.path,
    shape_code: self.shape_code,
    graph_hits: self.graph_hits,
    graph_misses: self.graph_misses,
  }
}
