///|
/// Immutable request-local scheduler observation. Cached input is the exact
/// full-page prefix ownership acquired for this live request, never a global
/// telemetry delta.
pub struct RequestStatus {
  handle : RequestHandle
  state : SchedulerRequestState
  input_tokens : Int32
  processed_input_tokens : Int32
  cached_input_tokens : Int32
  output_tokens : Int32
  deadline_at_millis : UInt64
  in_flight : Bool
} derive(Debug, Eq)

///|
pub fn RequestStatus::handle(self : RequestStatus) -> RequestHandle {
  self.handle
}

///|
pub fn RequestStatus::state(self : RequestStatus) -> SchedulerRequestState {
  self.state
}

///|
pub fn RequestStatus::input_tokens(self : RequestStatus) -> Int32 {
  self.input_tokens
}

///|
pub fn RequestStatus::processed_input_tokens(self : RequestStatus) -> Int32 {
  self.processed_input_tokens
}

///|
pub fn RequestStatus::cached_input_tokens(self : RequestStatus) -> Int32 {
  self.cached_input_tokens
}

///|
pub fn RequestStatus::output_tokens(self : RequestStatus) -> Int32 {
  self.output_tokens
}

///|
pub fn RequestStatus::deadline_at_millis(self : RequestStatus) -> UInt64 {
  self.deadline_at_millis
}

///|
pub fn RequestStatus::in_flight(self : RequestStatus) -> Bool {
  self.in_flight
}
