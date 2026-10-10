///|
/// Immutable request-local cancellation plan for bounded execution/replay.
/// Steps count actual submitted ordinary or verification transactions, not
/// tokens or polling iterations. Zero never means an implicit lifecycle limit.
pub struct SerialRequestCancellation {
  priv request_index : Int
  priv submitted_steps : Int
} derive(Eq)

///|
pub extend SerialRequestCancellation with Eq::{equal, not_equal}

///|
pub fn SerialRequestCancellation::new(
  request_index~ : Int,
  after_submitted_steps~ : Int,
) -> Self raise SerialGenerationError {
  guard request_index >= 0 && after_submitted_steps > 0 else {
    raise InvalidLimit
  }
  { request_index, submitted_steps: after_submitted_steps, }
}

///|
/// Validate the queue selection before allocating any device resources.
pub fn SerialRequestCancellation::validate_request_count(
  self : Self,
  request_count : Int,
) -> Unit raise SerialGenerationError {
  guard request_count > 0 && self.request_index < request_count else {
    raise InvalidLimit
  }
}

///|
/// Zero disables the optional plan for another request. The caller polls no
/// file or timer to consume this already-prepared scalar decision.
pub fn SerialRequestCancellation::after_steps_for(
  self : Self,
  request_index : Int,
) -> Int {
  if request_index == self.request_index {
    self.submitted_steps
  } else {
    0
  }
}
