///|
/// Version of the native LunaFlux protocol represented by this package.
pub fn protocol_version() -> String {
  "lunaflux.inference.v1alpha1"
}

///|
/// Coarse lifecycle of one engine instance.
pub(all) enum EngineState {
  Loading
  Warming
  Ready
  Draining
  Stopped
  Failed
} derive(Debug, Eq)

///|
/// Lifecycle of one accepted generation request.
///
/// The scheduler owns every transition after admission.
pub(all) enum RequestState {
  Admitted
  Tokenizing
  Waiting
  Prefill
  Decode
  Finishing
  Cancelled
  Finished
  Failed
} derive(Debug, Eq)

///|
/// Returns whether a request transition is valid in the v1 state machine.
///
/// Terminal states cannot transition to another state.
pub fn RequestState::can_transition_to(
  self : RequestState,
  next : RequestState,
) -> Bool {
  match (self, next) {
    (Admitted, Tokenizing) => true
    (Admitted, Cancelled) => true
    (Admitted, Failed) => true
    (Tokenizing, Waiting) => true
    (Tokenizing, Cancelled) => true
    (Tokenizing, Failed) => true
    (Waiting, Prefill) => true
    (Waiting, Cancelled) => true
    (Waiting, Failed) => true
    (Prefill, Prefill) => true
    (Prefill, Decode) => true
    (Prefill, Finishing) => true
    (Prefill, Cancelled) => true
    (Prefill, Failed) => true
    (Decode, Decode) => true
    (Decode, Finishing) => true
    (Decode, Cancelled) => true
    (Decode, Failed) => true
    (Finishing, Finished) => true
    (Finishing, Cancelled) => true
    (Finishing, Failed) => true
    _ => false
  }
}
