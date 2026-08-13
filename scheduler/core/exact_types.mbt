///|
#valtype
pub(all) enum ExactDeadlineResult {
  ExactDeadlinePending
  ExactDeadlineExpired(@inference.RequestGeneration)
  ExactDeadlineNaturalTerminal
} derive(Debug, Eq)

///|
/// Exact result of device-state invalidation for one authenticated request.
#valtype
pub(all) enum ExactDeviceInvalidation {
  ExactDeviceNoChange
  ExactDeviceInvalidated(Int, @inference.RequestGeneration)
  ExactDeviceAlreadyTerminal(@inference.RequestGeneration)
} derive(Debug, Eq)

///|
/// Exact one-request instance-loss retirement result.
#valtype
pub(all) enum ExactInstanceLossDrain {
  ExactLossRemaining(Int, @inference.RequestGeneration)
  ExactLossComplete(@inference.RequestGeneration)
} derive(Debug, Eq)
