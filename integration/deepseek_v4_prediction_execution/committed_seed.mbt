///|
/// The coordinator supplies its last published target token, including after
/// output-free accepted-prefix replay. The input is a prepared four-byte upload;
/// poll joins it before PredictCommitted and holds the base-capture lease until
/// prediction retirement. No sampled-output counts or stale cells are consulted.
pub fn DeepSeekPredictionExecution::submit_from_committed(
  self : Self,
  seed_token : Int,
) -> Unit raise {
  guard self.state is Prepared &&
    !self.leased &&
    self.seed_input is Some(seed) &&
    self.base is Some(base) &&
    self.queues is Some(_) else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  ignore(base.acquire_prediction_input())
  self.leased = true
  self.ready_result = false
  seed.begin(seed_token) catch {
    error => {
      base.release_prediction_input()
      self.leased = false
      raise error
    }
  }
  self.seed_pending = true
  self.ready_result = true
}
