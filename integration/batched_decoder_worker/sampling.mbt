///|
/// A prepared sampling effect consumes the model's completed logits. Sampling
/// and its request RNG stay outside the model/compiler; the rank waits for its
/// retirement exactly as for a prepared device sampler. All storage is startup-
/// owned and close drains its leases before model allocations are released.
pub struct BatchedSampling {
  priv prepare : (@wire.ValidatedPlanFrame) -> Unit raise
  priv begin : () -> Unit raise
  priv poll : () -> Bool raise
  priv token_at : (Int) -> Int raise
  priv close : () -> Unit raise
  priv outputs : Int
  priv vocabulary : Int
}

///|
pub fn BatchedSampling::new(
  maximum_outputs~ : Int,
  vocabulary_size~ : Int,
  prepare~ : (@wire.ValidatedPlanFrame) -> Unit raise,
  begin~ : () -> Unit raise,
  poll~ : () -> Bool raise,
  token_at~ : (Int) -> Int raise,
  close~ : () -> Unit raise,
) -> Self raise @frame.BatchFrameError {
  guard maximum_outputs > 0 && vocabulary_size > 0 else {
    raise InvalidCapacity
  }
  {
    prepare,
    begin,
    poll,
    token_at,
    close,
    outputs: maximum_outputs,
    vocabulary: vocabulary_size,
  }
}

///|
pub fn BatchedDecoderRank::bind_sampling(
  self : Self,
  sampling : BatchedSampling,
) -> Unit raise {
  guard self.state is Ready &&
    self.outputs_token &&
    self.sampling is None &&
    sampling.outputs == self.samples.length() &&
    sampling.vocabulary == self.vocabulary else {
    raise @frame.BatchFrameError::InvalidCapacity
  }
  self.sampling = Some(sampling)
}

///|
fn requires_stochastic(plan : @wire.ValidatedPlanFrame) -> Bool raise {
  for i in 0..<plan.prefill_count() {
    let row = plan.prefill_row(i)
    if row.produces_token() && row.sampling_mode_value() != 0 {
      return true
    }
  }
  for i in 0..<plan.decode_count() {
    if plan.decode_row(i).sampling_mode_value() != 0 {
      return true
    }
  }
  false
}
