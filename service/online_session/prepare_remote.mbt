///|
pub struct RemoteLunaOnlinePreparation {
  priv instance : LunaOnlineInstance
  priv prepared : LunaOnlineInstancePreparation
  priv service : @worker_service.RemoteWorkerServicePreparation
  priv mut consumed : Bool
}

///|
/// Uses the same online instance, request admission, streaming and API owner as
/// local execution. The caller drives the two authenticated control sessions
/// through control().group().mailbox(rank) until poll_ready publishes readiness.
pub fn prepare_remote_luna_online_instance(
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  blueprint : @core.SchedulerBlueprint,
  remote : @remote_startup.RemoteGroupStartup,
  admitted : @remote_bootstrap.TensorParallelGroupBootstrap,
  limits : @remote_wire.RankGroupWireLimits,
  timeout_millis~ : UInt64,
  restart_policy : @worker_service.WorkerRestartBackoffPolicy,
) -> RemoteLunaOnlinePreparation raise {
  if expected_model != blueprint.model_identity() ||
    remote.rank(0).unwrap().worker().identity() != expected_model {
    raise luna_preparation_failure(LunaOnlineInvalid(LunaOnlineModelIdentity))
  }
  if remote.rank(0).unwrap().worker().inference_limits() != inference {
    raise luna_preparation_failure(
      LunaOnlineInvalid(LunaOnlineInferenceEnvelope),
    )
  }
  let (instance, prepared) = new_pending_luna_online_preparation(
    expected_tokenizer, expected_model, inference, blueprint,
  )
  let service = @worker_service.prepare_owned_remote(
    blueprint,
    remote,
    admitted,
    limits,
    timeout_millis~,
    restart_policy,
  )
  { instance, prepared, service, consumed: false, }
}

///|
pub fn RemoteLunaOnlinePreparation::control(
  self : RemoteLunaOnlinePreparation,
) -> @worker_service.RemoteWorkerServiceControl {
  self.service.control()
}

///|
pub fn RemoteLunaOnlinePreparation::poll_ready(
  self : RemoteLunaOnlinePreparation,
) -> LunaOnlineInstancePreparation? raise {
  if self.consumed {
    raise luna_preparation_failure(LunaOnlineInvalid(LunaOnlineLifecycle))
  }
  guard self.service.poll_ready() is Some(owned) else { return None }
  self.consumed = true
  Some(
    publish_owned_luna_online_preparation(
      self.instance,
      self.prepared,
      owned,
      @monotonic_clock.MonotonicClock::system(),
    ),
  )
}
