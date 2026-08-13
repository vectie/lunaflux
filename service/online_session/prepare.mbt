///|
fn preparation_failure(
  failure : OnlineSessionFailure,
) -> OnlineSessionPreparationError {
  PreparationFailed(failure)
}

///|
fn publish_online_session(
  prepared : OnlineSessionPreparation,
  session : OnlineSession,
) -> OnlineSessionPreparation {
  prepared.session = Some(session)
  prepared.state = OnlineSessionReady
  prepared
}

///|
fn publish_online_cleanup(
  prepared : OnlineSessionPreparation,
  lease : @worker_service.OnlineWorkerLease?,
  service_cleanup : @worker_service.FailedOwnedWorkerServicePreparation?,
) -> OnlineSessionPreparation {
  prepared.session = None
  prepared.cleanup.lease = lease
  prepared.cleanup.service_cleanup = service_cleanup
  if service_cleanup is Some(_) {
    prepared.cleanup.cause_kind = 2
  }
  prepared.state = OnlineSessionCleanupRequired
  prepared
}

///|
/// Constructs one alias-free, single-request online session. Receipt parsing,
/// tokenization, output ownership, event storage, scheduler, and rooted worker
/// ownership remain inside the returned one-shot preparation.
pub fn prepare_owned_session(
  received : @request_admission.ReceivedRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  wire_limits : @framed_wire.FramedWireLimits,
  blueprint : @core.SchedulerBlueprint,
  binding : @worker_service.WorkerServiceBinding,
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> OnlineSessionPreparation raise OnlineSessionPreparationError {
  if wire_limits.inference_limits() != inference ||
    binding.inference_limits() != inference {
    raise preparation_failure(Invalid(InferenceEnvelope))
  }
  if expected_model != blueprint.model_identity() {
    raise preparation_failure(Invalid(ModelIdentity))
  }
  let writer = @framed_wire.CanonicalEventWriter::new(wire_limits)
  writer.preflight_online_capacity(inference.max_decoded_delta_bytes()) catch {
    _ => raise preparation_failure(WireFailure)
  }
  let clock = @monotonic_clock.MonotonicClock::system()
  let admitted = @request_admission.admit(
    received, tokenizer, expected_tokenizer, expected_model, inference, clock,
  ) catch {
    _ => raise preparation_failure(AdmissionFailure)
  }
  let request = admitted.scheduler_request()
  if request.stream_preference() != Streaming {
    raise preparation_failure(Invalid(Streaming))
  }
  writer.prepare_accepted(request.model_identity(), request.effective_limits()) catch {
    _ => raise preparation_failure(WireFailure)
  }
  let session : OnlineSession = {
    admitted,
    lease: None,
    writer,
    output: FixedArray::make(inference.max_decoded_delta_bytes(), b'\x00'),
    input_tokens: admitted.input_token_count(),
    max_new_tokens: request.effective_limits().max_new_tokens(),
    event_ready: true,
    accepted_acked: false,
    cleanup_phase: 0,
    recovery_close: false,
    closed: false,
  }
  let failed : FailedOnlineSessionPreparation = {
    lease: None,
    service_cleanup: None,
    cause_kind: 0,
    closed: false,
  }
  let prepared : OnlineSessionPreparation = {
    session: Some(session),
    cleanup: failed,
    state: OnlineSessionCleanupRequired,
    consumed: false,
  }
  let owned = @worker_service.prepare_owned_online(
    blueprint, binding, executable, startup, source, process_limits, model_root,
    kernel_root, request, clock,
  ) catch {
    _ => raise preparation_failure(ServicePreparationFailure)
  }
  match owned.state() {
    OwnedServiceCleanupRequired => {
      let cleanup = try! owned.take_cleanup()
      publish_online_cleanup(prepared, None, Some(cleanup))
    }
    OwnedServiceReady => {
      let lease = try! owned.take_prepared_online()
      session.lease = Some(lease)
      match lease.commit_prepared_admission() {
        OnlineAdmissionCommitted => ()
        OnlineAdmissionClockFailed | OnlineAdmissionExpired => {
          failed.cause_kind = 3
          return publish_online_cleanup(prepared, Some(lease), None)
        }
      }
      publish_online_session(prepared, session)
    }
  }
}
