///|
fn luna_preparation_failure(
  failure : LunaOnlineInstanceFailure,
) -> LunaOnlineInstancePreparationError {
  LunaOnlineInstancePreparationFailed(failure)
}

///|
fn publish_luna_instance(
  prepared : LunaOnlineInstancePreparation,
  instance : LunaOnlineInstance,
) -> LunaOnlineInstancePreparation {
  prepared.instance = Some(instance)
  prepared.state = LunaOnlineInstanceReady
  prepared
}

///|
fn publish_luna_cleanup(
  prepared : LunaOnlineInstancePreparation,
  service_cleanup : @worker_service.FailedOwnedWorkerServicePreparation,
) -> LunaOnlineInstancePreparation {
  prepared.instance = None
  prepared.cleanup.service_cleanup = Some(service_cleanup)
  prepared.cleanup.cause_kind = PreparationServiceCleanupFailure
  prepared.state = LunaOnlineInstanceCleanupRequired
  prepared
}

///|
/// Constructs one persistent, alias-free online instance. Event storage,
/// scheduler, rooted worker, and the online lease are created once;
/// off-reactor-prepared requests enter later through `LunaOnlineInstance::begin`.
pub fn prepare_owned_luna_online_instance(
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
) -> LunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError {
  if wire_limits.inference_limits() != inference ||
    binding.inference_limits() != inference {
    raise luna_preparation_failure(
      LunaOnlineInvalid(LunaOnlineInferenceEnvelope),
    )
  }
  if expected_model != blueprint.model_identity() {
    raise luna_preparation_failure(LunaOnlineInvalid(LunaOnlineModelIdentity))
  }
  let writer = @framed_wire.CanonicalEventWriter::new(wire_limits)
  writer.preflight_online_capacity(inference.max_decoded_delta_bytes()) catch {
    _ => raise luna_preparation_failure(LunaOnlineWireFailure)
  }
  let clock = @monotonic_clock.MonotonicClock::system()
  let instance : LunaOnlineInstance = {
    expected_tokenizer,
    expected_model,
    inference,
    wire_limits,
    claim: None,
    lease: None,
    writer,
    output: FixedArray::make(inference.max_decoded_delta_bytes(), b'\x00'),
    input_tokens: 0,
    request_epoch: 0UL,
    event_ready: false,
    event_kind: NoSessionEvent,
    accepted_acked: false,
    output_tokens: 0,
    final_tail_length: 0,
    finish_kind: FinishPending,
    string_cut: NoStringCut,
    terminal_cut: NoTerminalCut,
    deferred_cancel: false,
    deadline_code: deadline_failure_code(),
    output_code: output_failure_code(),
    worker_code: worker_failure_code(),
    failure_kind: NoLatchedFailure,
    phase: LunaInstanceIdlePhase,
    drain_requested: false,
    recovery_close: false,
  }
  let failed : FailedLunaOnlineInstancePreparation = {
    service_cleanup: None,
    cause_kind: PreparationOwnershipFailure,
    closed: false,
  }
  let prepared : LunaOnlineInstancePreparation = {
    instance: Some(instance),
    cleanup: failed,
    state: LunaOnlineInstanceCleanupRequired,
    consumed: false,
  }
  let owned = @worker_service.prepare_owned(
    blueprint, binding, executable, startup, source, process_limits, model_root,
    kernel_root,
  ) catch {
    _ => raise luna_preparation_failure(LunaOnlineServicePreparationFailure)
  }
  match owned.state() {
    OwnedServiceCleanupRequired =>
      publish_luna_cleanup(prepared, try! owned.take_cleanup())
    OwnedServiceReady => {
      instance.lease = Some(try! owned.take_online(clock))
      publish_luna_instance(prepared, instance)
    }
  }
}
