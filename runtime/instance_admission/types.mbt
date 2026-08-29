///|
/// Stable pure-join admission stages without paths, digests, or payloads.
pub enum RuntimeInstanceAdmissionStage {
  RuntimeDigest
  TokenizerBinding
  TokenizerEnvelope
  Scheduler
  Cache
  Layout
  Service
  ExternalProtocol
  WorkerProcess
  Restart
  Transport
  FramedWire
  Preparation
  Telemetry
  Capacity
  CapabilityRecipe
  SchedulerBlueprint
  WorkerBinding
} derive(Debug, Eq)

///|
/// Payload-safe pure instance-policy/runtime coherence failure.
pub suberror RuntimeInstanceAdmissionError {
  Rejected(RuntimeInstanceAdmissionStage)
} derive(Debug, Eq)

///| Immutable root-free activation foundation for one admitted instance.

///|
/// This owns no filesystem authority, file alias, native process, device
/// context, socket, mutable scheduler, or readiness claim.
pub struct RuntimeInstanceAdmission {
  priv runtime : @descriptor_file.RuntimeDescriptorAdmission
  priv core : RuntimeInstanceDerived
}

///| Immutable root-free activation foundation for one admitted I8 instance.

///|
/// This parallel owner retains the strict runtime-v2 descriptor while keeping
/// scheduler, tokenizer, cache, worker, transport, and telemetry construction
/// private. It owns no filesystem, process, socket, device, or readiness
/// authority.
pub struct I8RuntimeInstanceAdmission {
  priv runtime : @descriptor_file.I8RuntimeDescriptorAdmission
  priv core : RuntimeInstanceDerived
}

///|
pub struct Fp8RuntimeInstanceAdmission {
  priv runtime : @descriptor_file.Fp8RuntimeDescriptorAdmission
  priv core : RuntimeInstanceDerived
}

///|
/// Pure instance-policy/tokenizer join for one exact tensor-parallel runtime.
/// It retains no process, collective, device, socket, or readiness authority.
pub struct TensorParallelRuntimeInstanceAdmission {
  priv runtime : @descriptor_file.TensorParallelRuntimeDescriptorAdmission
  priv core : RuntimeInstanceDerived
}

///|
priv struct RuntimeInstanceDerived {
  policy_digest : @instance_policy_file.InstancePolicyDigest
  tokenizer : @tokenizer.TokenizerSpec
  tokenizer_digest : @tokenizer.TokenizerDigest
  capacity : @runtime_resolved.ResolvedRuntimeCapacity
  scheduler_blueprint : @core.SchedulerBlueprint
  worker_binding : @worker_service.WorkerServiceBinding?
  service : @service.ServiceConfig
  external_protocol : @instance_policy_file.ExternalProtocolPolicyClaims
  openai_service : @instance_policy_file.OpenAiServicePolicyClaims?
  inference : @inference.InferenceLimits
  worker_process : @worker_process.WorkerProcessLimits
  restart : @worker_service.WorkerRestartBackoffPolicy
  transport : @online_tcp.LunaOnlineTcpLimits
  framed : @framed_wire.FramedWireLimits
  preparation_lane_count : Int
  preparation_step : @request_admission.LunaRequestPreparationStepBudget
  preparation_work_limit : @request_admission.LunaRequestPreparationWorkLimit
  preparation_storage : @request_admission.LunaRequestPreparationStorageBudget
  event_step : @framed_wire.LunaFramedEventStepBudget
  telemetry_capacity : Int
}
