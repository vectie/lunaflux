///|
/// Exact inert runtime descriptor admission joined into this instance.
pub fn RuntimeInstanceAdmission::runtime_descriptor(
  self : RuntimeInstanceAdmission,
) -> @descriptor_file.RuntimeDescriptorAdmission {
  self.runtime
}

///|
/// Exact independently authenticated instance-policy digest.
pub fn RuntimeInstanceAdmission::policy_digest(
  self : RuntimeInstanceAdmission,
) -> @instance_policy_file.InstancePolicyDigest {
  self.core.policy_digest
}

///|
/// Exact admitted tokenizer specification.
pub fn RuntimeInstanceAdmission::tokenizer(
  self : RuntimeInstanceAdmission,
) -> @tokenizer.TokenizerSpec {
  self.core.tokenizer
}

///|
/// Exact tokenizer identity used by request and prefix admission.
pub fn RuntimeInstanceAdmission::tokenizer_digest(
  self : RuntimeInstanceAdmission,
) -> @tokenizer.TokenizerDigest {
  self.core.tokenizer_digest
}

///|
/// Cross-component resolved scheduler/cache/worker capacity.
pub fn RuntimeInstanceAdmission::capacity(
  self : RuntimeInstanceAdmission,
) -> @runtime_resolved.ResolvedRuntimeCapacity {
  self.core.capacity
}

///|
/// Immutable scheduler construction blueprint.
pub fn RuntimeInstanceAdmission::scheduler_blueprint(
  self : RuntimeInstanceAdmission,
) -> @core.SchedulerBlueprint {
  self.core.scheduler_blueprint
}

///|
/// Exact worker-service identity and inference binding.
pub fn RuntimeInstanceAdmission::worker_binding(
  self : RuntimeInstanceAdmission,
) -> @worker_service.WorkerServiceBinding {
  self.core.worker_binding.unwrap()
}

///|
/// Exact native service configuration.
pub fn RuntimeInstanceAdmission::service_config(
  self : RuntimeInstanceAdmission,
) -> @service.ServiceConfig {
  self.core.service
}

///|
pub fn RuntimeInstanceAdmission::external_protocol(
  self : RuntimeInstanceAdmission,
) -> @instance_policy_file.ExternalProtocolPolicyClaims {
  self.core.external_protocol
}

///|
pub fn RuntimeInstanceAdmission::openai_service_policy(
  self : RuntimeInstanceAdmission,
) -> @instance_policy_file.OpenAiServicePolicyClaims? {
  self.core.openai_service
}

///|
pub fn RuntimeInstanceAdmission::inference_limits(
  self : RuntimeInstanceAdmission,
) -> @inference.InferenceLimits {
  self.core.inference
}

///|
/// Exact worker-process envelope.
pub fn RuntimeInstanceAdmission::worker_process_limits(
  self : RuntimeInstanceAdmission,
) -> @worker_process.WorkerProcessLimits {
  self.core.worker_process
}

///|
/// Exact bounded worker-restart policy.
pub fn RuntimeInstanceAdmission::restart_policy(
  self : RuntimeInstanceAdmission,
) -> @worker_service.WorkerRestartBackoffPolicy {
  self.core.restart
}

///|
/// Exact admitted TCP transport limits.
pub fn RuntimeInstanceAdmission::transport_limits(
  self : RuntimeInstanceAdmission,
) -> @online_tcp.LunaOnlineTcpLimits {
  self.core.transport
}

///|
/// Exact framed request/event envelope.
pub fn RuntimeInstanceAdmission::framed_limits(
  self : RuntimeInstanceAdmission,
) -> @framed_wire.FramedWireLimits {
  self.core.framed
}

///|
/// Exact number of startup preparation lanes.
pub fn RuntimeInstanceAdmission::preparation_lane_count(
  self : RuntimeInstanceAdmission,
) -> Int {
  self.core.preparation_lane_count
}

///|
/// Exact bounded work per preparation progress call.
pub fn RuntimeInstanceAdmission::preparation_step_budget(
  self : RuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationStepBudget {
  self.core.preparation_step
}

///|
/// Exact lifetime work limit per preparation lane.
pub fn RuntimeInstanceAdmission::preparation_work_limit(
  self : RuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationWorkLimit {
  self.core.preparation_work_limit
}

///|
/// Exact startup storage budget for all preparation lanes.
pub fn RuntimeInstanceAdmission::preparation_storage_budget(
  self : RuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationStorageBudget {
  self.core.preparation_storage
}

///|
/// Exact bounded framed-event work per progress call.
pub fn RuntimeInstanceAdmission::event_step_budget(
  self : RuntimeInstanceAdmission,
) -> @framed_wire.LunaFramedEventStepBudget {
  self.core.event_step
}

///|
/// Exact bounded structured instance-log capacity.
pub fn RuntimeInstanceAdmission::telemetry_capacity(
  self : RuntimeInstanceAdmission,
) -> Int {
  self.core.telemetry_capacity
}
