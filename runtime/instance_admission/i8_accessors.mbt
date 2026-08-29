///|
pub fn I8RuntimeInstanceAdmission::runtime_descriptor(
  self : I8RuntimeInstanceAdmission,
) -> @descriptor_file.I8RuntimeDescriptorAdmission {
  self.runtime
}

///|
pub fn I8RuntimeInstanceAdmission::policy_digest(
  self : I8RuntimeInstanceAdmission,
) -> @instance_policy_file.InstancePolicyDigest {
  self.core.policy_digest
}

///|
pub fn I8RuntimeInstanceAdmission::tokenizer(
  self : I8RuntimeInstanceAdmission,
) -> @tokenizer.TokenizerSpec {
  self.core.tokenizer
}

///|
pub fn I8RuntimeInstanceAdmission::tokenizer_digest(
  self : I8RuntimeInstanceAdmission,
) -> @tokenizer.TokenizerDigest {
  self.core.tokenizer_digest
}

///|
pub fn I8RuntimeInstanceAdmission::capacity(
  self : I8RuntimeInstanceAdmission,
) -> @runtime_resolved.ResolvedRuntimeCapacity {
  self.core.capacity
}

///|
pub fn I8RuntimeInstanceAdmission::scheduler_blueprint(
  self : I8RuntimeInstanceAdmission,
) -> @core.SchedulerBlueprint {
  self.core.scheduler_blueprint
}

///|
pub fn I8RuntimeInstanceAdmission::worker_binding(
  self : I8RuntimeInstanceAdmission,
) -> @worker_service.WorkerServiceBinding {
  self.core.worker_binding.unwrap()
}

///|
pub fn I8RuntimeInstanceAdmission::service_config(
  self : I8RuntimeInstanceAdmission,
) -> @service.ServiceConfig {
  self.core.service
}

///|
pub fn I8RuntimeInstanceAdmission::external_protocol(
  self : I8RuntimeInstanceAdmission,
) -> @instance_policy_file.ExternalProtocolPolicyClaims {
  self.core.external_protocol
}

///|
pub fn I8RuntimeInstanceAdmission::openai_service_policy(
  self : I8RuntimeInstanceAdmission,
) -> @instance_policy_file.OpenAiServicePolicyClaims? {
  self.core.openai_service
}

///|
pub fn I8RuntimeInstanceAdmission::inference_limits(
  self : I8RuntimeInstanceAdmission,
) -> @inference.InferenceLimits {
  self.core.inference
}

///|
pub fn I8RuntimeInstanceAdmission::worker_process_limits(
  self : I8RuntimeInstanceAdmission,
) -> @worker_process.WorkerProcessLimits {
  self.core.worker_process
}

///|
pub fn I8RuntimeInstanceAdmission::restart_policy(
  self : I8RuntimeInstanceAdmission,
) -> @worker_service.WorkerRestartBackoffPolicy {
  self.core.restart
}

///|
pub fn I8RuntimeInstanceAdmission::transport_limits(
  self : I8RuntimeInstanceAdmission,
) -> @online_tcp.LunaOnlineTcpLimits {
  self.core.transport
}

///|
pub fn I8RuntimeInstanceAdmission::framed_limits(
  self : I8RuntimeInstanceAdmission,
) -> @framed_wire.FramedWireLimits {
  self.core.framed
}

///|
pub fn I8RuntimeInstanceAdmission::preparation_lane_count(
  self : I8RuntimeInstanceAdmission,
) -> Int {
  self.core.preparation_lane_count
}

///|
pub fn I8RuntimeInstanceAdmission::preparation_step_budget(
  self : I8RuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationStepBudget {
  self.core.preparation_step
}

///|
pub fn I8RuntimeInstanceAdmission::preparation_work_limit(
  self : I8RuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationWorkLimit {
  self.core.preparation_work_limit
}

///|
pub fn I8RuntimeInstanceAdmission::preparation_storage_budget(
  self : I8RuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationStorageBudget {
  self.core.preparation_storage
}

///|
pub fn I8RuntimeInstanceAdmission::event_step_budget(
  self : I8RuntimeInstanceAdmission,
) -> @framed_wire.LunaFramedEventStepBudget {
  self.core.event_step
}

///|
pub fn I8RuntimeInstanceAdmission::telemetry_capacity(
  self : I8RuntimeInstanceAdmission,
) -> Int {
  self.core.telemetry_capacity
}
