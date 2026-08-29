///|
pub fn Fp8RuntimeInstanceAdmission::runtime_descriptor(
  self : Self,
) -> @descriptor_file.Fp8RuntimeDescriptorAdmission {
  self.runtime
}

///|
pub fn Fp8RuntimeInstanceAdmission::policy_digest(
  self : Self,
) -> @instance_policy_file.InstancePolicyDigest {
  self.core.policy_digest
}

///|
pub fn Fp8RuntimeInstanceAdmission::tokenizer(
  self : Self,
) -> @tokenizer.TokenizerSpec {
  self.core.tokenizer
}

///|
pub fn Fp8RuntimeInstanceAdmission::tokenizer_digest(
  self : Self,
) -> @tokenizer.TokenizerDigest {
  self.core.tokenizer_digest
}

///|
pub fn Fp8RuntimeInstanceAdmission::scheduler_blueprint(
  self : Self,
) -> @core.SchedulerBlueprint {
  self.core.scheduler_blueprint
}

///|
pub fn Fp8RuntimeInstanceAdmission::worker_binding(
  self : Self,
) -> @worker_service.WorkerServiceBinding {
  self.core.worker_binding.unwrap()
}

///|
pub fn Fp8RuntimeInstanceAdmission::service_config(
  self : Self,
) -> @service.ServiceConfig {
  self.core.service
}

///|
pub fn Fp8RuntimeInstanceAdmission::external_protocol(
  self : Self,
) -> @instance_policy_file.ExternalProtocolPolicyClaims {
  self.core.external_protocol
}

///|
pub fn Fp8RuntimeInstanceAdmission::openai_service_policy(
  self : Self,
) -> @instance_policy_file.OpenAiServicePolicyClaims? {
  self.core.openai_service
}

///|
pub fn Fp8RuntimeInstanceAdmission::inference_limits(
  self : Self,
) -> @inference.InferenceLimits {
  self.core.inference
}

///|
pub fn Fp8RuntimeInstanceAdmission::worker_process_limits(
  self : Self,
) -> @worker_process.WorkerProcessLimits {
  self.core.worker_process
}

///|
pub fn Fp8RuntimeInstanceAdmission::restart_policy(
  self : Self,
) -> @worker_service.WorkerRestartBackoffPolicy {
  self.core.restart
}

///|
pub fn Fp8RuntimeInstanceAdmission::transport_limits(
  self : Self,
) -> @online_tcp.LunaOnlineTcpLimits {
  self.core.transport
}

///|
pub fn Fp8RuntimeInstanceAdmission::framed_limits(
  self : Self,
) -> @framed_wire.FramedWireLimits {
  self.core.framed
}

///|
pub fn Fp8RuntimeInstanceAdmission::preparation_lane_count(self : Self) -> Int {
  self.core.preparation_lane_count
}

///|
pub fn Fp8RuntimeInstanceAdmission::preparation_step_budget(
  self : Self,
) -> @request_admission.LunaRequestPreparationStepBudget {
  self.core.preparation_step
}

///|
pub fn Fp8RuntimeInstanceAdmission::preparation_work_limit(
  self : Self,
) -> @request_admission.LunaRequestPreparationWorkLimit {
  self.core.preparation_work_limit
}

///|
pub fn Fp8RuntimeInstanceAdmission::preparation_storage_budget(
  self : Self,
) -> @request_admission.LunaRequestPreparationStorageBudget {
  self.core.preparation_storage
}

///|
pub fn Fp8RuntimeInstanceAdmission::event_step_budget(
  self : Self,
) -> @framed_wire.LunaFramedEventStepBudget {
  self.core.event_step
}

///|
pub fn Fp8RuntimeInstanceAdmission::telemetry_capacity(self : Self) -> Int {
  self.core.telemetry_capacity
}
