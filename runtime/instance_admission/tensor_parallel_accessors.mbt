///|
pub fn TensorParallelRuntimeInstanceAdmission::runtime_descriptor(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @descriptor_file.TensorParallelRuntimeDescriptorAdmission {
  self.runtime
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::policy_digest(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @instance_policy_file.InstancePolicyDigest {
  self.core.policy_digest
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::tokenizer(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @tokenizer.TokenizerSpec {
  self.core.tokenizer
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::tokenizer_digest(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @tokenizer.TokenizerDigest {
  self.core.tokenizer_digest
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::scheduler_blueprint(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @core.SchedulerBlueprint {
  self.core.scheduler_blueprint
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::service_config(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @service.ServiceConfig {
  self.core.service
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::external_protocol(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @instance_policy_file.ExternalProtocolPolicyClaims {
  self.core.external_protocol
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::openai_service_policy(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @instance_policy_file.OpenAiServicePolicyClaims? {
  self.core.openai_service
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::inference_limits(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @inference.InferenceLimits {
  self.core.inference
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::rank_group_process_limits(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @rank_group_process.RankGroupProcessLimits raise RuntimeInstanceAdmissionError {
  let base = self.core.worker_process
  @rank_group_process.RankGroupProcessLimits::new(
    max_payload_bytes=base.max_frame_bytes(),
    startup_io_timeout_millis=base.startup_io_timeout_millis(),
    io_timeout_millis=base.io_timeout_millis(),
    shutdown_timeout_millis=base.shutdown_timeout_millis(),
    collective_exchange_timeout_millis=self.runtime.collective_exchange_timeout_millis(),
  ) catch {
    _ => raise reject(WorkerProcess)
  }
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::restart_policy(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @worker_service.WorkerRestartBackoffPolicy {
  self.core.restart
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::transport_limits(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @online_tcp.LunaOnlineTcpLimits {
  self.core.transport
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::framed_limits(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @framed_wire.FramedWireLimits {
  self.core.framed
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::preparation_lane_count(
  self : TensorParallelRuntimeInstanceAdmission,
) -> Int {
  self.core.preparation_lane_count
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::preparation_step_budget(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationStepBudget {
  self.core.preparation_step
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::preparation_work_limit(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationWorkLimit {
  self.core.preparation_work_limit
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::preparation_storage_budget(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @request_admission.LunaRequestPreparationStorageBudget {
  self.core.preparation_storage
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::event_step_budget(
  self : TensorParallelRuntimeInstanceAdmission,
) -> @framed_wire.LunaFramedEventStepBudget {
  self.core.event_step
}

///|
pub fn TensorParallelRuntimeInstanceAdmission::telemetry_capacity(
  self : TensorParallelRuntimeInstanceAdmission,
) -> Int {
  self.core.telemetry_capacity
}
