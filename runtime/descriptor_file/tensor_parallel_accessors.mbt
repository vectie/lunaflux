///|
pub fn TensorParallelRuntimeDescriptorAdmission::descriptor_digest(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> RuntimeDescriptorDigest {
  self.descriptor_digest
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::model_metadata(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> @spec.LlamaModelMetadata {
  self.model
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::model_plan(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> @plan.ModelPlan {
  self.plan
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::kv_layout(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> @device_layout.DeviceKvLayout {
  self.kv_layout
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::bootstrap_source(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> @worker_wire.EncodedBootstrapSource {
  self.source
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::worker_limits(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> @worker_protocol.WorkerProtocolLimits {
  self.worker
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::model_generation_value(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> UInt64 {
  self.model_generation.value()
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::inference_limits(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> @inference.InferenceLimits {
  self.inference
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::collective_exchange_timeout_millis(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> Int {
  self.collective_exchange_timeout_millis
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::topology(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> @device_topology.AdmittedLocalTopology {
  self.topology_selection.topology()
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::topology_declaration(
  self : TensorParallelRuntimeDescriptorAdmission,
) -> @device_topology.LocalTopologyDeclaration {
  self.topology_selection.declaration()
}

///|
pub fn TensorParallelRuntimeDescriptorAdmission::planning_inputs(
  self : TensorParallelRuntimeDescriptorAdmission,
  process_limits : @rank_group_process.RankGroupProcessLimits,
  admitted_rank_child_activation_path : Bytes,
  rank_child_executable_digest_sha256 : StringView,
) -> @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs raise RuntimeDescriptorFileError {
  if process_limits.collective_exchange_timeout_millis() !=
    self.collective_exchange_timeout_millis {
    raise AdmissionRejected(TensorParallelSource)
  }
  @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs::new(
    source=self.source,
    model_plan=self.plan,
    model_generation=self.model_generation,
    topology_declaration=self.topology_selection.declaration(),
    topology=self.topology_selection.topology(),
    inspection_blueprints=self.inspections,
    device_step_limits=self.device_step_limits,
    manifest_limits=self.manifest_limits,
    launch_limits=self.launch_limits,
    execution_limits=self.execution_limits,
    worker_limits=self.worker,
    inference_limits=self.inference,
    process_limits~,
    admitted_rank_child_activation_path~,
    rank_child_executable_digest_sha256~,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelSource)
  }
}
