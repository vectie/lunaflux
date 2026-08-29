///|
pub fn RuntimeDescriptorAdmission::descriptor_digest(
  self : RuntimeDescriptorAdmission,
) -> RuntimeDescriptorDigest {
  self.descriptor_digest
}

///|
pub fn RuntimeDescriptorAdmission::model_identity(
  self : RuntimeDescriptorAdmission,
) -> @spec.ModelIdentity {
  self.execution.model_plan().identity()
}

///|
pub fn RuntimeDescriptorAdmission::target(
  self : RuntimeDescriptorAdmission,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn RuntimeDescriptorAdmission::device_ordinal(
  self : RuntimeDescriptorAdmission,
) -> Int {
  self.device_ordinal
}

///|
pub fn RuntimeDescriptorAdmission::execution(
  self : RuntimeDescriptorAdmission,
) -> @execution_manifest_file.PagedExecutionAdmission {
  self.execution
}

///|
pub fn RuntimeDescriptorAdmission::bootstrap_digest(
  self : RuntimeDescriptorAdmission,
) -> @worker_wire.WorkerBootstrapDigest {
  self.startup.bootstrap_digest()
}

///|
pub fn RuntimeDescriptorAdmission::bootstrap_source(
  self : RuntimeDescriptorAdmission,
) -> @worker_wire.EncodedBootstrapSource {
  self.source
}

///|
pub fn RuntimeDescriptorAdmission::startup_contract(
  self : RuntimeDescriptorAdmission,
) -> @worker_wire.WorkerStartupContract {
  self.startup
}

///|
/// Returns the checked join of exact arena reservations, authenticated graph
/// metadata, and the independently digest-pinned descriptor ceiling.
pub fn RuntimeDescriptorAdmission::graph_memory_capacity(
  self : RuntimeDescriptorAdmission,
) -> @runtime_resolved.ResolvedGraphMemoryCapacity {
  self.graph_memory
}

///|
pub fn I8RuntimeDescriptorAdmission::descriptor_digest(
  self : Self,
) -> RuntimeDescriptorDigest {
  self.descriptor_digest
}

///|
pub fn I8RuntimeDescriptorAdmission::model_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.execution.model_plan().identity()
}

///|
pub fn I8RuntimeDescriptorAdmission::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn I8RuntimeDescriptorAdmission::device_ordinal(self : Self) -> Int {
  self.device_ordinal
}

///|
pub fn I8RuntimeDescriptorAdmission::weight_inspection(
  self : Self,
) -> @numeric_weight_file.NumericWeightFileInspection {
  self.weights
}

///|
pub fn I8RuntimeDescriptorAdmission::execution(
  self : Self,
) -> @execution_manifest_file.I8PagedExecutionAdmissionV4 {
  self.execution
}

///|
pub fn I8RuntimeDescriptorAdmission::bootstrap_digest(
  self : Self,
) -> @worker_wire.WorkerBootstrapDigest {
  self.startup.bootstrap_digest()
}

///|
pub fn I8RuntimeDescriptorAdmission::bootstrap_source(
  self : Self,
) -> @worker_wire.EncodedBootstrapSource {
  self.source
}

///|
pub fn I8RuntimeDescriptorAdmission::startup_contract(
  self : Self,
) -> @worker_wire.WorkerStartupContract {
  self.startup
}
