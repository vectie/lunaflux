///|
pub fn Fp8RuntimeDescriptorAdmission::descriptor_digest(
  self : Self,
) -> RuntimeDescriptorDigest {
  self.descriptor_digest
}

///|
pub fn Fp8RuntimeDescriptorAdmission::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn Fp8RuntimeDescriptorAdmission::device_ordinal(self : Self) -> Int {
  self.device_ordinal
}

///|
pub fn Fp8RuntimeDescriptorAdmission::execution(
  self : Self,
) -> @execution_manifest_file.Fp8PagedExecutionAdmissionV5 {
  self.execution
}

///|
pub fn Fp8RuntimeDescriptorAdmission::model_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.execution.model_plan().identity()
}

///|
pub fn Fp8RuntimeDescriptorAdmission::bootstrap_source(
  self : Self,
) -> @worker_wire.EncodedBootstrapSource {
  self.source
}

///|
pub fn Fp8RuntimeDescriptorAdmission::startup_contract(
  self : Self,
) -> @worker_wire.WorkerStartupContract {
  self.startup
}

///|
pub fn Fp8RuntimeDescriptorAdmission::bootstrap_digest(
  self : Self,
) -> @worker_wire.WorkerBootstrapDigest {
  self.startup.bootstrap_digest()
}
