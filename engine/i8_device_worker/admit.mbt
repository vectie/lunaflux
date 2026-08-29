///|
fn exact_i8_startup_contract(
  startup : @worker_wire.WorkerStartupContract,
  execution : @execution_manifest_file.I8PagedExecutionAdmissionV4,
) -> @worker_wire.WorkerStartupContract raise I8DeviceWorkerError {
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    startup.model_generation_value(),
  ) catch {
    error => raise Protocol(error)
  }
  let expected = execution
    .bootstrap_manifest()
    .worker_startup_contract(
      generation,
      bootstrap_source_digest=startup.bootstrap_source_digest(),
      startup.worker_limits(),
      startup.inference_limits(),
      has_predecessor=startup.has_predecessor(),
      predecessor_value=startup.predecessor_value(),
    ) catch {
      error => raise Bootstrap(error)
    }
  if expected != startup {
    raise Invalid(StartupContract)
  }
  expected
}

///|
fn validate_i8_admission_evidence(
  startup : @worker_wire.WorkerStartupContract,
  model_plan : @plan.ModelPlan,
  inspection : @numeric_weight_file.NumericWeightFileInspection,
  execution : @execution_manifest_file.I8PagedExecutionAdmissionV4,
) -> Unit raise I8DeviceWorkerError {
  let identity = model_plan.identity()
  let schema = model_plan.numeric_binding().digest()
  if startup.identity() != identity ||
    execution.model_plan() != model_plan ||
    execution.blueprint().identity() != identity ||
    execution.bootstrap_manifest().identity() != identity {
    raise Invalid(ModelIdentity)
  }
  if inspection.identity() != identity ||
    inspection.numeric_schema_digest() != schema ||
    inspection.layout() != execution.weight_layout() ||
    inspection.artifact_digest() != execution.weight_artifact_digest() ||
    execution.weight_layout().identity() != identity ||
    execution.weight_layout().numeric_schema_digest() != Some(schema) ||
    execution.bootstrap_manifest().numeric_schema_digest() != schema {
    raise Invalid(NumericWeightEnvelope)
  }
  if execution.bootstrap_manifest().device_numeric_digest() !=
    execution.device_numeric_capability().digest() ||
    execution.bootstrap_manifest().target() != execution.blueprint().target() ||
    execution.bootstrap_manifest().device_step_limits() !=
    execution.device_step_limits() {
    raise Invalid(DeviceNumericCapability)
  }
}

///|
/// Joins the exact startup contract, numeric model plan, authenticated file
/// inspection, schema-v4 execution admission, and bootstrap-v3 identity before
/// any device or filesystem resource opens.
pub fn admit_plan(
  startup~ : @worker_wire.WorkerStartupContract,
  model_plan~ : @plan.ModelPlan,
  inspection~ : @numeric_weight_file.NumericWeightFileInspection,
  execution~ : @execution_manifest_file.I8PagedExecutionAdmissionV4,
  expected_total_memory_bytes~ : Int64,
) -> I8DeviceWorkerPlan raise I8DeviceWorkerError {
  if expected_total_memory_bytes <= 0L {
    raise Invalid(DeviceTarget)
  }
  validate_i8_admission_evidence(startup, model_plan, inspection, execution)
  {
    startup: exact_i8_startup_contract(startup, execution),
    model_plan,
    inspection,
    execution,
    expected_total_memory_bytes,
  }
}
