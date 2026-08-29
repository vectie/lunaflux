///|
/// Authenticates the complete host-side child plan before any device opens.
pub fn admit_plan_v3(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  execution : @execution_manifest_file.Fp8PagedExecutionAdmissionV5,
  expected_total_memory_bytes : Int64,
) -> Fp8DeviceWorkerPlanV3 raise Fp8DeviceWorkerV3Error {
  guard source.fp8_execution() is Some(source_execution) else {
    raise Invalid(SourceRecipe)
  }
  if source.recipe() != DenseLlamaFp8ReusablePagedAotApprovedV9 ||
    expected_total_memory_bytes <= 0L ||
    source_execution.total_memory_bytes() != expected_total_memory_bytes {
    raise Invalid(SourceRecipe)
  }
  let plan = execution.model_plan()
  if plan.identity() != startup.identity() ||
    execution.weight_inspection().identity() != startup.identity() ||
    execution.kv_layout().identity() != startup.identity() {
    raise Invalid(ModelIdentity)
  }
  if source.digest() != startup.bootstrap_source_digest() ||
    source_execution.reusable_recipe_digest() !=
    execution.reusable_recipe().digest().as_hex() ||
    startup.bootstrap_digest().as_hex() !=
    execution.release_authority().digest().as_hex() {
    raise Invalid(ReleaseIdentity)
  }
  { startup, execution, expected_total_memory_bytes }
}
