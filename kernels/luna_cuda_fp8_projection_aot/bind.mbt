///|
pub fn bind_fp8_projection_manifest(
  candidate : Fp8ProjectionCudaLowering,
  operation : @plan.PlanOperation,
  contract : @launch_contract.PagedV4AotLaunchContract,
) -> Fp8ProjectionManifestCudaLowering raise Fp8ProjectionAotError {
  guard candidate.operands.last() is Some(workspace) else {
    raise Mismatch(FinalBinding)
  }
  if operation.id() != candidate.operation_id ||
    contract.operation_id() != operation.id() ||
    contract.operation_kind() != operation.kind() ||
    contract.operation_execution_digest() != candidate.execution_digest ||
    contract.entry_point().entry_point_id() != candidate.entry_point_id ||
    contract.profile() != candidate.profile ||
    contract.dimensions() != candidate.dimensions ||
    contract.operands() != candidate.operands ||
    contract.workspace_bytes() != 4L ||
    contract.workspace_alignment() != 4L ||
    workspace.role() != Workspace ||
    workspace.byte_count() != 4L ||
    workspace.alignment() != 4L {
    raise Mismatch(FinalBinding)
  }
  let record = @utf8.encode(
    "schema=lunaflux-luna-cuda-fp8-projection-binding-v1\n" +
    "operation_id=" +
    operation.id().as_int().to_string() +
    "\n" +
    "operation_execution_sha256=" +
    candidate.execution_digest.as_hex() +
    "\n" +
    "module_sha256=" +
    contract.entry_point().family().module_digest().sha256() +
    "\n" +
    "entry_point_id=" +
    candidate.entry_point_id.as_int().to_string() +
    "\n" +
    "function_symbol=" +
    candidate.function_symbol +
    "\n" +
    "source_sha256=" +
    candidate.source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\n",
  )
  {
    candidate,
    entry_point: contract.entry_point(),
    binding_digest: Fp8ProjectionManifestBindingDigest(fp8_sha256(record)),
  }
}
