///|
fn fp8_v2_kind_matches(
  family : Fp8ProjectionFamily,
  kind : @plan.OperationKind,
) -> Bool {
  match (family, kind) {
    (QkvProjection, QkvProjection)
    | (DenseProjection, OutputProjection)
    | (GatedMlp, GatedMlp)
    | (LanguageModelHead, LanguageModelHead) => true
    _ => false
  }
}

///|
pub fn bind_fp8_projection_manifest_v2(
  candidate : Fp8ProjectionCudaLoweringV2,
  operation : @plan.PlanOperation,
  contract : @launch_contract.PagedV4AotLaunchContract,
) -> Fp8ProjectionManifestCudaLoweringV2 raise Fp8ProjectionAotError {
  guard candidate.operands.last() is Some(workspace) else {
    raise Mismatch(FinalBinding)
  }
  if operation.id() != candidate.operation_id ||
    !fp8_v2_kind_matches(candidate.family, operation.kind()) ||
    operation.workspace_bytes() != candidate.workspace_bytes ||
    contract.operation_id() != operation.id() ||
    contract.operation_kind() != operation.kind() ||
    contract.operation_execution_digest() != candidate.execution_digest ||
    contract.entry_point().entry_point_id() != candidate.entry_point_id ||
    contract.profile() != candidate.profile ||
    contract.dimensions() != candidate.dimensions ||
    contract.operands() != candidate.operands ||
    contract.workspace_bytes() != candidate.workspace_bytes ||
    contract.workspace_alignment() != 4L ||
    workspace.role() != Workspace ||
    workspace.byte_count() != candidate.workspace_bytes ||
    workspace.alignment() != 4L {
    raise Mismatch(FinalBinding)
  }
  let record = @utf8.encode(
    "schema=lunaflux-luna-cuda-fp8-projection-binding-v2\n" +
    "canonical_domain=lunaflux.luna-cuda-fp8-projection-binding.v2\n" +
    "model_content_sha256=" +
    candidate.model_identity.content().as_hex() +
    "\nmodel_plan_sha256=" +
    candidate.model_identity.plan().as_hex() +
    "\n" +
    "operation_id=" +
    operation.id().as_int().to_string() +
    "\noperation_execution_sha256=" +
    candidate.execution_digest.as_hex() +
    "\nmodule_sha256=" +
    contract.entry_point().family().module_digest().sha256() +
    "\nentry_point_id=" +
    candidate.entry_point_id.as_int().to_string() +
    "\nfunction_symbol=" +
    candidate.function_symbol +
    "\nsource_sha256=" +
    candidate.source_digest.as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\nworkspace_bytes=" +
    candidate.workspace_bytes.to_string() +
    "\n",
  )
  {
    candidate,
    entry_point: contract.entry_point(),
    binding_digest: Fp8ProjectionManifestBindingDigest(fp8_sha256(record)),
  }
}
