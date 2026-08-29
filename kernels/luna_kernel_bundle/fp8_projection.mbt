///|
fn fp8_projection_kind_matches(
  family : @luna_cuda_fp8_projection_aot.Fp8ProjectionFamily,
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
fn checked_fp8_compiled_operation(
  contract : @launch_contract.PagedV4AotLaunchContract,
  candidate : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLowering,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  guard candidate.operands().last() is Some(workspace) else {
    raise Invalid(WorkspaceContract)
  }
  if candidate.launch_dimensions() != contract.dimensions() {
    raise Invalid(LaunchDimensions)
  }
  if candidate.operands() != contract.operands() {
    raise Invalid(OperandContract)
  }
  if contract.workspace_bytes() != 4L ||
    contract.workspace_alignment() != 4L ||
    workspace.role() != Workspace ||
    workspace.byte_count() != 4L ||
    workspace.alignment() != 4L {
    raise Invalid(WorkspaceContract)
  }
  let module_item = checked_compiled_module(
    contract.entry_point().family().module_digest(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    first_module,
    second_module,
    receipt,
    limits,
  )
  let artifact = @full_graph_manifest.FullGraphAotOperationArtifact::new(
    operation_id=contract.operation_id(),
    entry_point=contract.entry_point(),
    module_relative_path=module_item.relative_path,
    function_symbol=candidate.function_symbol(),
    dimensions=candidate.launch_dimensions(),
    operands=candidate.operands(),
    workspace_bytes=4L,
    workspace_alignment=contract.workspace_alignment(),
  ) catch {
    _ => raise Invalid(Manifest)
  }
  {
    artifact,
    module_item,
    target: candidate.target(),
    source_digest_sha256: candidate.source_digest().as_hex(),
    recipe_digest_sha256: candidate.recipe_digest().as_hex(),
    toolchain_digest_sha256: receipt.toolchain_digest_sha256,
    driver_identity_sha256: receipt.driver_identity_sha256,
  }
}

///|
/// Closes one typed FP8 projection candidate over an exact catalog-v4 launch
/// contract and two independently produced, byte-identical offline CUBINs.
/// This authenticates inert release material only; it loads or launches none.
pub fn admit_fp8_projection_compiled_operation(
  lowering : @luna_cuda_fp8_projection_aot.Fp8ProjectionManifestCudaLowering,
  contract : @launch_contract.PagedV4AotLaunchContract,
  artifacts : @artifact.KernelArtifactBundle,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  let candidate = lowering.candidate()
  if lowering.entry_point() != contract.entry_point() ||
    candidate.operation_id() != contract.operation_id() ||
    candidate.execution_digest() != contract.operation_execution_digest() {
    raise Invalid(OperationIdentity)
  }
  if artifacts.model_identity() != candidate.model_identity() ||
    artifacts.device_target() != candidate.target() ||
    artifacts.catalog_version() != @catalog.CatalogVersion::v4() {
    raise Invalid(ModelIdentity)
  }
  guard artifacts.entry_point(contract.entry_point()) is Some(artifact_entry) &&
    artifacts.find_module(contract.entry_point().family().module_digest())
    is Some(artifact_module) else {
    raise Invalid(ModuleIdentity)
  }
  if artifact_entry.function_symbol() != candidate.function_symbol() ||
    artifact_module.module_bytes() != first_module {
    raise Invalid(ModuleIdentity)
  }
  if !fp8_projection_kind_matches(candidate.family(), contract.operation_kind()) {
    raise Invalid(OperationKind)
  }
  if candidate.compiler_policy().identity().toolchain_digest_sha256() !=
    receipt.toolchain_digest_sha256 {
    raise Invalid(CompileReceipt)
  }
  checked_fp8_compiled_operation(
    contract, candidate, first_module, second_module, receipt, limits,
  )
}

///|
fn checked_fp8_compiled_operation_v2(
  contract : @launch_contract.PagedV4AotLaunchContract,
  candidate : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  guard candidate.operands().last() is Some(workspace) else {
    raise Invalid(WorkspaceContract)
  }
  let workspace_bytes = candidate.workspace_bytes()
  if (workspace_bytes != 4L && workspace_bytes != 8L) ||
    candidate.launch_dimensions() != contract.dimensions() {
    raise Invalid(LaunchDimensions)
  }
  if candidate.operands() != contract.operands() {
    raise Invalid(OperandContract)
  }
  if contract.workspace_bytes() != workspace_bytes ||
    contract.workspace_alignment() != 4L ||
    workspace.role() != Workspace ||
    workspace.byte_count() != workspace_bytes ||
    workspace.alignment() != 4L {
    raise Invalid(WorkspaceContract)
  }
  let module_item = checked_compiled_module(
    contract.entry_point().family().module_digest(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    first_module,
    second_module,
    receipt,
    limits,
  )
  let artifact = @full_graph_manifest.FullGraphAotOperationArtifact::new(
    operation_id=contract.operation_id(),
    entry_point=contract.entry_point(),
    module_relative_path=module_item.relative_path,
    function_symbol=candidate.function_symbol(),
    dimensions=candidate.launch_dimensions(),
    operands=candidate.operands(),
    workspace_bytes~,
    workspace_alignment=4L,
  ) catch {
    _ => raise Invalid(Manifest)
  }
  {
    artifact,
    module_item,
    target: candidate.target(),
    source_digest_sha256: candidate.source_digest().as_hex(),
    recipe_digest_sha256: candidate.recipe_digest().as_hex(),
    toolchain_digest_sha256: receipt.toolchain_digest_sha256,
    driver_identity_sha256: receipt.driver_identity_sha256,
  }
}

///|
/// Closes one v2 FP8 projection candidate over the exact catalog-v4 raw ABI
/// and two independently produced, byte-identical offline CUBINs. The final
/// Workspace operand is exactly four bytes for a simple projection or eight
/// bytes for compound GatedMlp. This loads or launches no artifact.
pub fn admit_fp8_projection_compiled_operation_v2(
  lowering : @luna_cuda_fp8_projection_aot.Fp8ProjectionManifestCudaLoweringV2,
  contract : @launch_contract.PagedV4AotLaunchContract,
  artifacts : @artifact.KernelArtifactBundle,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  let candidate = lowering.candidate()
  if lowering.entry_point() != contract.entry_point() ||
    candidate.operation_id() != contract.operation_id() ||
    candidate.execution_digest() != contract.operation_execution_digest() {
    raise Invalid(OperationIdentity)
  }
  if artifacts.model_identity() != candidate.model_identity() ||
    artifacts.device_target() != candidate.target() ||
    artifacts.catalog_version() != @catalog.CatalogVersion::v4() {
    raise Invalid(ModelIdentity)
  }
  guard artifacts.entry_point(contract.entry_point()) is Some(artifact_entry) &&
    artifacts.find_module(contract.entry_point().family().module_digest())
    is Some(artifact_module) else {
    raise Invalid(ModuleIdentity)
  }
  if artifact_entry.function_symbol() != candidate.function_symbol() ||
    artifact_module.module_bytes() != first_module {
    raise Invalid(ModuleIdentity)
  }
  if !fp8_projection_kind_matches(candidate.family(), contract.operation_kind()) {
    raise Invalid(OperationKind)
  }
  if candidate.compiler_policy().identity().toolchain_digest_sha256() !=
    receipt.toolchain_digest_sha256 {
    raise Invalid(CompileReceipt)
  }
  checked_fp8_compiled_operation_v2(
    contract, candidate, first_module, second_module, receipt, limits,
  )
}
