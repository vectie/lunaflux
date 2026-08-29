///|
fn checked_compiled_module(
  expected_digest : @catalog.AotArtifactDigest,
  source_digest_sha256 : String,
  recipe_digest_sha256 : String,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelBundleModule raise LunaKernelBundleError {
  if first_module.length() == 0 || second_module.length() == 0 {
    raise Invalid(ModuleBytes)
  }
  if first_module.length().to_int64() > limits.max_module_bytes ||
    second_module.length().to_int64() > limits.max_module_bytes {
    raise LimitExceeded(rule=ModuleBytes, maximum=limits.max_module_bytes)
  }
  if receipt.source_digest_sha256 != source_digest_sha256 {
    raise Invalid(SourceIdentity)
  }
  if receipt.recipe_digest_sha256 != recipe_digest_sha256 {
    raise Invalid(RecipeIdentity)
  }
  let first_digest = kernel_bundle_sha256(first_module)
  let second_digest = kernel_bundle_sha256(second_module)
  if receipt.first_artifact_digest_sha256 != first_digest ||
    receipt.second_artifact_digest_sha256 != second_digest {
    raise Invalid(CompileReceipt)
  }
  if first_digest != second_digest || first_module != second_module {
    raise Invalid(Determinism)
  }
  if expected_digest.sha256() != first_digest {
    raise Invalid(ModuleIdentity)
  }
  {
    digest: expected_digest,
    relative_path: "sha256/" + first_digest + ".cubin",
    module_bytes: first_module,
  }
}

///|
fn checked_compiled_operation(
  contract : @launch_contract.PagedKvAotLaunchContract,
  target : @catalog.DeviceTarget,
  function_symbol : String,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[@launch_contract.AotOperand],
  workspace_bytes : Int64,
  workspace_alignment : Int64,
  source_digest_sha256 : String,
  recipe_digest_sha256 : String,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  if dimensions != contract.dimensions() {
    raise Invalid(LaunchDimensions)
  }
  if operands != contract.operands() {
    raise Invalid(OperandContract)
  }
  if workspace_bytes != contract.workspace_bytes() ||
    workspace_alignment != contract.workspace_alignment() {
    raise Invalid(WorkspaceContract)
  }
  let module_item = checked_compiled_module(
    contract.entry_point().family().module_digest(),
    source_digest_sha256,
    recipe_digest_sha256,
    first_module,
    second_module,
    receipt,
    limits,
  )
  let artifact = @full_graph_manifest.FullGraphAotOperationArtifact::new(
    operation_id=contract.operation_id(),
    entry_point=contract.entry_point(),
    module_relative_path=module_item.relative_path,
    function_symbol~,
    dimensions~,
    operands~,
    workspace_bytes~,
    workspace_alignment~,
  ) catch {
    _ => raise Invalid(Manifest)
  }
  {
    artifact,
    module_item,
    target,
    source_digest_sha256,
    recipe_digest_sha256,
    toolchain_digest_sha256: receipt.toolchain_digest_sha256,
    driver_identity_sha256: receipt.driver_identity_sha256,
  }
}

///|
fn pointwise_kind_matches(
  family : @luna_cuda_pointwise_aot.PointwiseAotFamily,
  kind : @plan.OperationKind,
) -> Bool {
  match (family, kind) {
    (EmbeddingLookup, Embedding) => true
    (RmsNorm, RmsNorm) => true
    (PositionedRotary, RotaryEmbedding) => true
    (ResidualAdd, ResidualAdd) => true
    _ => false
  }
}

///|
/// Admits a pointwise candidate only after its strict final-contract binder and
/// an independent two-build receipt both agree with the actual CUBIN bytes.
pub fn admit_pointwise_compiled_operation(
  lowering : @luna_cuda_pointwise_aot.PointwiseManifestCudaLowering,
  contract : @launch_contract.PagedKvAotLaunchContract,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  let candidate = lowering.candidate()
  if lowering.entry_point() != contract.entry_point() ||
    candidate.operation_id() != contract.operation_id() {
    raise Invalid(OperationIdentity)
  }
  if !pointwise_kind_matches(candidate.family(), contract.operation_kind()) {
    raise Invalid(OperationKind)
  }
  if candidate.launch_dimensions() != contract.dimensions() ||
    candidate.operands() != contract.operands() {
    raise Invalid(OperandContract)
  }
  if contract.workspace_bytes() != 0L {
    raise Invalid(WorkspaceContract)
  }
  let recipe_text = @utf8.decode(candidate.recipe_bytes()) catch {
    _ => raise Invalid(RecipeIdentity)
  }
  if !recipe_text.contains(
      "toolchain_sha256=" + receipt.toolchain_digest_sha256 + "\n",
    ) {
    raise Invalid(CompileReceipt)
  }
  checked_compiled_operation(
    contract,
    candidate.target(),
    candidate.function_symbol(),
    candidate.launch_dimensions(),
    candidate.operands(),
    contract.workspace_bytes(),
    contract.workspace_alignment(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    first_module,
    second_module,
    receipt,
    limits,
  )
}

///|
/// Admits projection-family output only after the strict post-compile binder
/// has joined the candidate to the final content-addressed entry point.
pub fn admit_projection_compiled_operation(
  lowering : @luna_cuda_projection_aot.ProjectionManifestCudaLowering,
  contract : @launch_contract.PagedKvAotLaunchContract,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  let candidate = lowering.candidate()
  if candidate.operation_id() != contract.operation_id() ||
    lowering.entry_point() != contract.entry_point() {
    raise Invalid(OperationIdentity)
  }
  if candidate.compiler_policy().identity().toolchain_digest_sha256() !=
    receipt.toolchain_digest_sha256 {
    raise Invalid(CompileReceipt)
  }
  checked_compiled_operation(
    contract,
    candidate.target(),
    candidate.function_symbol(),
    candidate.launch_dimensions(),
    candidate.operands(),
    contract.workspace_bytes(),
    contract.workspace_alignment(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    first_module,
    second_module,
    receipt,
    limits,
  )
}

///|
/// Admits paged-attention output only after its strict post-compile binder.
pub fn admit_paged_attention_compiled_operation(
  lowering : @luna_cuda_paged_attention_aot.PagedAttentionManifestCudaLowering,
  contract : @launch_contract.PagedKvAotLaunchContract,
  first_module : Bytes,
  second_module : Bytes,
  receipt : LunaDeterministicCompileReceipt,
  limits : LunaKernelBundleLimits,
) -> LunaKernelCompiledOperation raise LunaKernelBundleError {
  let candidate = lowering.candidate()
  if candidate.operation_id() != contract.operation_id() ||
    lowering.entry_point() != contract.entry_point() ||
    contract.operation_kind() != CausalAttention {
    raise Invalid(OperationIdentity)
  }
  if candidate.compiler_policy().identity().toolchain_digest_sha256() !=
    receipt.toolchain_digest_sha256 {
    raise Invalid(CompileReceipt)
  }
  checked_compiled_operation(
    contract,
    candidate.target(),
    candidate.function_symbol(),
    candidate.launch_dimensions(),
    candidate.operands(),
    contract.workspace_bytes(),
    contract.workspace_alignment(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    first_module,
    second_module,
    receipt,
    limits,
  )
}
