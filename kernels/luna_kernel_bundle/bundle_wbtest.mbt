///|
fn bundle_fixture_limits(
  max_operations? : Int = 8,
  max_total_module_bytes? : Int64 = 4096L,
) -> LunaKernelBundleLimits raise {
  LunaKernelBundleLimits::new(
    max_module_bytes=1024L,
    max_total_module_bytes~,
    max_modules=4,
    max_operations~,
    manifest=@full_graph_manifest.FullGraphManifestLimits::new(
      max_manifest_bytes=65536L,
      artifact_file=@artifact_file.KernelArtifactFileLimits::new(
        max_manifest_bytes=65536L,
        max_json_depth=16,
        max_path_bytes=96,
        artifact=@artifact.KernelArtifactLimits::new(
          max_modules=4,
          max_entry_points=8,
          max_module_bytes=1024L,
          max_total_module_bytes=4096L,
          max_function_symbol_bytes=64,
        ),
      ),
    ),
  )
}

///|
fn bundle_fixture_receipt(
  module_bytes : Bytes,
  source? : String = "1".repeat(64),
  recipe? : String = "2".repeat(64),
) -> LunaDeterministicCompileReceipt raise {
  let digest = kernel_bundle_sha256(module_bytes)
  LunaDeterministicCompileReceipt::from_offline_builder(
    source_digest_sha256=source,
    recipe_digest_sha256=recipe,
    toolchain_digest_sha256="3".repeat(64),
    driver_identity_sha256="4".repeat(64),
    first_artifact_digest_sha256=digest,
    second_artifact_digest_sha256=digest,
  )
}

///|
fn bundle_fixture_compiled(
  operation : Int,
) -> LunaKernelCompiledOperation raise {
  let contract = bundle_fixture_contracts().contracts()[operation]
  let module_bytes = bundle_fixture_module(operation)
  checked_compiled_operation(
    contract,
    bundle_fixture_target(),
    "kernel_" + operation.to_string(),
    contract.dimensions(),
    contract.operands(),
    contract.workspace_bytes(),
    contract.workspace_alignment(),
    "1".repeat(64),
    "2".repeat(64),
    module_bytes,
    module_bytes,
    bundle_fixture_receipt(module_bytes),
    bundle_fixture_limits(),
  )
}

///|
fn bundle_fixture_compiled_all() -> Array[LunaKernelCompiledOperation] raise {
  [
    bundle_fixture_compiled(0),
    bundle_fixture_compiled(1),
    bundle_fixture_compiled(2),
  ]
}

///|
fn bundle_fixture_compiler() -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="3".repeat(64),
      major=13,
      minor=0,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=64,
  )
}

///|
test "typed pointwise join admits final contract and deterministic CUBIN" {
  let plan = bundle_fixture_plan()
  let contracts = bundle_fixture_contracts()
  let operation = plan.operations()[0]
  let contract = contracts.contracts()[0]
  let candidate = @luna_cuda_pointwise_aot.lower_pointwise_cuda_aot_candidate(
    operation,
    contracts.profiles()[0],
    contract.entry_point().entry_point_id(),
    bundle_fixture_target(),
    bundle_fixture_compiler(),
    contract.operands(),
    @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=65536,
    ),
  )
  let lowering = @luna_cuda_pointwise_aot.bind_manifest_pointwise_cuda_aot(
    candidate, operation, contracts,
  )
  let module_bytes = bundle_fixture_module(0)
  let receipt = LunaDeterministicCompileReceipt::from_offline_builder(
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    toolchain_digest_sha256="3".repeat(64),
    driver_identity_sha256="4".repeat(64),
    first_artifact_digest_sha256=kernel_bundle_sha256(module_bytes),
    second_artifact_digest_sha256=kernel_bundle_sha256(module_bytes),
  )
  let admitted = admit_pointwise_compiled_operation(
    lowering,
    contract,
    module_bytes,
    module_bytes,
    receipt,
    bundle_fixture_limits(),
  )
  assert_eq(admitted.operation_id(), operation.id())
  assert_eq(
    admitted.module_digest(),
    contract.entry_point().family().module_digest(),
  )
}

///|
test "typed attention join admits only the final bound candidate" {
  let plan = bundle_fixture_plan()
  let contracts = bundle_fixture_contracts()
  let operation = plan.operations()[2]
  let contract = contracts.contracts()[2]
  let candidate = @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
    operation,
    contract.profile(),
    bundle_fixture_layout(),
    contract.entry_point().entry_point_id(),
    bundle_fixture_target(),
    bundle_fixture_compiler(),
    contract.operands(),
    @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=65536,
    ),
  )
  let lowering = @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_cuda_aot(
    candidate, operation, contracts,
  )
  let module_bytes = bundle_fixture_module(2)
  let receipt = LunaDeterministicCompileReceipt::from_offline_builder(
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    toolchain_digest_sha256="3".repeat(64),
    driver_identity_sha256="4".repeat(64),
    first_artifact_digest_sha256=kernel_bundle_sha256(module_bytes),
    second_artifact_digest_sha256=kernel_bundle_sha256(module_bytes),
  )
  let admitted = admit_paged_attention_compiled_operation(
    lowering,
    contract,
    module_bytes,
    module_bytes,
    receipt,
    bundle_fixture_limits(),
  )
  assert_eq(admitted.operation_id(), operation.id())
  assert_eq(
    admitted.module_digest(),
    contract.entry_point().family().module_digest(),
  )
}

///|
test "bundle emits exact deterministic manifest and module inventory" {
  let first = produce_luna_kernel_bundle(
    bundle_fixture_plan(),
    bundle_fixture_target(),
    bundle_fixture_contracts(),
    bundle_fixture_compiled_all().rev(),
    bundle_fixture_limits(),
  )
  let second = produce_luna_kernel_bundle(
    bundle_fixture_plan(),
    bundle_fixture_target(),
    bundle_fixture_contracts(),
    bundle_fixture_compiled_all(),
    bundle_fixture_limits(),
  )
  assert_eq(
    first.execution_manifest().bytes(),
    second.execution_manifest().bytes(),
  )
  assert_eq(first.execution_manifest().operation_count(), 3)
  assert_eq(first.modules().length(), 2)
  assert_eq(first.operations().length(), 3)
  assert_eq(first.total_module_bytes(), 14L)
  assert_eq(first.operations()[0].operation_id().as_int(), 0)
  assert_eq(
    first.modules()[0].relative_path(),
    "sha256/" + first.modules()[0].digest().sha256() + ".cubin",
  )
}

///|
test "bundle rejects missing and duplicate operation coverage" {
  let operations = bundle_fixture_compiled_all()
  try
    produce_luna_kernel_bundle(
      bundle_fixture_plan(),
      bundle_fixture_target(),
      bundle_fixture_contracts(),
      operations[:2],
      bundle_fixture_limits(),
    )
  catch {
    Invalid(OperationCoverage) => ()
    _ => fail("unexpected missing-operation error")
  } noraise {
    _ => fail("expected missing-operation rejection")
  }
  try
    produce_luna_kernel_bundle(
      bundle_fixture_plan(),
      bundle_fixture_target(),
      bundle_fixture_contracts(),
      [operations[0], operations[0], operations[2]],
      bundle_fixture_limits(),
    )
  catch {
    Invalid(DuplicateOperation) => ()
    _ => fail("unexpected duplicate-operation error")
  } noraise {
    _ => fail("expected duplicate-operation rejection")
  }
}

///|
test "compile admission rejects receipt and CUBIN substitution" {
  let contract = bundle_fixture_contracts().contracts()[0]
  let expected = bundle_fixture_module(0)
  for case in 0..<3 {
    let first = if case == 2 { b"substitute" } else { expected }
    let second = expected
    let source = if case == 0 { "5".repeat(64) } else { "1".repeat(64) }
    let receipt = if case == 1 {
      bundle_fixture_receipt(b"different")
    } else {
      bundle_fixture_receipt(first, source~)
    }
    try
      checked_compiled_operation(
        contract,
        bundle_fixture_target(),
        "kernel_0",
        contract.dimensions(),
        contract.operands(),
        contract.workspace_bytes(),
        contract.workspace_alignment(),
        "1".repeat(64),
        "2".repeat(64),
        first,
        second,
        receipt,
        bundle_fixture_limits(),
      )
    catch {
      Invalid(SourceIdentity | CompileReceipt | Determinism) => ()
      _ => fail("unexpected substitution error")
    } noraise {
      _ => fail("expected substitution rejection")
    }
  }
}

///|
test "all model-plan operation kinds have a typed lowering family" {
  let kinds : Array[@plan.OperationKind] = [
    Embedding,
    RmsNorm,
    QkvProjection,
    QkRmsNorm,
    RotaryEmbedding,
    CausalAttention,
    OutputProjection,
    ResidualAdd,
    GatedMlp,
    LanguageModelHead,
  ]
  for kind in kinds {
    assert_true(model_operation_has_typed_cuda_lowering(kind))
  }
}

///|
test "Q/K RMSNorm bundle admission cannot masquerade as plain RMSNorm" {
  assert_true(pointwise_kind_matches(QkRmsNorm, QkRmsNorm))
  assert_false(pointwise_kind_matches(QkRmsNorm, RmsNorm))
  assert_false(pointwise_kind_matches(RmsNorm, QkRmsNorm))
}
