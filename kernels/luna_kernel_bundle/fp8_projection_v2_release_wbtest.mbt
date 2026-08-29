///|
fn fp8_v2_bundle_candidate(
  plan : @plan.ModelPlan,
  operation? : Int = 2,
) -> @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2 {
  try! @luna_cuda_fp8_projection_aot.lower_fp8_projection_cuda_aot_candidate_v2(
    plan,
    operation=@plan.OperationId::new(operation),
    fp8_v2_bundle_profile(),
    fp8_bundle_target(),
    fp8_bundle_compiler(),
    fp8_v2_bundle_operands(plan, operation),
    @luna_cuda_fp8_projection_aot.Fp8ProjectionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=65536,
    ),
  )
}

///|
fn fp8_v2_bundle_artifacts(
  plan : @plan.ModelPlan,
  gated_symbol? : String = "lunaflux_luna_gated_mlp_fp8_e4m3_w8a8_release_v2",
) -> @artifact.KernelArtifactBundle {
  let entries : Array[@artifact.KernelEntryPointInput] = []
  for index in 0..<5 {
    entries.push(
      @artifact.KernelEntryPointInput::new(
        entry_point=fp8_v2_bundle_entry(index),
        function_symbol=match index {
          1 => "lunaflux_luna_dense_projection_fp8_e4m3_w8a8_release_v2"
          2 => gated_symbol
          _ => "fixture_v2_kernel_" + index.to_string()
        },
      ),
    )
  }
  try! @artifact.admit_paged_v4(
    fp8_v2_bundle_contracts(plan),
    [
      @artifact.KernelModuleInput::from_bytes(
        digest=fp8_v2_bundle_family(0).module_digest(),
        module_bytes=fp8_v2_bundle_bytes(),
      ),
    ],
    entries,
    @artifact.KernelArtifactLimits::new(
      max_modules=2,
      max_entry_points=8,
      max_module_bytes=1024L,
      max_total_module_bytes=2048L,
      max_function_symbol_bytes=96,
    ),
  )
}

///|
fn fp8_v2_bundle_receipt(
  candidate : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2,
  first : Bytes,
  second : Bytes,
) -> LunaDeterministicCompileReceipt {
  try! LunaDeterministicCompileReceipt::from_offline_builder(
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    toolchain_digest_sha256="3".repeat(64),
    driver_identity_sha256="7".repeat(64),
    first_artifact_digest_sha256=kernel_bundle_sha256(first),
    second_artifact_digest_sha256=kernel_bundle_sha256(second),
  )
}

///|
test "FP8 v2 binding digest cannot collide across model identities" {
  let first_plan = fp8_v2_bundle_plan(content="6".repeat(64))
  let second_plan = fp8_v2_bundle_plan(content="8".repeat(64))
  let first_candidate = fp8_v2_bundle_candidate(first_plan)
  let second_candidate = fp8_v2_bundle_candidate(second_plan)
  assert_eq(first_candidate.source_digest(), second_candidate.source_digest())
  assert_eq(first_candidate.recipe_digest(), second_candidate.recipe_digest())
  guard fp8_v2_bundle_contracts(first_plan).contract(
      fp8_v2_bundle_profile().id(),
      try! @plan.OperationId::new(2),
    )
    is Some(first_contract) &&
    fp8_v2_bundle_contracts(second_plan).contract(
      fp8_v2_bundle_profile().id(),
      try! @plan.OperationId::new(2),
    )
    is Some(second_contract) else {
    fail("missing model-scoped FP8 v2 contracts")
  }
  let first = try! @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest_v2(
    first_candidate,
    first_plan.operations()[2],
    first_contract,
  )
  let second = try! @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest_v2(
    second_candidate,
    second_plan.operations()[2],
    second_contract,
  )
  assert_true(first.binding_digest() != second.binding_digest())
}

///|
test "FP8 v2 simple release closes exact 4-byte ABI" {
  let plan = fp8_v2_bundle_plan()
  let contracts = fp8_v2_bundle_contracts(plan)
  guard contracts.contract(
      fp8_v2_bundle_profile().id(),
      try! @plan.OperationId::new(1),
    )
    is Some(contract) else {
    fail("missing FP8 v2 simple contract")
  }
  let candidate = fp8_v2_bundle_candidate(plan, operation=1)
  let lowering = try! @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest_v2(
    candidate,
    plan.operations()[1],
    contract,
  )
  let bytes = fp8_v2_bundle_bytes()
  let compiled = try! admit_fp8_projection_compiled_operation_v2(
    lowering,
    contract,
    fp8_v2_bundle_artifacts(plan),
    bytes,
    bytes,
    fp8_v2_bundle_receipt(candidate, bytes, bytes),
    bundle_fixture_limits(),
  )
  assert_eq(candidate.workspace_bytes(), 4L)
  assert_eq(compiled.operation_id(), try! @plan.OperationId::new(1))
}

///|
test "FP8 v2 gated release closes exact 8-byte ABI and dual CUBIN receipt" {
  let plan = fp8_v2_bundle_plan()
  let contracts = fp8_v2_bundle_contracts(plan)
  guard contracts.contract(
      fp8_v2_bundle_profile().id(),
      try! @plan.OperationId::new(2),
    )
    is Some(contract) else {
    fail("missing FP8 v2 gated contract")
  }
  let candidate = fp8_v2_bundle_candidate(plan)
  let lowering = try! @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest_v2(
    candidate,
    plan.operations()[2],
    contract,
  )
  let bytes = fp8_v2_bundle_bytes()
  let compiled = try! admit_fp8_projection_compiled_operation_v2(
    lowering,
    contract,
    fp8_v2_bundle_artifacts(plan),
    bytes,
    bytes,
    fp8_v2_bundle_receipt(candidate, bytes, bytes),
    bundle_fixture_limits(),
  )
  assert_eq(compiled.operation_id(), try! @plan.OperationId::new(2))
  assert_eq(compiled.module_digest(), fp8_v2_bundle_family(2).module_digest())
}

///|
test "FP8 v2 gated release rejects non-identical second CUBIN" {
  let plan = fp8_v2_bundle_plan()
  let contracts = fp8_v2_bundle_contracts(plan)
  guard contracts.contract(
      fp8_v2_bundle_profile().id(),
      try! @plan.OperationId::new(2),
    )
    is Some(contract) else {
    fail("missing FP8 v2 gated contract")
  }
  let candidate = fp8_v2_bundle_candidate(plan)
  let lowering = try! @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest_v2(
    candidate,
    plan.operations()[2],
    contract,
  )
  let first = fp8_v2_bundle_bytes()
  let second = b"substituted-fp8-v2-cubin"
  try
    admit_fp8_projection_compiled_operation_v2(
      lowering,
      contract,
      fp8_v2_bundle_artifacts(plan),
      first,
      second,
      fp8_v2_bundle_receipt(candidate, first, second),
      bundle_fixture_limits(),
    )
  catch {
    Invalid(Determinism) => ()
    _ => fail("unexpected dual-CUBIN rejection")
  } noraise {
    _ => fail("non-identical second CUBIN was admitted")
  }
}

///|
test "FP8 v2 gated release rejects artifact symbol substitution" {
  let plan = fp8_v2_bundle_plan()
  let contracts = fp8_v2_bundle_contracts(plan)
  guard contracts.contract(
      fp8_v2_bundle_profile().id(),
      try! @plan.OperationId::new(2),
    )
    is Some(contract) else {
    fail("missing FP8 v2 gated contract")
  }
  let candidate = fp8_v2_bundle_candidate(plan)
  let lowering = try! @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest_v2(
    candidate,
    plan.operations()[2],
    contract,
  )
  let bytes = fp8_v2_bundle_bytes()
  try
    admit_fp8_projection_compiled_operation_v2(
      lowering,
      contract,
      fp8_v2_bundle_artifacts(plan, gated_symbol="substituted_v2_symbol"),
      bytes,
      bytes,
      fp8_v2_bundle_receipt(candidate, bytes, bytes),
      bundle_fixture_limits(),
    )
  catch {
    Invalid(ModuleIdentity) => ()
    _ => fail("unexpected artifact-symbol rejection")
  } noraise {
    _ => fail("substituted artifact symbol was admitted")
  }
}

///|
test "FP8 v2 gated candidate rejects simple four-byte contract replay" {
  let plan = fp8_v2_bundle_plan()
  let contracts = fp8_v2_bundle_contracts(plan)
  guard contracts.contract(
      fp8_v2_bundle_profile().id(),
      try! @plan.OperationId::new(1),
    )
    is Some(simple_contract) else {
    fail("missing FP8 v2 simple contract")
  }
  try
    @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest_v2(
      fp8_v2_bundle_candidate(plan),
      plan.operations()[2],
      simple_contract,
    )
  catch {
    Mismatch(FinalBinding) => ()
    _ => fail("unexpected four-byte contract rejection")
  } noraise {
    _ => fail("simple four-byte contract replayed as gated")
  }
}
