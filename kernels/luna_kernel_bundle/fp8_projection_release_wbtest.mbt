///|
test "FP8 projection release closes dual CUBIN receipt and artifact provenance" {
  let plan = fp8_bundle_plan()
  let contracts = fp8_bundle_contracts(plan)
  guard contracts.contract(
      fp8_bundle_profile().id(),
      try! @plan.OperationId::new(1),
    )
    is Some(contract) else {
    fail("missing FP8 contract")
  }
  let candidate = fp8_bundle_candidate(plan)
  let lowering = try! @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest(
    candidate,
    plan.operations()[1],
    contract,
  )
  let bytes = fp8_bundle_bytes()
  let compiled = try! admit_fp8_projection_compiled_operation(
    lowering,
    contract,
    fp8_bundle_artifacts(plan),
    bytes,
    bytes,
    fp8_bundle_receipt(candidate, bytes),
    bundle_fixture_limits(),
  )
  assert_eq(compiled.operation_id(), try! @plan.OperationId::new(1))
  assert_eq(compiled.module_digest(), fp8_bundle_family(1).module_digest())
}

///|
test "FP8 projection release rejects CUBIN and artifact-symbol substitution" {
  let plan = fp8_bundle_plan()
  let contracts = fp8_bundle_contracts(plan)
  let operation = try! @plan.OperationId::new(1)
  guard contracts.contract(fp8_bundle_profile().id(), operation)
    is Some(contract) else {
    fail("missing FP8 contract")
  }
  let candidate = fp8_bundle_candidate(plan)
  let lowering = try! @luna_cuda_fp8_projection_aot.bind_fp8_projection_manifest(
    candidate,
    plan.operations()[1],
    contract,
  )
  let first = fp8_bundle_bytes()
  let substituted = b"substituted-fp8-cubin"
  try
    admit_fp8_projection_compiled_operation(
      lowering,
      contract,
      fp8_bundle_artifacts(plan),
      first,
      substituted,
      fp8_bundle_receipt(candidate, first, second=substituted),
      bundle_fixture_limits(),
    )
  catch {
    Invalid(Determinism) => ()
    _ => fail("unexpected substituted-CUBIN rejection")
  } noraise {
    _ => fail("different deterministic builds were admitted")
  }
  try
    admit_fp8_projection_compiled_operation(
      lowering,
      contract,
      fp8_bundle_artifacts(plan, output_symbol="substituted_symbol"),
      first,
      first,
      fp8_bundle_receipt(candidate, first),
      bundle_fixture_limits(),
    )
  catch {
    Invalid(ModuleIdentity) => ()
    _ => fail("unexpected artifact-symbol rejection")
  } noraise {
    _ => fail("substituted artifact symbol was admitted")
  }
}
