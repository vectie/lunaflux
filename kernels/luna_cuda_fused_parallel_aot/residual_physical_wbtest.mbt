///|
test "residual physical migration preserves qualification and production source" {
  let fixture = fused_test_fixture()
  let candidate = fused_test_residual_norm_candidate(fixture)
  let production = lower_fused_residual_rmsnorm_production_candidate(
    candidate,
    fused_test_limits(),
  )
  inspect(
    candidate.source_digest().as_hex(),
    content="4951b6eb49319819a897a2aee6d80ae8ee34989dec04ae1928e1eb4dc9b52a37",
  )
  inspect(
    production.source_digest().as_hex(),
    content="fbd08da504d836623ec59b6f467c192c9a3dc653e672d2315849798b638aeefa",
  )
  assert_eq(candidate.physical_program.geometry(), (128, 128, 32))
  assert_eq(candidate.physical_program.partial_count(), 128)
  assert_eq(candidate.physical_program.preparation(), [
    LoadInput,
    AddResidual,
    RoundBf16,
    AccumulateSquare,
  ])
  assert_eq(candidate.physical_program.reduction(), [
    PublishPartials,
    SynchronizeOwners,
    ReduceOwnerTree(64),
  ])
}

///|
test "residual lowering cannot silently replace shared tree with subgroup fold" {
  guard @elementwise_physical_ir.NormalizationProgram::new(
      width=128,
      owners=128,
      subgroup_lanes=32,
      residual=true,
    )
    is Some(program) else {
    fail("valid subgroup geometry rejected")
  }
  try residual_program_source(program) catch {
    FusedParallelUnsupported(StaticResources) => ()
    _ => fail("unexpected reduction rejection")
  } noraise {
    _ => fail("numerical reduction topology silently changed")
  }
}
