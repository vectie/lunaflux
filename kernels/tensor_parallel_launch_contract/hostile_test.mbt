///|
fn expect_rule(
  body : () -> Unit raise,
  expected : TensorParallelLaunchContractRule,
) -> Unit raise {
  try body() catch {
    TensorParallelLaunchContractError::InvalidTensorParallelLaunchContract(
      rule~,
      ..
    ) if rule == expected => ()
    _ => fail("unexpected tensor-parallel launch rejection")
  } noraise {
    _ => fail("expected tensor-parallel launch rejection")
  }
}

///|
test "world rank target layout and profile substitution fail closed" {
  let source = fixture()
  let profile = fixture_profile()
  let inputs = fixture_inputs(
    source.model,
    source.rank_plan,
    source.target,
    profile,
  )
  expect_rule(
    () => ignore(admit_fixture(source, profile, inputs, world=1)),
    WorldSize,
  )
  expect_rule(
    () => ignore(admit_fixture(source, profile, inputs, rank=0)),
    Rank,
  )
  let foreign_target = @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  expect_rule(
    () => {
      ignore(
        admit_fixture(source, profile, inputs, target=Some(foreign_target)),
      )
    },
    DeviceTarget,
  )
  expect_rule(
    () => {
      ignore(
        admit_fixture(source, profile, inputs, layout=Some(source.full_layout)),
      )
    },
    KvLayout,
  )
  let oversized = fixture_profile(tokens=65)
  expect_rule(
    () => {
      ignore(
        admit_fixture(
          source,
          oversized,
          fixture_inputs(
            source.model,
            source.rank_plan,
            source.target,
            oversized,
          ),
        ),
      )
    },
    ProfileEnvelope,
  )
}

///|
test "limits and contract multiplication are bounded" {
  expect_rule(
    () => {
      ignore(
        TensorParallelLaunchContractLimits::new(
          max_profiles=257,
          max_contracts=64,
          max_operands_per_contract=16,
        ),
      )
    },
    InvalidLimit,
  )
  let source = fixture()
  let profile = fixture_profile()
  let tiny = TensorParallelLaunchContractLimits::new(
    max_profiles=1,
    max_contracts=11,
    max_operands_per_contract=16,
  )
  expect_rule(
    () => {
      ignore(
        admit_paged_rank(
          source.model,
          source.rank_plan,
          source.local_layout,
          source.resolved,
          expected_rank=1,
          expected_world_size=2,
          expected_target=source.target,
          [profile],
          fixture_inputs(source.model, source.rank_plan, source.target, profile),
          tiny,
        ),
      )
    },
    ContractCount,
  )
}
