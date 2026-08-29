///|
test "rank-local contracts preserve exact dense operation and collective order" {
  let source = fixture()
  let profile = fixture_profile()
  let set = admit_fixture(
    source,
    profile,
    fixture_inputs(source.model, source.rank_plan, source.target, profile),
  )
  assert_eq(set.rank(), 1)
  assert_eq(set.world_size(), 2)
  assert_eq(set.device_target(), source.target)
  assert_eq(set.device_kv_layout(), source.local_layout)
  let constants = set.rank_constants()
  assert_eq(constants.version(), 2)
  assert_eq(constants.byte_count(), 32L)
  assert_eq(constants.alignment(), 16L)
  assert_eq(constants.rank(), 1)
  assert_eq(constants.world_size(), 2)
  assert_eq(constants.compute_major(), 8)
  assert_eq(constants.kv_layout_version(), 1)
  assert_eq(constants.tokens_per_page(), 64)
  let contracts = set.contracts()
  let expected : Array[@plan.OperationKind] = [
    Embedding,
    RmsNorm,
    QkvProjection,
    RotaryEmbedding,
    CausalAttention,
    OutputProjection,
    ResidualAdd,
    RmsNorm,
    GatedMlp,
    ResidualAdd,
    RmsNorm,
    LanguageModelHead,
  ]
  assert_eq(contracts.length(), expected.length())
  for index, kind in expected {
    assert_eq(contracts[index].operation_id().as_int(), index)
    assert_eq(contracts[index].operation_kind(), kind)
    let operands = contracts[index].operands()
    let constants_index = source.model.operations()[index]
      .runtime_inputs()
      .length()
    assert_eq(operands[constants_index].role(), RankConstants(version=2))
    let mut constants_count = 0
    for operand in operands {
      if operand.role() is RankConstants(_) {
        constants_count += 1
      }
    }
    assert_eq(constants_count, 1)
  }
  assert_eq(
    contracts[0].collective_after(),
    CollectiveSumAllReduce(sequence=1, width=8),
  )
  assert_eq(
    contracts[5].collective_after(),
    CollectiveSumAllReduce(sequence=2, width=8),
  )
  assert_eq(
    contracts[8].collective_after(),
    CollectiveSumAllReduce(sequence=3, width=8),
  )
  assert_eq(
    contracts[11].collective_after(),
    CollectiveLastAxisAllGather(sequence=4, full_width=16, local_width=8),
  )
}

///|
test "operand geometry is derived from the exact local rank plan" {
  let source = fixture()
  let profile = fixture_profile()
  let set = admit_fixture(
    source,
    profile,
    fixture_inputs(source.model, source.rank_plan, source.target, profile),
  )
  let contracts = set.contracts()
  let qkv = contracts[2]
  assert_eq(qkv.input_width(), 8)
  assert_eq(qkv.output_width(), 8)
  assert_eq(qkv.query_head_count(), 2)
  assert_eq(qkv.kv_head_count(), 1)
  assert_eq(qkv.head_dimension(), 2)
  let qkv_operands = qkv.operands()
  assert_eq(qkv_operands[0].role(), RuntimeInput(StepCounts))
  assert_eq(qkv_operands[1].role(), RankConstants(version=2))
  assert_eq(qkv_operands[1].byte_count(), 32L)
  assert_eq(qkv_operands[1].alignment(), 16L)
  assert_eq(
    qkv_operands[2].role(),
    ActivationInput(@plan.ActivationRef::new(1)),
  )
  assert_eq(qkv_operands[3].role(), WeightInput(@plan.TensorRef::new(2)))
  assert_eq(qkv_operands[4].role(), WeightInput(@plan.TensorRef::new(3)))
  assert_eq(qkv_operands[5].role(), WeightInput(@plan.TensorRef::new(4)))
  assert_eq(
    qkv_operands[6].role(),
    ActivationOutput(@plan.ActivationRef::new(2)),
  )
  assert_eq(qkv_operands[6].byte_count(), 64L)
  let attention = contracts[4]
  assert_eq(attention.input_width(), 8)
  assert_eq(attention.output_width(), 4)
  assert_eq(contracts[8].inner_width(), 6)
  let attention_operands = attention.operands()
  assert_eq(attention_operands.length(), 11)
  assert_eq(attention_operands[6].role(), RankConstants(version=2))
  assert_eq(
    attention_operands[9].role(),
    PersistentKvComponent(@plan.KvLayerId::new(0), Key),
  )
  assert_eq(
    attention_operands[10].role(),
    PersistentKvComponent(@plan.KvLayerId::new(0), Value),
  )
  assert_eq(
    attention_operands[9].byte_count(),
    source.local_layout.component_stride_bytes(),
  )
  assert_eq(contracts[11].output_width(), 8)
}

///|
test "declared full-profile artifact input fails before contract publication" {
  let source = fixture()
  let profile = fixture_profile()
  try
    admit_fixture(
      source,
      profile,
      fixture_inputs(
        source.model,
        source.rank_plan,
        source.target,
        profile,
        changed_index=2,
        full_profile=true,
      ),
    )
  catch {
    TensorParallelLaunchContractError::InvalidTensorParallelLaunchContract(
      rule=ArtifactProfile,
      profile=0,
      operation=2
    ) => ()
    _ => fail("unexpected full-profile rejection")
  } noraise {
    _ => fail("expected full-profile artifact rejection")
  }
}

///|
test "foreign artifact family and reordered launch input fail closed" {
  let source = fixture()
  let profile = fixture_profile()
  try
    admit_fixture(
      source,
      profile,
      fixture_inputs(
        source.model,
        source.rank_plan,
        source.target,
        profile,
        changed_index=4,
        changed_family=true,
      ),
    )
  catch {
    TensorParallelLaunchContractError::InvalidTensorParallelLaunchContract(
      rule=ArtifactFamily,
      profile=0,
      operation=4
    ) => ()
    _ => fail("unexpected artifact-family rejection")
  } noraise {
    _ => fail("expected artifact-family rejection")
  }
  let reordered = fixture_inputs(
    source.model,
    source.rank_plan,
    source.target,
    profile,
  )
  let first = reordered[0]
  reordered[0] = reordered[1]
  reordered[1] = first
  try admit_fixture(source, profile, reordered) catch {
    InvalidTensorParallelLaunchContract(rule=ContractOrder, ..) => ()
    _ => fail("unexpected launch-order rejection")
  } noraise {
    _ => fail("expected launch-order rejection")
  }
  let missing = fixture_inputs(
    source.model,
    source.rank_plan,
    source.target,
    profile,
  )
  ignore(missing.pop())
  try admit_fixture(source, profile, missing) catch {
    InvalidTensorParallelLaunchContract(rule=ContractCount, ..) => ()
    _ => fail("unexpected missing-launch rejection")
  } noraise {
    _ => fail("expected missing launch rejection")
  }
}
