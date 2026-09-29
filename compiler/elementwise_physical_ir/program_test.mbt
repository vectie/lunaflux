///|
test "normalization preserves residual rounding and explicit reduction topology" {
  guard @elementwise_physical_ir.NormalizationProgram::new(
      width=129,
      owners=64,
      subgroup_lanes=16,
      residual=true,
    )
    is Some(norm) else {
    fail("valid norm")
  }
  assert_eq(norm.values_per_owner(), 3)
  assert_eq(norm.partial_count(), 4)
  assert_eq(norm.preparation(), [
    LoadInput,
    AddResidual,
    RoundBf16,
    AccumulateSquare,
  ])
  assert_eq(norm.reduction(), [
    ReduceSubgroup,
    PublishPartials,
    SynchronizeOwners,
    ReducePartials,
    PublishTotal,
    SynchronizeOwners,
  ])
  guard @elementwise_physical_ir.NormalizationProgram::new(
      width=129,
      owners=64,
      subgroup_lanes=16,
      residual=false,
      owner_tree=true,
    )
    is Some(tree) else {
    fail("valid tree")
  }
  assert_eq(tree.partial_count(), 64)
  assert_eq(tree.reduction(), [
    PublishPartials,
    SynchronizeOwners,
    ReduceOwnerTree(32),
  ])
  assert_eq(tree.preparation(), [LoadInput, AccumulateSquare])
  assert_eq(
    @elementwise_physical_ir.NormalizationProgram::new(
      width=1,
      owners=33,
      subgroup_lanes=32,
      residual=false,
    ),
    None,
  )
  assert_eq(
    @elementwise_physical_ir.NormalizationProgram::new(
      width=1,
      owners=64,
      subgroup_lanes=3,
      residual=false,
    ),
    None,
  )
}

///|
test "rotary disjoint domains cover each pair and exact value tail" {
  guard @elementwise_physical_ir.RotaryProgram::new(
      width=1024,
      query_heads=4,
      key_heads=2,
      head_dimension=128,
      owners=32,
    )
    is Some(rotary) else {
    fail("valid rotary")
  }
  assert_eq(rotary.geometry(), (64, 6, 768, 8, 32))
  assert_eq(rotary.work_per_token(), 72)
  assert_eq(
    @elementwise_physical_ir.RotaryProgram::new(
      width=8,
      query_heads=1,
      key_heads=1,
      head_dimension=3,
      owners=32,
    ),
    None,
  )
  assert_eq(
    @elementwise_physical_ir.RotaryProgram::new(
      width=1024,
      query_heads=32,
      key_heads=1,
      head_dimension=2,
      owners=32,
    ),
    None,
  )
}

///|
test "KV effects maintain split stores and reject overflow without wrap" {
  guard @elementwise_physical_ir.KvWriteProgram::new(
      query_heads=4,
      kv_heads=2,
      head_dimension=128,
      page_stride_bytes=8192L,
      element_bytes=2,
      owners=64,
    )
    is Some(kv) else {
    fail("valid KV")
  }
  assert_eq(kv.geometry(), (512, 256, 4096L))
  assert_eq(kv.ownership().geometry(), (256, 64, 4))
  assert_eq(kv.effects(), [StoreKey, StoreValue])
  assert_eq(
    @elementwise_physical_ir.KvWriteProgram::new(
      query_heads=2147483647,
      kv_heads=2147483647,
      head_dimension=2147483647,
      page_stride_bytes=2L,
      element_bytes=2,
      owners=32,
    ),
    None,
  )
  assert_eq(
    @elementwise_physical_ir.KvWriteProgram::new(
      query_heads=1,
      kv_heads=1,
      head_dimension=2,
      page_stride_bytes=3L,
      element_bytes=2,
      owners=32,
    ),
    None,
  )
}

///|
test "greedy tree ownership and total order are backend independent" {
  guard @elementwise_physical_ir.SelectionProgram::new(
      values=129,
      owners=64,
      subgroup_lanes=16,
      reduction=SubgroupTree,
    )
    is Some(program) else {
    fail("valid selection")
  }
  assert_eq(program.geometry(), (129, 64, 16))
  assert_eq(program.effects(), [
    ScanOwnedValues,
    ReduceSubgroupSummaries(8),
    PublishSubgroupSummaries,
    SynchronizeSelection,
    ReducePublishedSubgroups(8),
    StoreSelection,
  ])
  assert_true(program.prefers(3.0, 5, 2.0, 1))
  assert_true(program.prefers(3.0, 1, 3.0, 5))
  assert_false(program.prefers(3.0, 5, 3.0, 1))
  assert_eq(
    @elementwise_physical_ir.SelectionProgram::new(
      values=1,
      owners=63,
      subgroup_lanes=16,
      reduction=OwnerTree,
    ),
    None,
  )
}

///|
test "stochastic replay order and ingress publication are explicit" {
  guard @elementwise_physical_ir.StochasticProgram::new(
      values=151936,
      owners=128,
      subgroup_lanes=64,
      maximum_top_k=64,
    )
    is Some(program) else {
    fail("valid stochastic")
  }
  assert_eq(program.final_steps(), [
    OrderedSoftmax,
    RequireFiniteMass,
    MinimalProbabilityPrefix,
    SumRetainedMass,
    CounterDraw,
    OrderedInverseCdf,
    PublishSample,
  ])
  assert_eq(
    @elementwise_physical_ir.StochasticProgram::new(
      values=10,
      owners=128,
      subgroup_lanes=64,
      maximum_top_k=11,
    ),
    None,
  )
  guard @elementwise_physical_ir.IngressProgram::new(
      query_heads=4,
      kv_heads=2,
      head_dimension=128,
      owners=32,
    )
    is Some(ingress) else {
    fail("valid ingress")
  }
  assert_eq(ingress.geometry(), (8, 6, 256))
  assert_eq(ingress.ownership().geometry(), (128, 32, 4))
  assert_eq(ingress.effects(), [
    LoadHead,
    PublishHead,
    NormalizeRotateQueryKey,
    StoreActivationAndCache,
    RetireHead,
  ])
  assert_eq(
    @elementwise_physical_ir.IngressProgram::new(
      query_heads=2147483647,
      kv_heads=2147483647,
      head_dimension=2147483647,
      owners=32,
    ),
    None,
  )
}
