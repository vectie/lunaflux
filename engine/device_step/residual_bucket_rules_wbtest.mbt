///|
fn residual_bucket_test_launch(
  tokens : Int,
) -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=tokens,
    grid_y=1,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=1024,
  )
}

///|
test "production residual row effects survive bounded decode and prefill launches" {
  let rule = residual_bucket_rule(
    ResidualRmsNormProductionAbiV2,
    residual_bucket_test_launch(1024),
    1024,
  )
  for tokens in [1, 8, 16, 32, 128, 512, 1024] {
    let launch = paged_bucket_launch(rule, tokens)
    assert_eq(launch.grid_x(), tokens)
    assert_eq(launch.block_x(), 128)
    assert_eq(launch.shared_memory_bytes(), 1024)
  }
  assert_true(
    residual_bucket_rule(
      ResidualRmsNormAbiV1,
      residual_bucket_test_launch(1024),
      1024,
    )
    is FixedBucketLaunch,
  )
  try
    residual_bucket_rule(
      ResidualRmsNormProductionAbiV2,
      residual_bucket_test_launch(32),
      1024,
    )
  catch {
    InvalidExecutor(GraphBucket) => ()
    _ => fail("unexpected failure")
  } noraise {
    _ => fail("truncated envelope accepted")
  }
}

///|
test "residual bounds survive split attention operation expansion" {
  let operation = @plan.OperationId::new(7)
  let attention = @plan.OperationId::new(6)
  let rule = residual_bucket_rule(
    ResidualRmsNormProductionAbiV2,
    residual_bucket_test_launch(1024),
    1024,
  )
  let contract : PagedGraphLaunchContract = {
    step_launches: [FixedBucketLaunch, rule],
    decode_step_launches: [FixedBucketLaunch, rule],
    prefill_wide_step_launches: None,
    prefill_split_step_launches: None,
  }
  let remapped = remap_decode_split_bucket_rules(
    ReadOnlyArray::from_array([(attention, 1), (operation, 2)]),
    ReadOnlyArray::from_array([(attention, 3), (attention, 4), (operation, 2)]),
    contract,
  )
  assert_true(remapped[0] is FixedBucketLaunch)
  assert_true(remapped[1] is FixedBucketLaunch)
  assert_eq(paged_bucket_launch(remapped[2], 1).grid_x(), 1)
  assert_eq(paged_bucket_launch(remapped[2], 8).grid_x(), 8)
}
