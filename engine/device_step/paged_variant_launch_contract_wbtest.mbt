///|
fn variant_test_launch(
  shared : Int,
) -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=94,
    grid_y=16,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=shared,
  )
}

///|
test "wide prefill buckets preserve selected tile and shared memory together" {
  let primary = variant_test_launch(45056)
  let wide = variant_test_launch(73728)
  let contract : PagedGraphLaunchContract = {
    step_launches: [QueryTileCappedGridX(primary, 32)],
    decode_step_launches: [QueryTokenCappedGridX(primary)],
    prefill_wide_step_launches: Some([QueryTileCappedGridX(wide, 64)]),
    prefill_split_step_launches: None,
  }
  // 59 tokens reproduced the physical failure: the old composer launched
  // c314 with the primary kernel's 45,056-byte arena instead of 73,728 bytes.
  for bound in [1, 32, 59, 64, 65, 128, 256, 512, 1024] {
    let selected = paged_bucket_launch(
      paged_phase_bucket_rule(contract, WidePrefillLaunch, 0),
      bound,
    )
    assert_eq(selected.grid_x(), (bound + 63) / 64)
    assert_eq(selected.grid_y(), 16)
    assert_eq(selected.block_x(), 256)
    assert_eq(selected.shared_memory_bytes(), 73728)
    let baseline = paged_bucket_launch(
      paged_phase_bucket_rule(contract, BaselineLaunch, 0),
      bound,
    )
    assert_eq(baseline.grid_x(), (bound + 31) / 32)
    assert_eq(baseline.shared_memory_bytes(), 45056)
  }
}

///|
test "missing variant rule preserves exact selected launch without baseline fallback" {
  let primary = variant_test_launch(45056)
  let contract : PagedGraphLaunchContract = {
    step_launches: [QueryTileCappedGridX(primary, 32)],
    decode_step_launches: [QueryTokenCappedGridX(primary)],
    prefill_wide_step_launches: None,
    prefill_split_step_launches: None,
  }
  for phase in [WidePrefillLaunch, SplitPrefillLaunch] {
    assert_true(
      paged_phase_bucket_rule(contract, phase, 0) is FixedBucketLaunch,
    )
  }
  assert_true(
    paged_phase_bucket_rule(contract, BaselineLaunch, 1) is FixedBucketLaunch,
  )
  assert_true(
    paged_phase_bucket_rule(contract, DecodeLaunch, 0)
    is QueryTokenCappedGridX(_),
  )
}
