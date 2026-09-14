///|
test "production Qwen c32 profile is admitted by fused common validation" {
  let fixture = fused_test_fixture()
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=256,
    max_page_table_entries=8192,
  )
  fused_validate_common(profile, fixture.target, fixture.compiler)
  assert_true(
    profile.max_query_tokens().to_int64() * 263U.to_int64() <= 4294967295L,
  )
}

///|
test "fused token envelope supports coalesced prefill and rejects its next boundary" {
  let fixture = fused_test_fixture()
  for tokens in [1024, 1528, 1536, 2048] {
    let profile = @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(1),
      max_query_rows=32,
      max_query_tokens=tokens,
      max_page_table_entries=8192,
    )
    fused_validate_common(profile, fixture.target, fixture.compiler)
    assert_true(fused_checked_mul(tokens.to_int64(), 16384L) <= 33554432L)
  }
  let over = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=2049,
    max_page_table_entries=8192,
  )
  try fused_validate_common(over, fixture.target, fixture.compiler) catch {
    FusedParallelUnsupported(Profile) => ()
    _ => fail("unexpected token-envelope rejection")
  } noraise {
    _ => fail("oversized fused token envelope admitted")
  }
}

///|
test "fused page envelope includes decode growth of a full long prompt batch" {
  let fixture = fused_test_fixture()
  for entries in [8192, 8193, 8320, 9216, 16384] {
    let profile = @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(1),
      max_query_rows=32,
      max_query_tokens=2048,
      max_page_table_entries=entries,
    )
    fused_validate_common(profile, fixture.target, fixture.compiler)
  }
}

///|
test "fused common validation rejects page tables above the bounded envelope" {
  let fixture = fused_test_fixture()
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=256,
    max_page_table_entries=16385,
  )
  try fused_validate_common(profile, fixture.target, fixture.compiler) catch {
    FusedParallelUnsupported(Profile) => ()
    _ => fail("unexpected fused c32 boundary rejection")
  } noraise {
    _ => fail("16,385 page-table entries reached fused artifact export")
  }
}
