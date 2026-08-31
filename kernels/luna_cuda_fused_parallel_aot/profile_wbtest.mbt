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
