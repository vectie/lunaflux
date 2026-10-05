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
  for tokens in [1024, 1528, 1536, 2048, 2049, 4096, 8192] {
    let profile = @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(1),
      max_query_rows=32,
      max_query_tokens=tokens,
      max_page_table_entries=8192,
    )
    fused_validate_common(profile, fixture.target, fixture.compiler)
    assert_true(fused_checked_mul(tokens.to_int64(), 16384L) <= 134217728L)
  }
  let over = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=8193,
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
test "fused compiler versions retain exact identities without a Spark source patch" {
  let fixture = fused_test_fixture()
  for (minor, patch) in [(0, 88), (1, 115)] {
    let compiler = fused_test_compiler(minor~, patch~)
    fused_validate_common(fixture.profile, fixture.target, compiler)
    assert_eq(compiler.identity().minor(), minor)
    assert_eq(compiler.identity().patch(), patch)
  }
  for
    (major, minor, patch) in [
      (12, 0, 88),
      (13, 0, 87),
      (13, 1, 114),
      (13, 2, 115),
    ] {
    try
      fused_validate_common(
        fixture.profile,
        fixture.target,
        fused_test_compiler(major~, minor~, patch~),
      )
    catch {
      FusedParallelUnsupported(CompilerPolicy) => ()
      _ => fail("unexpected version rejection")
    } noraise {
      _ => fail("unqualified compiler version admitted")
    }
  }
}

///|
test "full fused ingress carries the larger profile through physical lowering" {
  let candidate = qwen_fused_test_candidate(maximum_tokens=8192)
  let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
    candidate,
    fused_test_limits(),
  )
  assert_eq(production.profile().max_query_tokens(), 8192)
  assert_true(@utf8.decode(production.source_bytes()).contains("tokens > 8192"))
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
