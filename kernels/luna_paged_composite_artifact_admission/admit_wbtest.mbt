///|
fn paged_composite_test_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn paged_composite_test_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(7),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=8,
  )
}

///|
test "qualified KV fallback identity is content-only and exact" {
  let identity = qualified_paged_kv_write_fallback_identity(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    candidate_sha256="3".repeat(64),
    source_sha256="4".repeat(64),
    recipe_sha256="5".repeat(64),
    compiled_binding_sha256="6".repeat(64),
    module_sha256="7".repeat(64),
    physical_evidence_sha256="8".repeat(64),
    approved_policy_sha256="9".repeat(64),
    fallback_source_sha256="a".repeat(64),
    fallback_recipe_sha256="b".repeat(64),
    operation=@plan.OperationId::new(2),
    activation=@plan.ActivationRef::new(3),
    row_width=8,
    target=paged_composite_test_target(),
    profile=paged_composite_test_profile(),
    tokens_per_page=16,
    total_page_count=32,
    page_stride_bytes=256L,
    toolchain_sha256="c".repeat(64),
    driver_sha256="d".repeat(64),
  )
  assert_eq(identity.operation_id().as_int(), 2)
  assert_eq(identity.positioned_qkv_activation().as_int(), 3)
  assert_eq(identity.positioned_qkv_row_width(), 8)
  assert_eq(identity.fallback_source_digest_sha256(), "a".repeat(64))
}

///|
test "qualified KV fallback rejects malformed digest and resource shape" {
  for selector in 0..<2 {
    try
      ignore(
        qualified_paged_kv_write_fallback_identity(
          model_content_sha256=if selector == 0 {
            "bad"
          } else {
            "1".repeat(64)
          },
          model_plan_sha256="2".repeat(64),
          candidate_sha256="3".repeat(64),
          source_sha256="4".repeat(64),
          recipe_sha256="5".repeat(64),
          compiled_binding_sha256="6".repeat(64),
          module_sha256="7".repeat(64),
          physical_evidence_sha256="8".repeat(64),
          approved_policy_sha256="9".repeat(64),
          fallback_source_sha256="a".repeat(64),
          fallback_recipe_sha256="b".repeat(64),
          operation=@plan.OperationId::new(2),
          activation=@plan.ActivationRef::new(3),
          row_width=if selector == 1 { 0 } else { 8 },
          target=paged_composite_test_target(),
          profile=paged_composite_test_profile(),
          tokens_per_page=16,
          total_page_count=32,
          page_stride_bytes=256L,
          toolchain_sha256="c".repeat(64),
          driver_sha256="d".repeat(64),
        ),
      )
    catch {
      InvalidPagedCompositeArtifact(Digest | Fallback) => ()
      _ => fail("unexpected hostile fallback rejection")
    } noraise {
      _ => fail("hostile fallback identity was accepted")
    }
  }
}
