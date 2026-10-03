///|
test "dual score numerical ABI survives serialization and cannot downgrade" {
  let input = runtime_bundle_test_input(
    PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10,
    5,
    1,
    "attention_decode_tile_compiler_v1_dual_score_blockwise_f32_v2",
  )
  let bytes = export_reusable_fused_runtime_bundle(
    model_content_sha256="1".repeat(64),
    model_plan_sha256="2".repeat(64),
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    modules=[input],
  )
  let text = @utf8.decode(bytes)
  assert_true(
    text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v9\n"),
  )
  let runtime = admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  for
    module_ in [
      runtime.qwen_paged_attention_readonly_module(),
      runtime.qwen_paged_attention_decode_module(),
      runtime.qwen_paged_attention_decode_split_module(),
    ] {
    assert_eq(
      module_.unwrap().kernel_abi(),
      PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10,
    )
  }
  for version in [7, 8] {
    let downgraded = @utf8.encode(
      text.replace(
        old="schema=lunaflux-reusable-fused-runtime-bundle.v9",
        new="schema=lunaflux-reusable-fused-runtime-bundle.v\{version}",
      ),
    )
    try
      admit_digest_pinned_reusable_fused_runtime_bundle(
        downgraded,
        @crypto.bytes_to_hex_string(@crypto.sha256(downgraded)),
      )
    catch {
      InvalidFusedArtifact(KernelAbi) => ()
      _ => fail("unexpected downgrade error")
    } noraise {
      _ => fail("dual score ABI relabeled as another numerical law")
    }
  }
}
