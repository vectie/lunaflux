///|
test "contracting decode has a distinct round trip and cannot downgrade its schema" {
  let input = runtime_bundle_test_input(
    PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9,
    5,
    1,
    "attention_decode_tile_compiler_v1_blockwise_fma_f32_v3",
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
    text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v8\n"),
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
      PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9,
    )
  }
  let downgraded = @utf8.encode(
    text.replace(
      old="schema=lunaflux-reusable-fused-runtime-bundle.v8",
      new="schema=lunaflux-reusable-fused-runtime-bundle.v7",
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
    _ => fail("contracting ABI mislabeled as strict schema")
  }
}
