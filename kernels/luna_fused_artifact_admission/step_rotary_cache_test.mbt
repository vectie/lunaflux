///|
test "prepared rotary ABI is distinct and cannot downgrade to an uncached bundle" {
  let input = runtime_bundle_test_input(
    AttentionIngressPreparedRotaryProductionV11,
    2,
    4,
    "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3",
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
    text.has_prefix("schema=lunaflux-reusable-fused-runtime-bundle.v10\n"),
  )
  let runtime = admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  assert_eq(
    runtime.qwen_full_attention_ingress_module().unwrap().kernel_abi(),
    AttentionIngressPreparedRotaryProductionV11,
  )
  let downgraded = @utf8.encode(text.replace(old="bundle.v10", new="bundle.v9"))
  try
    admit_digest_pinned_reusable_fused_runtime_bundle(
      downgraded,
      @crypto.bytes_to_hex_string(@crypto.sha256(downgraded)),
    )
  catch {
    InvalidFusedArtifact(KernelAbi) => ()
    _ => fail("unexpected cache ABI downgrade error")
  } noraise {
    _ => fail("cached consumer relabeled with an uncached bundle schema")
  }
}
