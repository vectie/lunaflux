///|
test "step rotary selection propagates the exact cached consumer ABI" {
  let raw = runtime_bundle_test_arguments("5")
  raw.append(["--ingress-evaluate", "full", "--step-rotary-cache"])
  let (args, choice) = prepare_ingress_choice(raw)
  assert_true(args.step_rotary_cache)
  let modules = selected_ingress_modules(
    args, choice, b"residual", b"cached-ingress",
  )
  let bytes = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256=args.model_content,
    model_plan_sha256=args.model_plan,
    target=runtime_device_target(args),
    modules~,
  )
  let text = @utf8.decode(bytes)
  assert_true(
    text.contains("abi=attention-ingress-prepared-rotary-production-v11\n"),
  )
  assert_true(
    text.contains(
      "function_symbol=lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3\n",
    ),
  )
  for
    suffix in [
      ["--step-rotary-cache", "--step-rotary-cache"],
      ["--step-rotary-cache"],
    ] {
    let invalid = runtime_bundle_test_arguments("5")
    invalid.append(["--ingress-evaluate", "unfused"])
    invalid.append(suffix)
    try prepare_ingress_choice(invalid) catch {
      InvalidArguments => ()
      _ => fail("unexpected cache route selection failure")
    } noraise {
      _ =>
        fail("cached consumer selected without its prepared full-ingress route")
    }
  }
}
