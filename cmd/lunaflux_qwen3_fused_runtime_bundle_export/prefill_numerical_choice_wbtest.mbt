///|
test "bundle exp2 opt-in is distinct from ingress and decode choices" {
  let strict = runtime_bundle_test_arguments("5")
  let (_, _, strict_law, strict_wide_law) = prepare_prefill_numerical_choice(
    strict,
  )
  assert_eq(
    strict_wide_law,
    @luna_fused_artifact_admission.StrictNaturalExponential,
  )
  assert_eq(
    prefill_function_symbol(strict_law),
    "lunaflux_attention_prefill_tile_compiler_v1",
  )
  let experimental = runtime_bundle_test_arguments("5")
  for index, value in experimental {
    if value == "lunaflux_attention_prefill_tile_compiler_v1_c314" {
      experimental[index] = "lunaflux_attention_prefill_tile_compiler_exp2_v1"
    }
  }
  try prepare_prefill_numerical_choice(experimental) catch {
    InvalidArguments => ()
    _ => fail("unexpected implicit approximate admission error")
  } noraise {
    _ => fail("approximate symbol accepted without explicit option")
  }
  experimental.append([
    "--ingress-evaluate", "full", "--prefill-approximate-exp2", "--prefill-wide-approximate-exp2",
  ])
  let (_, choice, law, wide_law) = prepare_prefill_numerical_choice(
    experimental,
  )
  assert_eq(choice, @luna_fusion_strategy.FullyFused)
  assert_eq(law, @luna_fused_artifact_admission.ApproximateBaseTwoExponential)
  assert_eq(
    wide_law,
    @luna_fused_artifact_admission.ApproximateBaseTwoExponential,
  )
  assert_eq(
    prefill_function_symbol(law),
    "lunaflux_attention_prefill_tile_compiler_exp2_v1",
  )
  experimental.push("--prefill-approximate-exp2")
  try prepare_prefill_numerical_choice(experimental) catch {
    InvalidArguments => ()
    _ => fail("unexpected duplicate approximate option error")
  } noraise {
    _ => fail("duplicate approximate option accepted")
  }
}

///|
test "baseline exponential permission does not relabel frozen strict wide artifacts" {
  let baseline_only = runtime_bundle_test_arguments("5")
  baseline_only.append([
    "--ingress-evaluate", "full", "--prefill-approximate-exp2",
  ])
  let (args, _, law, wide_law) = prepare_prefill_numerical_choice(baseline_only)
  assert_eq(law, @luna_fused_artifact_admission.ApproximateBaseTwoExponential)
  assert_eq(wide_law, @luna_fused_artifact_admission.StrictNaturalExponential)
  assert_eq(
    args.attention_prefill_wide_symbol,
    "lunaflux_attention_prefill_tile_compiler_v1_c314",
  )
  baseline_only.push("--prefill-wide-approximate-exp2")
  try prepare_prefill_numerical_choice(baseline_only) catch {
    InvalidArguments => ()
    _ => fail("unexpected unqualified wide permission error")
  } noraise {
    _ => fail("strict wide artifact implicitly promoted")
  }
}
