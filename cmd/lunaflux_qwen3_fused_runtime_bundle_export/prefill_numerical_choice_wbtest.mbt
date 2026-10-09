///|
fn reference_prefill_test_args() -> Array[String] {
  let args = runtime_bundle_test_arguments("5")
  // Preserve the positional ABI while replacing both equivalent prefill slots.
  args[18] = "64"
  args[21] = "128"
  args[22] = "81920"
  args[25] = "64"
  args[28] = "128"
  args[29] = "81920"
  args[30] = "lunaflux_attention_mixed_flashattention_bf16_exp2_v1"
  args[31] = args[23]
  args[67] = "64"
  args.append([
    "--ingress-evaluate", "full", "--prefill-reference-flashattention",
  ])
  args
}

///|
test "reference prefill requires explicit numeric law and raw matching phase slots" {
  let args = reference_prefill_test_args()
  let (_, _, law, wide) = prepare_prefill_numerical_choice(args)
  assert_eq(law, @luna_fused_artifact_admission.ReferenceBf16ProbabilityBaseTwo)
  assert_eq(wide, law)
  for
    extra in [
      "--prefill-approximate-exp2", "--prefill-reference-flashattention", "--query-metadata-v1",
    ] {
    let bad = reference_prefill_test_args()
    bad.push(extra)
    try prepare_prefill_numerical_choice(bad) catch {
      InvalidArguments => ()
      _ => fail("wrong reference option failure")
    } noraise {
      _ => fail("incompatible reference option accepted")
    }
  }
  for
    (index, value) in [
      (21, "256"),
      (29, "81919"),
      (31, "/tmp/other.cubin"),
      (67, "32"),
    ] {
    let bad = reference_prefill_test_args()
    bad[index] = value
    try prepare_prefill_numerical_choice(bad) catch {
      InvalidArguments => ()
      _ => fail("wrong reference geometry failure")
    } noraise {
      _ => fail("incompatible reference geometry accepted")
    }
  }
}

///|
test "reference KV64 geometry is admitted only with identical writer slots" {
  let args = reference_prefill_test_args()
  args[22] = "49152"
  args[29] = "49152"
  let (_, _, law, wide) = prepare_prefill_numerical_choice(args)
  assert_eq(law, @luna_fused_artifact_admission.ReferenceBf16ProbabilityBaseTwo)
  assert_eq(wide, law)
  for shared in ["81920", "32768", "49151"] {
    args[29] = shared
    try prepare_prefill_numerical_choice(args) catch {
      InvalidArguments => ()
      _ => fail("wrong reference schedule failure")
    } noraise {
      _ => fail("inconsistent or unqualified reference schedule accepted")
    }
  }
}

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
