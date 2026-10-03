///|
/// The numeric opt-in is peeled independently from ingress/decode choices.
/// It is export/startup policy, never a request-path heuristic.
fn prepare_prefill_numerical_choice(
  raw_args : ArrayView[String],
) -> (
  RuntimeBundleArguments,
  @luna_fusion_strategy.FusionSchedule,
  @luna_fused_artifact_admission.PrefillExponentialLaw,
  @luna_fused_artifact_admission.PrefillExponentialLaw,
) raise Qwen3FusedRuntimeBundleExportError {
  let flags = raw_args.filter(value => value == "--prefill-approximate-exp2")
  let wide_flags = raw_args.filter(value => {
    value == "--prefill-wide-approximate-exp2"
  })
  guard flags.length() <= 1 else { raise InvalidArguments }
  guard wide_flags.length() <= 1 else { raise InvalidArguments }
  let stripped = raw_args.filter(value => {
    value != "--prefill-approximate-exp2" &&
    value != "--prefill-wide-approximate-exp2"
  })
  let (args, choice) = prepare_ingress_choice(stripped[:])
  let law : @luna_fused_artifact_admission.PrefillExponentialLaw = if flags.is_empty() {
    StrictNaturalExponential
  } else {
    ApproximateBaseTwoExponential
  }
  let wide_law : @luna_fused_artifact_admission.PrefillExponentialLaw = if wide_flags.is_empty() {
    StrictNaturalExponential
  } else {
    ApproximateBaseTwoExponential
  }
  if (
      law == ApproximateBaseTwoExponential ||
      wide_law == ApproximateBaseTwoExponential
    ) &&
    choice == Unfused {
    raise InvalidArguments
  }
  if (wide_law == ApproximateBaseTwoExponential) !=
    args.attention_prefill_wide_symbol.contains("_exp2_v1") {
    raise InvalidArguments
  }
  (args, choice, law, wide_law)
}

///|
fn prefill_function_symbol(
  law : @luna_fused_artifact_admission.PrefillExponentialLaw,
) -> String {
  match law {
    StrictNaturalExponential => "lunaflux_attention_prefill_tile_compiler_v1"
    ApproximateBaseTwoExponential =>
      "lunaflux_attention_prefill_tile_compiler_exp2_v1"
  }
}
