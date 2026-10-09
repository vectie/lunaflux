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
  let reference_flags = raw_args.filter(value => {
    value == "--prefill-reference-flashattention"
  })
  let flags = raw_args.filter(value => value == "--prefill-approximate-exp2")
  let wide_flags = raw_args.filter(value => {
    value == "--prefill-wide-approximate-exp2"
  })
  guard flags.length() <= 1 else { raise InvalidArguments }
  guard wide_flags.length() <= 1 else { raise InvalidArguments }
  guard reference_flags.length() <= 1 else { raise InvalidArguments }
  if !reference_flags.is_empty() &&
    (!flags.is_empty() || !wide_flags.is_empty()) {
    raise InvalidArguments
  }
  let stripped = raw_args.filter(value => {
    value != "--prefill-reference-flashattention" &&
    value != "--prefill-approximate-exp2" &&
    value != "--prefill-wide-approximate-exp2"
  })
  let (args, choice) = prepare_ingress_choice(stripped[:])
  if !reference_flags.is_empty() {
    validate_reference_prefill_choice(args, choice)
    return (
      args,
      choice,
      ReferenceBf16ProbabilityBaseTwo,
      ReferenceBf16ProbabilityBaseTwo,
    )
  }
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
    ReferenceBf16ProbabilityBaseTwo =>
      "lunaflux_attention_mixed_flashattention_bf16_exp2_v1"
  }
}

///|
/// The reference consumes raw CSR sequence offsets, not the compiler tile
/// worklist. Its two identical slots deliberately create only one executor.
fn validate_reference_prefill_choice(
  args : RuntimeBundleArguments,
  choice : @luna_fusion_strategy.FusionSchedule,
) -> Unit raise Qwen3FusedRuntimeBundleExportError {
  if choice == Unfused ||
    args.query_metadata ||
    args.prefill_query_tile_rows != 64 ||
    args.wide_query_tile_rows != 64 ||
    args.attention_prefill_wide_symbol !=
    prefill_function_symbol(ReferenceBf16ProbabilityBaseTwo) ||
    args.attention_prefill.module_path !=
    args.attention_prefill_wide.module_path {
    raise InvalidArguments
  }
  let legal_schedules : Array[
    @luna_cuda_reference_attention.ReferencePrefillSchedule,
  ] = [Query64Kv128, Query64Kv64]
  for launch in [args.attention_prefill, args.attention_prefill_wide] {
    if launch.grid_z != 1 ||
      launch.block_x != 128 ||
      !legal_schedules.any(schedule => {
        schedule.shared_memory_bytes() == launch.shared_memory_bytes
      }) {
      raise InvalidArguments
    }
  }
  if args.attention_prefill.grid_x != args.attention_prefill_wide.grid_x ||
    args.attention_prefill.grid_y != args.attention_prefill_wide.grid_y ||
    args.attention_prefill.shared_memory_bytes !=
    args.attention_prefill_wide.shared_memory_bytes {
    raise InvalidArguments
  }
}
