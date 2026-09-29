///|
fn positive_nanoseconds(
  value : String,
) -> Int64 raise Qwen3FusedRuntimeBundleExportError {
  let result : Int64 = @string.parse_int64(value) catch {
    _ => raise InvalidArguments
  }
  if result <= 0L || result.to_string() != value {
    raise InvalidArguments
  }
  result
}

///|
fn prepare_ingress_choice(
  raw_args : ArrayView[String],
) -> (RuntimeBundleArguments, @luna_fusion_strategy.FusionSchedule) raise Qwen3FusedRuntimeBundleExportError {
  // Global ABI options are independent of the optional ingress experiment.
  // Peel the global suffix before interpreting that experiment's fixed arity.
  if raw_args is [.. rest, "--query-metadata-v1"] {
    let (args, choice) = prepare_ingress_choice(rest)
    if args.query_metadata {
      raise InvalidArguments
    }
    return ({ ..args, query_metadata: true, }, choice)
  }
  if raw_args is [.. rest, "--ingress-query-tile", value] {
    let rows = positive_int(value)
    let (args, choice) = prepare_ingress_choice(rest)
    guard rows <= 1048576 &&
      args.ingress_query_tile_rows is None &&
      choice == FullyFused else {
      raise InvalidArguments
    }
    return ({ ..args, ingress_query_tile_rows: Some(rows), }, choice)
  }
  if raw_args is [.. rest, "--ingress-evaluate", name] {
    let args = parse_arguments(rest)
    let choice : @luna_fusion_strategy.FusionSchedule = match name {
      "full" => FullyFused
      "unfused" => Unfused
      _ => raise InvalidArguments
    }
    return ({ ..args, ingress_selection: ExplicitEvaluation, }, choice)
  }
  let evaluation_start = raw_args.length() - 8
  if evaluation_start > 0 && raw_args[evaluation_start] == "--ingress-evaluate" {
    let args = parse_arguments(raw_args[:evaluation_start])
    let choice : @luna_fusion_strategy.FusionSchedule = match
      raw_args[evaluation_start + 1] {
      "full" => FullyFused
      "partial" => ProducerSeparated
      "unfused" => Unfused
      _ => raise InvalidArguments
    }
    return bind_ingress_choice(
      { ..args, ingress_selection: ExplicitEvaluation, },
      choice,
      raw_args[evaluation_start + 2:],
    )
  }
  let pair_start = raw_args.length() - 10
  let paired = pair_start > 0 && raw_args[pair_start] == "--ingress-pair"
  let suffix_size = if paired { 10 } else { 11 }
  let suffix_start = raw_args.length() - suffix_size
  if suffix_start < 1 ||
    (!paired && raw_args[suffix_start] != "--ingress-comparison") {
    // No observation does not imply maximal fusion is fastest. Preserve all
    // semantic stages and their KV-write owner unless a cut is explicitly
    // evaluated or selected from comparable full-span measurements.
    return (parse_arguments(raw_args), Unfused)
  }
  let args = parse_arguments(raw_args[:suffix_start])
  let suffix = raw_args[suffix_start:]
  let schedules : Array[@luna_fusion_strategy.FusionSchedule] = if paired {
    [FullyFused, ProducerSeparated]
  } else {
    [FullyFused, ProducerSeparated, Unfused]
  }
  let samples = positive_int(suffix[schedules.length() + 1])
  let alternatives = schedules.mapi((index, schedule) => {
    @luna_fusion_strategy.FusionAlternative::new(
      schedule~,
      cost=@luna_fusion_strategy.FusionSpanCost::measured(
        median_nanoseconds=positive_nanoseconds(suffix[index + 1]),
        samples~,
      ),
      workspace_bytes=0L,
      local_bytes=0L,
    )
  }) catch {
    _ => raise InvalidArguments
  }
  let cut = @luna_fusion_plan.select_attention_ingress_cut(
    alternatives,
    qkv_operation=args.ingress.first_operation,
    maximum_workspace_bytes=0L,
    maximum_local_bytes=0L,
  ) catch {
    _ => raise InvalidArguments
  }
  let choice = cut.schedule()
  bind_ingress_choice(
    { ..args, ingress_selection: MeasuredComparison, },
    choice,
    suffix[schedules.length() + 2:],
  )
}

///|
fn bind_ingress_choice(
  args : RuntimeBundleArguments,
  choice : @luna_fusion_strategy.FusionSchedule,
  partial_args : ArrayView[String],
) -> (RuntimeBundleArguments, @luna_fusion_strategy.FusionSchedule) raise Qwen3FusedRuntimeBundleExportError {
  if partial_args.length() != 6 || !safe_absolute_file(partial_args[5]) {
    raise InvalidArguments
  }
  let cut = @luna_fusion_plan.AttentionIngressCutPlan::new(
    ProducerSeparated,
    qkv_operation=args.ingress.first_operation,
  ) catch {
    _ => raise InvalidArguments
  }
  let partial : ModuleLaunchInput = {
    first_operation: cut.module_span().unwrap().0,
    grid_x: positive_int(partial_args[0]),
    grid_y: positive_int(partial_args[1]),
    grid_z: positive_int(partial_args[2]),
    block_x: positive_int(partial_args[3]),
    shared_memory_bytes: non_negative_int(partial_args[4]),
    module_path: partial_args[5],
  }
  (
    if choice == ProducerSeparated {
      { ..args, ingress: partial, }
    } else {
      args
    },
    choice,
  )
}

///|
fn selected_ingress_modules(
  args : RuntimeBundleArguments,
  choice : @luna_fusion_strategy.FusionSchedule,
  residual_bytes : Bytes,
  ingress_bytes : Bytes,
) -> Array[@luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput] raise Qwen3FusedRuntimeBundleExportError {
  let modules = [
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=ResidualRmsNormProductionV2,
      first_operation=args.residual.first_operation,
      operation_count=2,
      function_symbol="lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2",
      launch=launch_dimensions(args.residual),
      module_bytes=residual_bytes,
    ),
  ] catch {
    _ => raise InvalidRuntime
  }
  let cut = selected_ingress_cut(args, choice)
  guard cut.module_span() is Some((first_operation, operation_count)) else {
    return modules
  }
  modules.push(
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=if choice == FullyFused {
        QwenQkvQkNormPositionedRopePagedKvWriteProductionV4
      } else {
        QwenQkNormPositionedRopePagedKvWriteProductionV2
      },
      first_operation~,
      operation_count~,
      function_symbol=if choice == FullyFused {
        "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_v2"
      } else {
        "lunaflux_qwen_qknorm_rope_kvwrite_bf16_head128_production_v2"
      },
      launch=launch_dimensions(args.ingress),
      query_tile_rows?=args.ingress_query_tile_rows,
      module_bytes=ingress_bytes,
    ) catch {
      _ => raise InvalidRuntime
    },
  )
  modules
}

///|
async fn write_unfused_ingress_bundle(
  args : RuntimeBundleArguments,
  residual_bytes : Bytes,
) -> Unit raise Qwen3FusedRuntimeBundleExportError {
  let runtime = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256=args.model_content,
    model_plan_sha256=args.model_plan,
    target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    modules=selected_ingress_modules(args, Unfused, residual_bytes, b""),
  ) catch {
    _ => raise InvalidRuntime
  }
  @fs.write_file(
    args.output_path,
    runtime,
    create_mode=CreateNew,
    permission=0o444,
    sync=Full,
  ) catch {
    _ => raise WriteFailed
  }
  println("schema=lunaflux-qwen3-fused-runtime-bundle-export.v5")
  println("ingress_selection_basis=" + args.ingress_selection.canonical_name())
  println("ingress_schedule=unfused")
  println(
    "runtime_sha256=" + @crypto.bytes_to_hex_string(@crypto.sha256(runtime)),
  )
}
