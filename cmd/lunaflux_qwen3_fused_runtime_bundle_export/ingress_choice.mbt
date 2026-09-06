///|
fn positive_nanoseconds(
  value : String,
) -> Int64 raise Qwen3FusedRuntimeBundleExportError {
  let result : Int64 = @strconv.from_str(value) catch {
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
    return bind_ingress_choice(args, choice, raw_args[evaluation_start + 2:])
  }
  let suffix_size = 11
  let suffix_start = raw_args.length() - suffix_size
  if suffix_start < 1 || raw_args[suffix_start] != "--ingress-comparison" {
    return (parse_arguments(raw_args), FullyFused)
  }
  let args = parse_arguments(raw_args[:suffix_start])
  let suffix = raw_args[suffix_start:]
  let samples = positive_int(suffix[4])
  let schedules : Array[@luna_fusion_strategy.FusionSchedule] = [
    FullyFused,
    ProducerSeparated,
    Unfused,
  ]
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
  let choice = @luna_fusion_strategy.select_fusion_schedule(
    alternatives,
    maximum_workspace_bytes=0L,
    maximum_local_bytes=0L,
  ) catch {
    _ => raise InvalidArguments
  }
  bind_ingress_choice(args, choice, suffix[5:])
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
  let partial : ModuleLaunchInput = {
    first_operation: args.ingress.first_operation + 1,
    grid_x: positive_int(partial_args[0]),
    grid_y: positive_int(partial_args[1]),
    grid_z: positive_int(partial_args[2]),
    block_x: positive_int(partial_args[3]),
    shared_memory_bytes: non_negative_int(partial_args[4]),
    module_path: partial_args[5],
  }
  (
    if choice == ProducerSeparated {
      { ..args, ingress: partial }
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
  if choice == Unfused {
    return modules
  }
  modules.push(
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=if choice == FullyFused {
        QwenQkvQkNormPositionedRopePagedKvWriteProductionV4
      } else {
        QwenQkNormPositionedRopePagedKvWriteProductionV2
      },
      first_operation=args.ingress.first_operation,
      operation_count=if choice == FullyFused { 4 } else { 3 },
      function_symbol=if choice == FullyFused {
        "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_v2"
      } else {
        "lunaflux_qwen_qknorm_rope_kvwrite_bf16_head128_production_v2"
      },
      launch=launch_dimensions(args.ingress),
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
  println("ingress_schedule=unfused")
  println(
    "runtime_sha256=" + @crypto.bytes_to_hex_string(@crypto.sha256(runtime)),
  )
}
