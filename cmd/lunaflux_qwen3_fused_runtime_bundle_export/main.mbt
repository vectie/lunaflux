///|
fn launch_dimensions(
  input : ModuleLaunchInput,
) -> @launch_contract.AotLaunchDimensions raise Qwen3FusedRuntimeBundleExportError {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=input.grid_x,
    grid_y=input.grid_y,
    grid_z=input.grid_z,
    block_x=input.block_x,
    block_y=1,
    block_z=1,
    shared_memory_bytes=input.shared_memory_bytes,
  ) catch {
    _ => raise InvalidRuntime
  }
}

///|
async fn read_module(
  path : String,
) -> Bytes raise Qwen3FusedRuntimeBundleExportError {
  let bytes = @fs.read_file(path).binary() catch { _ => raise InvalidModule }
  if bytes.is_empty() || bytes.length() > 8388608 {
    raise InvalidModule
  }
  bytes
}

///|
async fn run(
  raw_args : ArrayView[String],
) -> Unit raise Qwen3FusedRuntimeBundleExportError {
  let (raw_args, connected_choice) = read_connected_choice(raw_args)
  let connected_context = raw_args
  let (raw_args, vendor_projection_exact_rows) = parse_vendor_projection_option(
    raw_args,
  )
  let (raw_args, attention_bucket_routes) = match raw_args {
    [.. rest, "--attention-buckets", path, digest] => {
      if !safe_absolute_file(path) || (digest != "-" && !lower_sha256(digest)) {
        raise InvalidArguments
      }
      let bytes = read_module(path)
      if bytes.length() > 1048576 {
        raise InvalidModule
      }
      (rest, Some(bytes))
    }
    _ => (raw_args, None)
  }
  let (
    args,
    ingress_choice,
    prefill_exponential_law,
    wide_prefill_exponential_law,
  ) = prepare_prefill_numerical_choice(raw_args)
  let (args, vendor_projection_exact_rows, connected_plan) = match
    connected_choice {
    None => (args, vendor_projection_exact_rows, None)
    Some(measured) => {
      let (args, rows, plan) = bind_connected_choice(
        connected_context, args, ingress_choice, prefill_exponential_law, measured,
      )
      (args, rows, Some(plan))
    }
  }
  let attention_bucket_routes = match (connected_choice, connected_plan) {
    (Some(measured), Some(plan)) => connected_route_bytes(measured, plan)
    _ => attention_bucket_routes
  }
  let residual_bytes = read_module(args.residual.module_path)
  if !selected_ingress_cut(args, ingress_choice).permits_readonly_attention() {
    if attention_bucket_routes is Some(_) || vendor_projection_exact_rows != 0 {
      raise InvalidArguments
    }
    write_unfused_ingress_bundle(args, residual_bytes)
    return
  }
  let ingress_bytes = read_module(args.ingress.module_path)
  let attention_prefill_bytes = read_module(args.attention_prefill.module_path)
  let attention_prefill_wide_bytes = read_module(
    args.attention_prefill_wide.module_path,
  )
  let attention_prefill_partial_bytes = read_module(
    args.attention_prefill_partial.module_path,
  )
  let attention_prefill_merge_bytes = if args.attention_prefill_merge.module_path ==
    args.attention_prefill_partial.module_path {
    attention_prefill_partial_bytes
  } else {
    read_module(args.attention_prefill_merge.module_path)
  }
  let attention_prefill_partitioned8_partial_bytes = read_module(
    args.attention_prefill_partitioned8_partial.module_path,
  )
  let attention_prefill_partitioned8_merge_bytes = if args.attention_prefill_partitioned8_merge.module_path ==
    args.attention_prefill_partitioned8_partial.module_path {
    attention_prefill_partitioned8_partial_bytes
  } else {
    read_module(args.attention_prefill_partitioned8_merge.module_path)
  }
  // Both entries have independently validated launch contracts. Deployment
  // owns their code association; do not compare opaque CUBINs byte by byte.
  let attention_bytes = read_module(args.attention_read.module_path)
  let target = runtime_device_target(args)
  let modules = selected_ingress_modules(
    args,
    ingress_choice,
    residual_bytes,
    ingress_bytes,
    row_variant?=match args.ingress_row_variant {
      Some(variant) => Some(read_ingress_row_variant(args, variant))
      None => None
    },
  )
  modules.push(
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=QwenPagedAttentionReadOnlyProductionV2,
      first_operation=args.attention_prefill.first_operation,
      operation_count=1,
      function_symbol=prefill_function_symbol(prefill_exponential_law),
      prefill_exponential_law~,
      allow_approximate_prefill_exponential=prefill_exponential_law !=
        StrictNaturalExponential,
      launch=launch_dimensions(args.attention_prefill),
      module_bytes=attention_prefill_bytes,
      module_label_sha256=runtime_module_locator_label(
        args.attention_prefill.module_path,
      ),
      query_tile_rows=args.prefill_query_tile_rows,
      query_metadata_version=if args.query_metadata { 1 } else { 0 },
    ) catch {
      _ => raise InvalidRuntime
    },
  )
  modules.push(
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=QwenPagedAttentionPrefillWideQueryProductionV5,
      first_operation=args.attention_prefill_wide.first_operation,
      operation_count=1,
      function_symbol=args.attention_prefill_wide_symbol,
      prefill_exponential_law=wide_prefill_exponential_law,
      allow_approximate_prefill_exponential=wide_prefill_exponential_law !=
        StrictNaturalExponential,
      launch=launch_dimensions(args.attention_prefill_wide),
      module_bytes=attention_prefill_wide_bytes,
      module_label_sha256=runtime_module_locator_label(
        args.attention_prefill_wide.module_path,
      ),
      query_tile_rows=args.wide_query_tile_rows,
      query_metadata_version=if args.query_metadata { 1 } else { 0 },
    ) catch {
      _ => raise InvalidRuntime
    },
  )
  // A different numerical algorithm cannot inherit old partitioned prefill
  // companions. Fresh decode observations may bind the new exact bundle scope.
  if prefill_exponential_law != ReferenceBf16ProbabilityBaseTwo {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedPartialProductionV4,
        first_operation=args.attention_prefill_partial.first_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1",
        launch=launch_dimensions(args.attention_prefill_partial),
        module_bytes=attention_prefill_partial_bytes,
        module_label_sha256=runtime_module_locator_label(
          args.attention_prefill_partial.module_path,
        ),
        query_tile_rows=args.partitioned_query_tile_rows,
        query_metadata_version=if args.query_metadata { 1 } else { 0 },
      ) catch {
        _ => raise InvalidRuntime
      },
    )
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedMergeProductionV4,
        first_operation=args.attention_prefill_merge.first_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1",
        launch=launch_dimensions(args.attention_prefill_merge),
        module_bytes=attention_prefill_merge_bytes,
        module_label_sha256=runtime_module_locator_label(
          args.attention_prefill_merge.module_path,
        ),
        query_tile_rows=args.partitioned_query_tile_rows,
      ) catch {
        _ => raise InvalidRuntime
      },
    )
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitioned8PartialProductionV6,
        first_operation=args.attention_prefill_partitioned8_partial.first_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1_p8",
        launch=launch_dimensions(args.attention_prefill_partitioned8_partial),
        module_bytes=attention_prefill_partitioned8_partial_bytes,
        module_label_sha256=runtime_module_locator_label(
          args.attention_prefill_partitioned8_partial.module_path,
        ),
        query_tile_rows=args.deep_partitioned_query_tile_rows,
        query_metadata_version=if args.query_metadata { 1 } else { 0 },
      ) catch {
        _ => raise InvalidRuntime
      },
    )
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitioned8MergeProductionV6,
        first_operation=args.attention_prefill_partitioned8_merge.first_operation,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1_p8",
        launch=launch_dimensions(args.attention_prefill_partitioned8_merge),
        module_bytes=attention_prefill_partitioned8_merge_bytes,
        module_label_sha256=runtime_module_locator_label(
          args.attention_prefill_partitioned8_merge.module_path,
        ),
        query_tile_rows=args.deep_partitioned_query_tile_rows,
      ) catch {
        _ => raise InvalidRuntime
      },
    )
  }
  modules.push(
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=if args.decode_execution is Some(_) {
        PagedAttentionReadOnlyExtendedDecodeProductionV11
      } else if args.dual_score_decode {
        PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10
      } else if args.contracting_decode {
        PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9
      } else if args.blockwise_decode {
        PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8
      } else {
        PagedAttentionReadOnlyCompilerDecodeProductionV7
      },
      first_operation=args.attention_read.first_operation,
      operation_count=1,
      function_symbol=if args.decode_execution is Some(descriptor) {
        args.decode_function_symbol.unwrap_or(descriptor.function_symbol())
      } else if args.dual_score_decode {
        "lunaflux_attention_decode_tile_compiler_v1_dual_score_blockwise_f32_v2"
      } else if args.contracting_decode {
        "lunaflux_attention_decode_tile_compiler_v1_blockwise_fma_f32_v3"
      } else if args.blockwise_decode {
        "lunaflux_attention_decode_tile_compiler_v1_blockwise_f32_v1"
      } else {
        "lunaflux_attention_decode_tile_compiler_v1"
      },
      launch=launch_dimensions(args.attention_read),
      module_bytes=attention_bytes,
      module_label_sha256=runtime_module_locator_label(
        args.attention_read.module_path,
      ),
      decode_execution?=args.decode_execution,
    ) catch {
      _ => raise InvalidRuntime
    },
  )
  let runtime = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256=args.model_content,
    model_plan_sha256=args.model_plan,
    target~,
    modules~,
    attention_bucket_routes?,
    vendor_projection_exact_rows~,
  ) catch {
    _ => raise InvalidRuntime
  }
  let route_scope = @luna_fused_artifact_admission.reusable_fused_runtime_attention_route_scope(
    model_content_sha256=args.model_content,
    model_plan_sha256=args.model_plan,
    target~,
    modules~,
  ) catch {
    _ => raise InvalidRuntime
  }
  if attention_bucket_routes is Some(bytes) {
    let table = @luna_execution_graph_strategy.build_execution_graph_bucket_table(
      1048576, 1048576, 1048576, 65536,
    ) catch {
      _ => raise InvalidRuntime
    }
    let selected = @luna_attention_tuning.parse_attention_bucket_routes(
      bytes,
      scope=route_scope,
      table,
      baseline=1,
      if prefill_exponential_law == ReferenceBf16ProbabilityBaseTwo {
        [1, 3, 4]
      } else {
        [1, 2, 3, 4, 5, 6, 7]
      },
    ) catch {
      _ => raise InvalidRuntime
    }
    for slot, candidate in selected.bucket_candidates() {
      let phase = (table.bucket_at(slot) catch { _ => raise InvalidRuntime }).phase()
      if ((candidate == 3 || candidate == 4) && phase != DecodeGraph) ||
        (
          (candidate == 2 || candidate == 5 || candidate == 6) &&
          phase == DecodeGraph
        ) {
        raise InvalidRuntime
      }
    }
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
  if connected_plan is Some(plan) {
    println("connected_selection_basis=measured-complete-continuation")
    println("connected_plan_id=\{plan}")
  }
  println("ingress_selection_basis=" + args.ingress_selection.canonical_name())
  println("attention_route_scope=" + route_scope)
  println(
    "ingress_schedule=" +
    (if ingress_choice == FullyFused { "full" } else { "producer-separated" }),
  )
  // Legacy field name; this is a measurement/plan label, not a byte checksum.
  println("runtime_sha256=" + route_scope)
  println("runtime_identity_kind=attention-route-scope-label")
  println("payload_hashing=false")
  println("payload_checksum_verified=0")
  // Report the identities already in the typed bundle, not another SHA pass
  // over all module payloads for logging.
  for module_ in modules {
    println(
      "module_symbol=" +
      module_.function_symbol() +
      " module_identity=" +
      module_.module_digest().as_hex(),
    )
  }
}

///|
async fn main {
  try! run(@env.args())
}
