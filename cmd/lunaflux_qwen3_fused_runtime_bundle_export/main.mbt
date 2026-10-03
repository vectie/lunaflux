///|
suberror Qwen3FusedRuntimeBundleExportError {
  InvalidArguments
  InvalidModule
  InvalidRuntime
  WriteFailed
} derive(Debug, Eq)

///|
priv struct ModuleLaunchInput {
  first_operation : Int
  grid_x : Int
  grid_y : Int
  grid_z : Int
  block_x : Int
  shared_memory_bytes : Int
  module_path : String
}

///|
priv struct RuntimeBundleArguments {
  device_target : @catalog.DeviceTarget?
  ingress_selection : IngressSelectionBasis
  ingress_query_tile_rows : Int?
  ingress_heads_per_cta : Int?
  blockwise_decode : Bool
  contracting_decode : Bool
  dual_score_decode : Bool
  decode_execution : @luna_fused_artifact_admission.DecodeExecutionDescriptor?
  decode_function_symbol : String?
  step_rotary_cache : Bool
  query_metadata : Bool
  model_content : String
  model_plan : String
  residual : ModuleLaunchInput
  ingress : ModuleLaunchInput
  attention_prefill : ModuleLaunchInput
  attention_prefill_wide : ModuleLaunchInput
  attention_prefill_wide_symbol : String
  attention_prefill_partial : ModuleLaunchInput
  attention_prefill_merge : ModuleLaunchInput
  attention_prefill_partitioned8_partial : ModuleLaunchInput
  attention_prefill_partitioned8_merge : ModuleLaunchInput
  attention_read : ModuleLaunchInput
  prefill_query_tile_rows : Int
  wide_query_tile_rows : Int
  partitioned_query_tile_rows : Int
  deep_partitioned_query_tile_rows : Int
  output_path : String
}

///|
fn lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn positive_int(value : String) -> Int raise Qwen3FusedRuntimeBundleExportError {
  let parsed : Int = @string.parse_int(value) catch {
    _ => raise InvalidArguments
  }
  if parsed <= 0 || parsed.to_string() != value {
    raise InvalidArguments
  }
  parsed
}

///|
fn non_negative_int(
  value : String,
) -> Int raise Qwen3FusedRuntimeBundleExportError {
  if value == "0" {
    return 0
  }
  positive_int(value)
}

///|
fn safe_absolute_file(path : String) -> Bool {
  path.length() > 1 &&
  path.length() <= 4096 &&
  path.has_prefix("/") &&
  !path.contains("//") &&
  !path.contains("/./") &&
  !path.contains("/../") &&
  !path.has_suffix("/.") &&
  !path.has_suffix("/..")
}

///|
fn safe_function_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 512 &&
  !symbol.contains("\n") &&
  !symbol.contains("\r")
}

///|
fn parse_arguments(
  args : ArrayView[String],
) -> RuntimeBundleArguments raise Qwen3FusedRuntimeBundleExportError {
  let (args, query_metadata) = match args {
    [.. rest, "--query-metadata-v1"] => (rest, true)
    _ => (args, false)
  }
  let (
    model_content,
    model_plan,
    residual_operation,
    residual_grid_x,
    residual_grid_y,
    residual_grid_z,
    residual_block_x,
    residual_shared,
    residual_module,
    ingress_operation,
    ingress_grid_x,
    ingress_grid_y,
    ingress_grid_z,
    ingress_block_x,
    ingress_shared,
    ingress_module,
    prefill_operation,
    prefill_grid_x,
    prefill_grid_y,
    prefill_grid_z,
    prefill_block_x,
    prefill_shared,
    prefill_module,
    prefill_wide_operation,
    prefill_wide_grid_x,
    prefill_wide_grid_y,
    prefill_wide_grid_z,
    prefill_wide_block_x,
    prefill_wide_shared,
    prefill_wide_symbol,
    prefill_wide_module,
    prefill_partial_operation,
    prefill_partial_grid_x,
    prefill_partial_grid_y,
    prefill_partial_grid_z,
    prefill_partial_block_x,
    prefill_partial_shared,
    prefill_partial_module,
    prefill_merge_operation,
    prefill_merge_grid_x,
    prefill_merge_grid_y,
    prefill_merge_grid_z,
    prefill_merge_block_x,
    prefill_merge_shared,
    prefill_merge_module,
    prefill_partitioned8_partial_operation,
    prefill_partitioned8_partial_grid_x,
    prefill_partitioned8_partial_grid_y,
    prefill_partitioned8_partial_grid_z,
    prefill_partitioned8_partial_block_x,
    prefill_partitioned8_partial_shared,
    prefill_partitioned8_partial_module,
    prefill_partitioned8_merge_operation,
    prefill_partitioned8_merge_grid_x,
    prefill_partitioned8_merge_grid_y,
    prefill_partitioned8_merge_grid_z,
    prefill_partitioned8_merge_block_x,
    prefill_partitioned8_merge_shared,
    prefill_partitioned8_merge_module,
    attention_operation,
    attention_grid_x,
    attention_grid_y,
    attention_grid_z,
    attention_block_x,
    attention_shared,
    attention_module,
    prefill_query_tile_rows,
    wide_query_tile_rows,
    partitioned_query_tile_rows,
    deep_partitioned_query_tile_rows,
    output_path,
  ) = match args {
    [
      _,
      model_content,
      model_plan,
      residual_operation,
      residual_grid_x,
      residual_grid_y,
      residual_grid_z,
      residual_block_x,
      residual_shared,
      residual_module,
      ingress_operation,
      ingress_grid_x,
      ingress_grid_y,
      ingress_grid_z,
      ingress_block_x,
      ingress_shared,
      ingress_module,
      prefill_operation,
      prefill_grid_x,
      prefill_grid_y,
      prefill_grid_z,
      prefill_block_x,
      prefill_shared,
      prefill_module,
      prefill_wide_operation,
      prefill_wide_grid_x,
      prefill_wide_grid_y,
      prefill_wide_grid_z,
      prefill_wide_block_x,
      prefill_wide_shared,
      prefill_wide_symbol,
      prefill_wide_module,
      prefill_partial_operation,
      prefill_partial_grid_x,
      prefill_partial_grid_y,
      prefill_partial_grid_z,
      prefill_partial_block_x,
      prefill_partial_shared,
      prefill_partial_module,
      prefill_merge_operation,
      prefill_merge_grid_x,
      prefill_merge_grid_y,
      prefill_merge_grid_z,
      prefill_merge_block_x,
      prefill_merge_shared,
      prefill_merge_module,
      prefill_partitioned8_partial_operation,
      prefill_partitioned8_partial_grid_x,
      prefill_partitioned8_partial_grid_y,
      prefill_partitioned8_partial_grid_z,
      prefill_partitioned8_partial_block_x,
      prefill_partitioned8_partial_shared,
      prefill_partitioned8_partial_module,
      prefill_partitioned8_merge_operation,
      prefill_partitioned8_merge_grid_x,
      prefill_partitioned8_merge_grid_y,
      prefill_partitioned8_merge_grid_z,
      prefill_partitioned8_merge_block_x,
      prefill_partitioned8_merge_shared,
      prefill_partitioned8_merge_module,
      attention_operation,
      attention_grid_x,
      attention_grid_y,
      attention_grid_z,
      attention_block_x,
      attention_shared,
      attention_module,
      prefill_query_tile_rows,
      wide_query_tile_rows,
      partitioned_query_tile_rows,
      deep_partitioned_query_tile_rows,
      output_path,
    ] =>
      (
        model_content, model_plan, residual_operation, residual_grid_x, residual_grid_y,
        residual_grid_z, residual_block_x, residual_shared, residual_module, ingress_operation,
        ingress_grid_x, ingress_grid_y, ingress_grid_z, ingress_block_x, ingress_shared,
        ingress_module, prefill_operation, prefill_grid_x, prefill_grid_y, prefill_grid_z,
        prefill_block_x, prefill_shared, prefill_module, prefill_wide_operation,
        prefill_wide_grid_x, prefill_wide_grid_y, prefill_wide_grid_z, prefill_wide_block_x,
        prefill_wide_shared, prefill_wide_symbol, prefill_wide_module, prefill_partial_operation,
        prefill_partial_grid_x, prefill_partial_grid_y, prefill_partial_grid_z, prefill_partial_block_x,
        prefill_partial_shared, prefill_partial_module, prefill_merge_operation,
        prefill_merge_grid_x, prefill_merge_grid_y, prefill_merge_grid_z, prefill_merge_block_x,
        prefill_merge_shared, prefill_merge_module, prefill_partitioned8_partial_operation,
        prefill_partitioned8_partial_grid_x, prefill_partitioned8_partial_grid_y,
        prefill_partitioned8_partial_grid_z, prefill_partitioned8_partial_block_x,
        prefill_partitioned8_partial_shared, prefill_partitioned8_partial_module,
        prefill_partitioned8_merge_operation, prefill_partitioned8_merge_grid_x,
        prefill_partitioned8_merge_grid_y, prefill_partitioned8_merge_grid_z, prefill_partitioned8_merge_block_x,
        prefill_partitioned8_merge_shared, prefill_partitioned8_merge_module, attention_operation,
        attention_grid_x, attention_grid_y, attention_grid_z, attention_block_x,
        attention_shared, attention_module, prefill_query_tile_rows, wide_query_tile_rows,
        partitioned_query_tile_rows, deep_partitioned_query_tile_rows, output_path,
      )
    _ => raise InvalidArguments
  }
  if !lower_sha256(model_content) ||
    !lower_sha256(model_plan) ||
    !safe_absolute_file(residual_module) ||
    !safe_absolute_file(ingress_module) ||
    !safe_absolute_file(prefill_module) ||
    !safe_absolute_file(prefill_wide_module) ||
    !safe_function_symbol(prefill_wide_symbol) ||
    !safe_absolute_file(prefill_partial_module) ||
    !safe_absolute_file(prefill_merge_module) ||
    !safe_absolute_file(prefill_partitioned8_partial_module) ||
    !safe_absolute_file(prefill_partitioned8_merge_module) ||
    !safe_absolute_file(attention_module) ||
    !safe_absolute_file(output_path) {
    raise InvalidArguments
  }
  let residual : ModuleLaunchInput = {
    first_operation: non_negative_int(residual_operation),
    grid_x: positive_int(residual_grid_x),
    grid_y: positive_int(residual_grid_y),
    grid_z: positive_int(residual_grid_z),
    block_x: positive_int(residual_block_x),
    shared_memory_bytes: non_negative_int(residual_shared),
    module_path: residual_module,
  }
  let ingress : ModuleLaunchInput = {
    first_operation: non_negative_int(ingress_operation),
    grid_x: positive_int(ingress_grid_x),
    grid_y: positive_int(ingress_grid_y),
    grid_z: positive_int(ingress_grid_z),
    block_x: positive_int(ingress_block_x),
    shared_memory_bytes: non_negative_int(ingress_shared),
    module_path: ingress_module,
  }
  let attention_prefill_partial : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_partial_operation),
    grid_x: positive_int(prefill_partial_grid_x),
    grid_y: positive_int(prefill_partial_grid_y),
    grid_z: positive_int(prefill_partial_grid_z),
    block_x: positive_int(prefill_partial_block_x),
    shared_memory_bytes: non_negative_int(prefill_partial_shared),
    module_path: prefill_partial_module,
  }
  let attention_prefill : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_operation),
    grid_x: positive_int(prefill_grid_x),
    grid_y: positive_int(prefill_grid_y),
    grid_z: positive_int(prefill_grid_z),
    block_x: positive_int(prefill_block_x),
    shared_memory_bytes: non_negative_int(prefill_shared),
    module_path: prefill_module,
  }
  let attention_prefill_wide : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_wide_operation),
    grid_x: positive_int(prefill_wide_grid_x),
    grid_y: positive_int(prefill_wide_grid_y),
    grid_z: positive_int(prefill_wide_grid_z),
    block_x: positive_int(prefill_wide_block_x),
    shared_memory_bytes: non_negative_int(prefill_wide_shared),
    module_path: prefill_wide_module,
  }
  let attention_prefill_merge : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_merge_operation),
    grid_x: positive_int(prefill_merge_grid_x),
    grid_y: positive_int(prefill_merge_grid_y),
    grid_z: positive_int(prefill_merge_grid_z),
    block_x: positive_int(prefill_merge_block_x),
    shared_memory_bytes: non_negative_int(prefill_merge_shared),
    module_path: prefill_merge_module,
  }
  let attention_prefill_partitioned8_partial : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_partitioned8_partial_operation),
    grid_x: positive_int(prefill_partitioned8_partial_grid_x),
    grid_y: positive_int(prefill_partitioned8_partial_grid_y),
    grid_z: positive_int(prefill_partitioned8_partial_grid_z),
    block_x: positive_int(prefill_partitioned8_partial_block_x),
    shared_memory_bytes: non_negative_int(prefill_partitioned8_partial_shared),
    module_path: prefill_partitioned8_partial_module,
  }
  let attention_prefill_partitioned8_merge : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_partitioned8_merge_operation),
    grid_x: positive_int(prefill_partitioned8_merge_grid_x),
    grid_y: positive_int(prefill_partitioned8_merge_grid_y),
    grid_z: positive_int(prefill_partitioned8_merge_grid_z),
    block_x: positive_int(prefill_partitioned8_merge_block_x),
    shared_memory_bytes: non_negative_int(prefill_partitioned8_merge_shared),
    module_path: prefill_partitioned8_merge_module,
  }
  let attention_read : ModuleLaunchInput = {
    first_operation: non_negative_int(attention_operation),
    grid_x: positive_int(attention_grid_x),
    grid_y: positive_int(attention_grid_y),
    grid_z: positive_int(attention_grid_z),
    block_x: positive_int(attention_block_x),
    shared_memory_bytes: non_negative_int(attention_shared),
    module_path: attention_module,
  }
  if attention_read.first_operation != ingress.first_operation + 3 {
    raise InvalidArguments
  }
  if attention_prefill.first_operation != attention_read.first_operation ||
    attention_prefill_wide.first_operation != attention_read.first_operation ||
    attention_prefill_partial.first_operation != attention_read.first_operation ||
    attention_prefill_merge.first_operation != attention_read.first_operation ||
    attention_prefill_partitioned8_partial.first_operation !=
    attention_read.first_operation ||
    attention_prefill_partitioned8_merge.first_operation !=
    attention_read.first_operation {
    raise InvalidArguments
  }
  {
    model_content,
    model_plan,
    device_target: None,
    ingress_selection: ConservativeUnmeasured,
    ingress_query_tile_rows: None,
    ingress_heads_per_cta: None,
    blockwise_decode: false,
    contracting_decode: false,
    dual_score_decode: false,
    decode_execution: None,
    decode_function_symbol: None,
    step_rotary_cache: false,
    residual,
    ingress,
    attention_prefill,
    attention_prefill_wide,
    attention_prefill_wide_symbol: prefill_wide_symbol,
    attention_prefill_partial,
    attention_prefill_merge,
    attention_prefill_partitioned8_partial,
    attention_prefill_partitioned8_merge,
    attention_read,
    prefill_query_tile_rows: positive_int(prefill_query_tile_rows),
    query_metadata,
    wide_query_tile_rows: positive_int(wide_query_tile_rows),
    partitioned_query_tile_rows: positive_int(partitioned_query_tile_rows),
    deep_partitioned_query_tile_rows: positive_int(
      deep_partitioned_query_tile_rows,
    ),
    output_path,
  }
}

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
  let (raw_args, attention_bucket_routes) = match raw_args {
    [.. rest, "--attention-buckets", path, digest] => {
      if !safe_absolute_file(path) || !lower_sha256(digest) {
        raise InvalidArguments
      }
      let bytes = read_module(path)
      if bytes.length() > 1048576 ||
        @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) != digest {
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
  let residual_bytes = read_module(args.residual.module_path)
  if !selected_ingress_cut(args, ingress_choice).permits_readonly_attention() {
    if attention_bucket_routes is Some(_) {
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
  let attention_prefill_merge_bytes = read_module(
    args.attention_prefill_merge.module_path,
  )
  let attention_prefill_partitioned8_partial_bytes = read_module(
    args.attention_prefill_partitioned8_partial.module_path,
  )
  let attention_prefill_partitioned8_merge_bytes = read_module(
    args.attention_prefill_partitioned8_merge.module_path,
  )
  if @crypto.sha256(attention_prefill_partial_bytes) !=
    @crypto.sha256(attention_prefill_merge_bytes) ||
    @crypto.sha256(attention_prefill_partitioned8_partial_bytes) !=
    @crypto.sha256(attention_prefill_partitioned8_merge_bytes) {
    raise InvalidModule
  }
  let attention_bytes = read_module(args.attention_read.module_path)
  let target = runtime_device_target(args)
  let modules = selected_ingress_modules(
    args, ingress_choice, residual_bytes, ingress_bytes,
  )
  modules.push(
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=QwenPagedAttentionReadOnlyProductionV2,
      first_operation=args.attention_prefill.first_operation,
      operation_count=1,
      function_symbol=prefill_function_symbol(prefill_exponential_law),
      prefill_exponential_law~,
      allow_approximate_prefill_exponential=prefill_exponential_law ==
        ApproximateBaseTwoExponential,
      launch=launch_dimensions(args.attention_prefill),
      module_bytes=attention_prefill_bytes,
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
      allow_approximate_prefill_exponential=wide_prefill_exponential_law ==
        ApproximateBaseTwoExponential,
      launch=launch_dimensions(args.attention_prefill_wide),
      module_bytes=attention_prefill_wide_bytes,
      query_tile_rows=args.wide_query_tile_rows,
      query_metadata_version=if args.query_metadata { 1 } else { 0 },
    ) catch {
      _ => raise InvalidRuntime
    },
  )
  modules.push(
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=QwenPagedAttentionPrefillPartitionedPartialProductionV4,
      first_operation=args.attention_prefill_partial.first_operation,
      operation_count=1,
      function_symbol="lunaflux_attention_prefill_partitioned_partial_v1",
      launch=launch_dimensions(args.attention_prefill_partial),
      module_bytes=attention_prefill_partial_bytes,
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
      query_tile_rows=args.deep_partitioned_query_tile_rows,
    ) catch {
      _ => raise InvalidRuntime
    },
  )
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
  ) catch {
    _ => raise InvalidRuntime
  }
  let admitted = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    runtime,
    @crypto.bytes_to_hex_string(@crypto.sha256(runtime)),
  ) catch {
    _ => raise InvalidRuntime
  }
  let route_scope = admitted.attention_route_scope() catch {
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
      [1, 2, 3, 4, 5, 6, 7],
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
  println("ingress_selection_basis=" + args.ingress_selection.canonical_name())
  println("attention_route_scope=" + route_scope)
  println(
    "ingress_schedule=" +
    (if ingress_choice == FullyFused { "full" } else { "producer-separated" }),
  )
  println(
    "runtime_sha256=" + @crypto.bytes_to_hex_string(@crypto.sha256(runtime)),
  )
  println(
    "residual_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(residual_bytes)),
  )
  println(
    "ingress_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(ingress_bytes)),
  )
  println(
    "prefill_wide_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(attention_prefill_wide_bytes)),
  )
  println(
    "attention_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(attention_bytes)),
  )
  println(
    "attention_prefill_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(attention_prefill_bytes)),
  )
  println(
    "attention_prefill_partitioned_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(attention_prefill_partial_bytes)),
  )
  println(
    "attention_prefill_partitioned8_module_sha256=" +
    @crypto.bytes_to_hex_string(
      @crypto.sha256(attention_prefill_partitioned8_partial_bytes),
    ),
  )
}

///|
async fn main {
  try! run(@env.args())
}
