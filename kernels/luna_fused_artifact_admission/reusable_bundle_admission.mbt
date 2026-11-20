///|
/// Parses the bundle once at startup. Supplied digests are labels, not verified
/// checksums. Shape, symbol, duplicate and memory bounds remain checked before
/// the immutable typed bundle can reach a runtime owner.
pub fn admit_digest_pinned_reusable_fused_runtime_bundle(
  canonical : Bytes,
  expected_sha256 : String,
) -> ApprovedReusableFusedRuntimeBundle raise FusedArtifactAdmissionError {
  let source_digest = fused_deployment_digest(expected_sha256)
  let reader = fused_deployment_reader(canonical)
  let schema = reader.value("schema")
  let (
    has_schedule_metadata,
    has_query_metadata,
    has_bucket_routes,
    has_ownership,
  ) = match schema {
    "lunaflux-reusable-fused-runtime-bundle.v15" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v14" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v13" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v12" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v11" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v10" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v9" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v8" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v7" => (true, true, true, true)
    "lunaflux-reusable-fused-runtime-bundle.v6" => (true, true, true, false)
    "lunaflux-reusable-fused-runtime-bundle.v5" => (true, true, false, false)
    "lunaflux-reusable-fused-runtime-bundle.v4" => (true, false, false, false)
    "lunaflux-reusable-fused-runtime-bundle.v3" => (false, false, false, false)
    _ => raise InvalidFusedArtifact(Manifest)
  }
  let has_partition_metadata = schema ==
    "lunaflux-reusable-fused-runtime-bundle.v15"
  let has_vendor_projection = has_partition_metadata ||
    schema == "lunaflux-reusable-fused-runtime-bundle.v14"
  let has_row_metadata = has_vendor_projection ||
    schema == "lunaflux-reusable-fused-runtime-bundle.v13"
  let has_prefill_metadata = has_row_metadata ||
    schema == "lunaflux-reusable-fused-runtime-bundle.v12"
  let has_decode_metadata = has_prefill_metadata ||
    schema == "lunaflux-reusable-fused-runtime-bundle.v11"
  let model_content = fused_deployment_digest(
    reader.value("model_content_sha256"),
  )
  let model_plan = fused_deployment_digest(reader.value("model_plan_sha256"))
  let compute_major = reader.int("compute_major")
  let compute_minor = reader.int("compute_minor")
  let supports_bf16 = reader.int("supports_bf16")
  let supports_cublas_lt = reader.int("supports_cublas_lt")
  if (supports_bf16 != 0 && supports_bf16 != 1) ||
    (supports_cublas_lt != 0 && supports_cublas_lt != 1) {
    raise InvalidFusedArtifact(Target)
  }
  let target = @catalog.DeviceTarget::new(
    compute_major~,
    compute_minor~,
    supports_bf16=supports_bf16 == 1,
    supports_cublas_lt=supports_cublas_lt == 1,
  ) catch {
    _ => raise InvalidFusedArtifact(Target)
  }
  if !target.supports_bf16() {
    raise InvalidFusedArtifact(Target)
  }
  let allow_approximate_prefill = if has_prefill_metadata {
    let enabled = reader.int("allow_approximate_prefill_exponential")
    if enabled != 1 && !(has_row_metadata && enabled == 0) {
      raise InvalidFusedArtifact(KernelAbi)
    }
    enabled == 1
  } else {
    false
  }
  let attention_bucket_routes = if has_bucket_routes {
    let encoded_routes = reader.value("attention_bucket_routes_hex")
    if encoded_routes == "-" && has_ownership {
      None
    } else {
      let routes = fused_deployment_decode_hex(encoded_routes)
      if routes.is_empty() || routes.length() > 1048576 {
        raise InvalidFusedArtifact(Manifest)
      }
      Some(routes)
    }
  } else {
    None
  }
  let vendor_projection_exact_rows = if has_vendor_projection {
    let rows = reader.int("vendor_projection_exact_rows")
    if (rows < 2 && !(has_partition_metadata && rows == 0)) || rows > 1048576 {
      raise InvalidFusedArtifact(Manifest)
    }
    rows
  } else {
    0
  }
  let module_count = reader.int("module_count")
  if module_count <= 0 || module_count > REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULES {
    raise InvalidFusedArtifact(Binding)
  }
  let modules : Array[ApprovedReusableFusedRuntimeModule] = []
  for ordinal in 0..<module_count {
    let prefix = "module_" + ordinal.to_string() + "_"
    let abi = reusable_runtime_parse_abi(reader.value(prefix + "abi"))
    let first_operation = reader.int(prefix + "first_operation")
    let operation_count = reader.int(prefix + "operation_count")
    let symbol = reader.value(prefix + "function_symbol")
    let query_tile_rows = if has_schedule_metadata {
      let rows = reader.int(prefix + "query_tile_rows")
      if rows < 0 || rows > 1048576 {
        raise InvalidFusedArtifact(Launch)
      }
      if rows == 0 {
        None
      } else {
        Some(rows)
      }
    } else {
      None
    }
    let query_metadata_version = if has_query_metadata {
      reader.int(prefix + "query_metadata_version")
    } else {
      0
    }
    let heads_per_cta = if has_ownership {
      reader.int(prefix + "heads_per_cta")
    } else {
      1
    }
    let supplied_decode_execution = if has_decode_metadata {
      let law = reader.value(prefix + "decode_numeric_law")
      let rows = reader.int(prefix + "decode_key_value_tile_rows")
      let stages = reader.int(prefix + "decode_pipeline_stages")
      if law == "-" {
        if rows != 0 || stages != 0 {
          raise InvalidFusedArtifact(KernelAbi)
        }
        None
      } else {
        Some(
          DecodeExecutionDescriptor::from_numeric_law(
            law,
            key_value_tile_rows=rows,
            pipeline_stages=stages,
          ),
        )
      }
    } else {
      None
    }
    let decode_execution = reusable_runtime_decode_execution(
      abi, supplied_decode_execution,
    )
    if abi == PagedAttentionReadOnlyExtendedDecodeProductionV11 &&
      decode_execution is Some(descriptor) &&
      !descriptor.matches_function_symbol(symbol) {
      raise InvalidFusedArtifact(KernelAbi)
    }
    let prefill_exponential_law = reusable_runtime_read_prefill_exponential(
      reader, prefix, abi, symbol, has_prefill_metadata, allow_approximate_prefill,
    )
    let decode_execution = if has_partition_metadata {
      let count = reader.int(prefix + "decode_partition_count")
      match decode_execution {
        Some(descriptor) => {
          let (rows, stages) = descriptor.storage_geometry()
          Some(
            DecodeExecutionDescriptor::from_numeric_law(
              descriptor.numeric_law(),
              key_value_tile_rows=rows,
              pipeline_stages=stages,
              partition_count=count,
            ),
          )
        }
        None => {
          if count != 0 {
            raise InvalidFusedArtifact(KernelAbi)
          }
          None
        }
      }
    } else {
      decode_execution
    }
    reusable_runtime_validate_head_ownership(
      abi, heads_per_cta, query_tile_rows,
    )
    if abi == PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9 &&
      schema != "lunaflux-reusable-fused-runtime-bundle.v8" &&
      schema != "lunaflux-reusable-fused-runtime-bundle.v9" &&
      schema != "lunaflux-reusable-fused-runtime-bundle.v10" &&
      !has_decode_metadata {
      raise InvalidFusedArtifact(KernelAbi)
    }
    if abi == PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10 &&
      schema != "lunaflux-reusable-fused-runtime-bundle.v9" &&
      schema != "lunaflux-reusable-fused-runtime-bundle.v10" &&
      !has_decode_metadata {
      raise InvalidFusedArtifact(KernelAbi)
    }
    if abi == AttentionIngressPreparedRotaryProductionV11 &&
      schema != "lunaflux-reusable-fused-runtime-bundle.v10" &&
      !has_decode_metadata {
      raise InvalidFusedArtifact(KernelAbi)
    }
    if abi == PagedAttentionReadOnlyExtendedDecodeProductionV11 &&
      !has_decode_metadata {
      raise InvalidFusedArtifact(KernelAbi)
    }
    if abi.is_compiler_blockwise_decode() && !has_ownership {
      raise InvalidFusedArtifact(KernelAbi)
    }
    if query_metadata_version != 0 &&
      (
        query_metadata_version != 1 ||
        !(query_tile_rows is Some(16 | 32 | 64 | 128)) ||
        !reusable_runtime_is_qwen_attention_reader(abi)
      ) {
      raise InvalidFusedArtifact(KernelAbi)
    }
    let launch = @launch_contract.AotLaunchDimensions::new(
      grid_x=reader.int(prefix + "grid_x"),
      grid_y=reader.int(prefix + "grid_y"),
      grid_z=reader.int(prefix + "grid_z"),
      block_x=reader.int(prefix + "block_x"),
      block_y=reader.int(prefix + "block_y"),
      block_z=reader.int(prefix + "block_z"),
      shared_memory_bytes=reader.int(prefix + "shared_bytes"),
    ) catch {
      _ => raise InvalidFusedArtifact(Launch)
    }
    let module_digest = fused_deployment_digest(
      reader.value(prefix + "module_sha256"),
    )
    let module_bytes = fused_deployment_decode_hex(
      reader.value(prefix + "module_hex"),
    )
    reusable_runtime_validate_module(
      abi, first_operation, operation_count, symbol, launch, module_bytes,
    )
    let row_variant = if has_row_metadata {
      reusable_runtime_read_row_variant(reader, prefix)
    } else {
      None
    }
    reusable_runtime_validate_row_variant(
      abi, symbol, launch, heads_per_cta, row_variant,
    )
    modules.push({
      abi,
      first_operation,
      operation_count,
      symbol,
      launch,
      query_tile_rows,
      query_metadata_version,
      heads_per_cta,
      decode_execution,
      prefill_exponential_law,
      module_digest,
      module_bytes,
      row_variant,
    })
  }
  if reader.cursor != reader.lines.length() - 1 {
    raise InvalidFusedArtifact(Manifest)
  }
  reusable_runtime_validate_unique_approved(modules)
  {
    source_digest,
    model_content,
    model_plan,
    target,
    modules: ReadOnlyArray::from_array(modules),
    attention_bucket_routes,
    vendor_projection_exact_rows,
  }
}
