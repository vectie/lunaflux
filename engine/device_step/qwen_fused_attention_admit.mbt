///|
/// Expands one digest-pinned Qwen ingress V4/read-only V2 module pair over
/// every graph-identical QKV -> QkRmsNorm -> RoPE -> Attention layer span.
/// Parsing, model identity, shape matching, and blueprint joins all complete
/// during startup; the resulting spans require no token-step validation.
pub fn admit_reusable_qwen_fused_attention_spans(
  plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  blueprint : PagedKvExecutionBlueprint,
  runtime : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle,
) -> ReadOnlyArray[ApprovedQwenFusedAttentionSpan] raise FusedGroupedOperationError {
  if runtime.model_content_sha256() != plan.identity().content().as_hex() ||
    runtime.model_plan_sha256() != plan.identity().plan().as_hex() ||
    blueprint.identity() != plan.identity() ||
    runtime.target() != blueprint.target() {
    raise MismatchedFusedGroupedOperation(ModelIdentity)
  }
  admit_prefill_exponential_model_contract(plan, runtime)
  let ingress_selection = match runtime.qwen_full_attention_ingress_module() {
    Some(module_) => Some((module_, true))
    None =>
      match runtime.qwen_qknorm_rope_kv_write_module() {
        Some(module_) => Some((module_, false))
        None => None
      }
  }
  let partitioned_modules = match
    (
      runtime.qwen_partitioned_prefill_partial_module(),
      runtime.qwen_partitioned_prefill_merge_module(),
    ) {
    (Some(partial), Some(merge)) => Some((partial, merge))
    (None, None) => None
    _ => raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let deep_partitioned_modules = match
    (
      runtime.qwen_partitioned8_prefill_partial_module(),
      runtime.qwen_partitioned8_prefill_merge_module(),
    ) {
    (Some(partial), Some(merge)) => Some((partial, merge))
    (None, None) => None
    _ => raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let prefill_module = runtime.qwen_paged_attention_prefill_module()
  // An offline profitability choice may retain only independent residual
  // fusions. Absence is legal only for the entire ingress/read-only family:
  // never silently install read-only attention without its KV-write producer.
  if ingress_selection is None &&
    runtime.attention_bucket_routes() is None &&
    prefill_module is None &&
    runtime.qwen_paged_attention_decode_module() is None &&
    runtime.qwen_paged_attention_readonly_module() is None &&
    runtime.qwen_wide_query_prefill_module() is None &&
    partitioned_modules is None &&
    deep_partitioned_modules is None {
    return ReadOnlyArray::from_array([])
  }
  guard ingress_selection is Some((postprocess, includes_projection)) &&
    prefill_module is Some(readonly_module) &&
    runtime.qwen_paged_attention_decode_module() is Some(decode_module) &&
    (
      (
        includes_projection &&
        (
          postprocess.kernel_abi() ==
          QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 ||
          postprocess.kernel_abi() ==
          AttentionIngressPreparedRotaryProductionV11
        ) &&
        postprocess.operation_count() == 4
      ) ||
      (
        !includes_projection &&
        postprocess.kernel_abi() ==
        QwenQkNormPositionedRopePagedKvWriteProductionV2 &&
        postprocess.operation_count() == 3
      )
    ) &&
    (
      readonly_module.kernel_abi() == QwenPagedAttentionReadOnlyProductionV2 ||
      readonly_module.kernel_abi() ==
      QwenPagedAttentionReadOnlyDecodeSplitProductionV3 ||
      readonly_module.kernel_abi() ==
      PagedAttentionReadOnlyCompilerDecodeProductionV7
    ) &&
    (
      decode_module.kernel_abi() == QwenPagedAttentionReadOnlyProductionV2 ||
      decode_module.kernel_abi() ==
      QwenPagedAttentionReadOnlyDecodeSplitProductionV3 ||
      decode_module.kernel_abi() ==
      PagedAttentionReadOnlyCompilerDecodeProductionV7 ||
      decode_module.kernel_abi().is_compiler_blockwise_decode()
    ) &&
    runtime.qwen_paged_attention_readonly_operation_index()
    is Some(readonly_attention) &&
    decode_module.operation_span() == (readonly_attention, 1) else {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let seed_qkv = if includes_projection {
    postprocess.first_operation_index()
  } else {
    postprocess.first_operation_index() - 1
  }
  let seed_qk_norm = seed_qkv + 1
  let seed_rope = seed_qkv + 2
  let seed_attention = seed_qkv + 3
  if seed_qkv < 0 ||
    postprocess.first_operation_index() !=
    (if includes_projection { seed_qkv } else { seed_qk_norm }) ||
    seed_rope != seed_qk_norm + 1 ||
    seed_attention != seed_qk_norm + 2 ||
    readonly_attention != seed_attention {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  require_qwen_fused_attention_plan_span(plan, blueprint, seed_qkv)
  let phase_specific = readonly_module.module_digest() !=
    decode_module.module_digest() ||
    readonly_module.function_symbol() != decode_module.function_symbol() ||
    readonly_module.launch_dimensions() != decode_module.launch_dimensions()
  if decode_module.query_metadata_version() != 0 ||
    (readonly_module.query_metadata_version() != 0 && !phase_specific) {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let readonly_launch = readonly_module.launch_dimensions()
  let readonly_query_tile_rows = match readonly_module.query_tile_rows() {
    Some(rows) => rows
    None => {
      if phase_specific {
        raise MismatchedFusedGroupedOperation(Launch)
      }
      1
    }
  }
  let decode_launch = decode_module.launch_dimensions()
  let maximum_rows = blueprint.profile().max_query_rows()
  let maximum_tokens = blueprint.profile().max_query_tokens()
  let compiler_prefill_grid = maximum_rows +
    (maximum_tokens - maximum_rows + 15) / 16
  let reference_prefill = readonly_module.prefill_exponential_law() ==
    ReferenceBf16ProbabilityBaseTwo
  let expected_readonly_grid = if reference_prefill {
    maximum_rows + (maximum_tokens - maximum_rows + 63) / 64
  } else {
    compiler_prefill_grid
  }
  guard plan.operations()[seed_qk_norm].shape()
    is QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, ..) &&
    kv_heads > 0 &&
    query_heads % kv_heads == 0 else {
    raise MismatchedFusedGroupedOperation(Launch)
  }
  if reference_prefill &&
    (
      head_dimension != 128 ||
      !((query_heads / kv_heads) is (1 | 2 | 4 | 8)) ||
      blueprint.layout().tokens_per_page() < 8 ||
      blueprint.layout().tokens_per_page() % 8 != 0 ||
      readonly_query_tile_rows != 64 ||
      readonly_module.query_metadata_version() != 0 ||
      readonly_launch.block_x() != 128 ||
      (
        readonly_launch.shared_memory_bytes() !=
        @luna_cuda_reference_attention.Query64Kv128.shared_memory_bytes() &&
        readonly_launch.shared_memory_bytes() !=
        @luna_cuda_reference_attention.Query64Kv64.shared_memory_bytes()
      ) ||
      partitioned_modules is Some(_) ||
      deep_partitioned_modules is Some(_)
    ) {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let blockwise_decode = decode_module
    .kernel_abi()
    .is_compiler_blockwise_decode()
  let (expected_blockwise_symbol, expected_grouped_threads, expected_shared) = match
    decode_module.decode_execution() {
    Some(descriptor) => {
      let shape = @luna_attention_strategy.AttentionShape::new(
        query_heads~,
        kv_heads~,
        head_dimension~,
        tokens_per_page=blueprint.layout().tokens_per_page(),
      ) catch {
        _ => raise MismatchedFusedGroupedOperation(Launch)
      }
      let (rows, stages) = descriptor.storage_geometry()
      let (_, _, threads, shared) = @luna_cuda_paged_attention_aot.blockwise_decode_execution_launch_geometry(
        shape,
        8,
        descriptor.compute(),
        rows,
        stages,
      ) catch {
        _ => raise MismatchedFusedGroupedOperation(Launch)
      }
      if !descriptor.matches_function_symbol(decode_module.function_symbol()) {
        raise MismatchedFusedGroupedOperation(KernelAbi)
      }
      (decode_module.function_symbol(), threads, shared.to_int())
    }
    None =>
      ("", query_heads / kv_heads * 32, decode_launch.shared_memory_bytes())
  }
  if (
      blockwise_decode &&
      decode_module.function_symbol() != expected_blockwise_symbol
    ) ||
    (!blockwise_decode && decode_module.function_symbol().contains("blockwise")) {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let compiler_decode = blockwise_decode ||
    decode_module.function_symbol() ==
    "lunaflux_attention_decode_tile_compiler_v1"
  let grouped_decode = compiler_decode &&
    decode_launch.grid_y() == kv_heads &&
    decode_launch.block_x() == expected_grouped_threads &&
    decode_launch.shared_memory_bytes() == expected_shared &&
    expected_shared > 0
  let direct_decode = compiler_decode &&
    decode_launch.grid_y() == query_heads &&
    decode_launch.block_x() == 32 &&
    decode_launch.shared_memory_bytes() == 0
  let split_grouped_decode = compiler_decode &&
    decode_launch.grid_y() == kv_heads &&
    decode_launch.block_x() > query_heads / kv_heads * 32 &&
    decode_launch.block_x() <= 256 &&
    decode_launch.block_x() % (query_heads / kv_heads * 32) == 0 &&
    decode_launch.shared_memory_bytes() > 0
  let wide_query_prefill = match runtime.qwen_wide_query_prefill_module() {
    Some(module_) => {
      let launch = module_.launch_dimensions()
      if module_.module_digest() == readonly_module.module_digest() &&
        module_.function_symbol() == readonly_module.function_symbol() &&
        launch == readonly_launch {
        if module_.query_tile_rows() != readonly_module.query_tile_rows() {
          raise MismatchedFusedGroupedOperation(Launch)
        }
        if module_.query_metadata_version() !=
          readonly_module.query_metadata_version() {
          raise MismatchedFusedGroupedOperation(KernelAbi)
        }
        // A stable bundle slot may carry the selected prefill module when the
        // semantic profile has no strictly wider query tile. Do not build a
        // duplicate executor or route a bucket through a fake specialization.
        None
      } else if module_.kernel_abi() !=
        QwenPagedAttentionPrefillWideQueryProductionV5 ||
        module_.operation_span() != (seed_attention, 1) ||
        launch.grid_x() != compiler_prefill_grid ||
        launch.grid_y() != query_heads ||
        launch.grid_z() != 1 ||
        launch.block_y() != 1 ||
        launch.block_z() != 1 ||
        launch.shared_memory_bytes() <= 0 ||
        launch.shared_memory_bytes() > 100000 {
        raise MismatchedFusedGroupedOperation(Launch)
      } else {
        guard module_.query_tile_rows() is Some(query_tile_rows) else {
          raise MismatchedFusedGroupedOperation(Launch)
        }
        Some({
          mixed_row_coverage: module_.mixed_row_coverage(),
          query_metadata: module_.query_metadata_version() == 1,
          module_digest: module_.module_digest(),
          module_bytes: module_.module_bytes(),
          symbol: module_.function_symbol(),
          launch,
          query_tile_rows,
        })
      }
    }
    None => None
  }
  let partitioned_prefill = admit_qwen_partitioned_prefill_pair(
    partitioned_modules,
    QwenPagedAttentionPrefillPartitionedPartialProductionV4,
    QwenPagedAttentionPrefillPartitionedMergeProductionV4,
    seed_attention,
    compiler_prefill_grid,
    maximum_tokens,
    query_heads,
    head_dimension,
  )
  let deep_partitioned_prefill = admit_qwen_partitioned_prefill_pair(
    deep_partitioned_modules,
    QwenPagedAttentionPrefillPartitioned8PartialProductionV6,
    QwenPagedAttentionPrefillPartitioned8MergeProductionV6,
    seed_attention,
    compiler_prefill_grid,
    maximum_tokens,
    query_heads,
    head_dimension,
  )
  match (partitioned_prefill, deep_partitioned_prefill) {
    (Some(base), Some(deep)) =>
      if base.partial_launch.grid_z() >= deep.partial_launch.grid_z() {
        raise MismatchedFusedGroupedOperation(Launch)
      }
    (None, Some(_)) => raise MismatchedFusedGroupedOperation(KernelAbi)
    _ => ()
  }
  let valid_prefill_launch = if partitioned_prefill is Some(_) {
    true
  } else if phase_specific {
    readonly_module.kernel_abi() == QwenPagedAttentionReadOnlyProductionV2 &&
    (
      decode_module.kernel_abi() ==
      QwenPagedAttentionReadOnlyDecodeSplitProductionV3 ||
      decode_module.kernel_abi() ==
      PagedAttentionReadOnlyCompilerDecodeProductionV7 ||
      decode_module.kernel_abi().is_compiler_blockwise_decode()
    ) &&
    readonly_launch.grid_x() == expected_readonly_grid &&
    readonly_launch.shared_memory_bytes() > 0 &&
    readonly_launch.shared_memory_bytes() <= 100000
  } else {
    readonly_launch.grid_x() == Int32::min(maximum_tokens, 128) &&
    readonly_launch.block_x() == 128 &&
    readonly_launch.shared_memory_bytes() == 0
  }
  let valid_decode_launch = if grouped_decode {
    decode_launch.grid_x() == maximum_rows &&
    decode_launch.grid_y() == kv_heads &&
    decode_launch.block_x() == expected_grouped_threads &&
    decode_launch.shared_memory_bytes() > 0 &&
    decode_launch.shared_memory_bytes() <= 100000
  } else if direct_decode {
    decode_launch.grid_x() == maximum_rows &&
    decode_launch.grid_y() == query_heads &&
    decode_launch.block_x() == 32 &&
    decode_launch.shared_memory_bytes() == 0
  } else if split_grouped_decode {
    decode_launch.grid_x() == maximum_rows &&
    decode_launch.grid_y() == kv_heads &&
    decode_launch.block_x() <= 256 &&
    decode_launch.shared_memory_bytes() > 0 &&
    decode_launch.shared_memory_bytes() <= 100000
  } else {
    decode_launch.grid_x() == Int32::min(maximum_tokens, 128) &&
    decode_launch.block_x() == 128 &&
    decode_launch.shared_memory_bytes() == 0
  }
  let ingress_grid_x = if includes_projection {
    // CTA row ownership is selected by the compiler, not by the model adapter.
    // Bundles predating explicit schedule metadata retain the old 16-row ABI.
    let rows = postprocess.query_tile_rows().unwrap_or(16)
    (maximum_tokens - 1) / rows + 1
  } else {
    Int32::min(blueprint.profile().max_query_tokens(), 4096)
  }
  let ingress_heads = query_heads + 2 * kv_heads
  let ingress_head_groups = postprocess.launch_dimensions().grid_y()
  let valid_ingress_head_groups = if includes_projection {
    ingress_head_groups == (ingress_heads - 1) / postprocess.heads_per_cta() + 1
  } else {
    @luna_cuda_qknorm_rope_paged_kv_write_aot.supports_production_head_group_count(
      ingress_heads, ingress_head_groups,
    )
  }
  guard maximum_tokens >= maximum_rows &&
    (!blockwise_decode || grouped_decode) &&
    valid_prefill_launch &&
    readonly_launch.grid_y() == query_heads &&
    readonly_launch.grid_z() == 1 &&
    readonly_launch.block_y() == 1 &&
    readonly_launch.block_z() == 1 &&
    (
      grouped_decode ||
      split_grouped_decode ||
      direct_decode ||
      decode_launch.grid_y() == query_heads
    ) &&
    decode_launch.grid_z() == 1 &&
    valid_decode_launch &&
    decode_launch.block_y() == 1 &&
    decode_launch.block_z() == 1 &&
    postprocess.launch_dimensions().grid_x() == ingress_grid_x &&
    valid_ingress_head_groups &&
    postprocess.launch_dimensions().grid_z() == 1 &&
    postprocess.launch_dimensions().block_x() == 128 &&
    postprocess.launch_dimensions().block_y() == 1 &&
    postprocess.launch_dimensions().block_z() == 1 &&
    (if includes_projection {
      postprocess.launch_dimensions().shared_memory_bytes() >= 0 &&
      postprocess.launch_dimensions().shared_memory_bytes() <= 98304
    } else {
      postprocess.launch_dimensions().shared_memory_bytes() == 0
    }) else {
    raise MismatchedFusedGroupedOperation(Launch)
  }
  let measured_attention_candidates = admit_qwen_attention_routes(
    runtime,
    blueprint.profile(),
    wide_query_prefill is Some(_),
  )
  let ingress_row_variant = postprocess.row_variant()
  if ingress_row_variant is Some(variant) &&
    (!includes_projection || variant.maximum_token_rows() > maximum_tokens) {
    raise MismatchedFusedGroupedOperation(Launch)
  }
  let operations = plan.operations()
  let spans : Array[ApprovedQwenFusedAttentionSpan] = []
  let step_rotary_cache = if postprocess.kernel_abi() ==
    AttentionIngressPreparedRotaryProductionV11 {
    if postprocess.function_symbol() !=
      "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3" {
      raise MismatchedFusedGroupedOperation(KernelAbi)
    }
    guard @elementwise_physical_ir.StepRotaryCachePlan::new(
        maximum_tokens=blueprint.profile().max_query_tokens(),
        head_dimension~,
      )
      is Some(cache) else {
      raise MismatchedFusedGroupedOperation(Launch)
    }
    Some(cache)
  } else {
    None
  }
  if operations.length() < 4 {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  for qkv_index in 0..<(operations.length() - 3) {
    if !qwen_fused_attention_shapes_equal(operations, seed_qkv, qkv_index) {
      continue
    }
    require_qwen_fused_attention_plan_span(plan, blueprint, qkv_index)
    require_qwen_fused_blueprint_span_shape(blueprint, seed_qkv, qkv_index)
    spans.push({
      generation,
      target: runtime.target(),
      runtime_digest: runtime.source_digest(),
      measured_attention_candidates,
      ingress_includes_projection: includes_projection,
      ingress_row_variant,
      step_rotary_cache,
      ingress_query_tile_rows: if includes_projection {
        postprocess.query_tile_rows().unwrap_or(16)
      } else {
        1
      },
      postprocess_module_digest: postprocess.module_digest(),
      postprocess_module_bytes: postprocess.module_bytes(),
      postprocess_symbol: postprocess.function_symbol(),
      postprocess_launch: postprocess.launch_dimensions(),
      readonly_module_digest: readonly_module.module_digest(),
      readonly_module_bytes: readonly_module.module_bytes(),
      readonly_symbol: readonly_module.function_symbol(),
      readonly_launch: readonly_module.launch_dimensions(),
      readonly_query_tile_rows,
      readonly_query_metadata: readonly_module.query_metadata_version() == 1,
      mixed_row_coverage: readonly_module.mixed_row_coverage(),
      decode_module_digest: decode_module.module_digest(),
      decode_module_bytes: decode_module.module_bytes(),
      decode_symbol: decode_module.function_symbol(),
      decode_launch: decode_module.launch_dimensions(),
      wide_query_prefill,
      partitioned_prefill,
      deep_partitioned_prefill,
      qkv_index,
      qk_norm_index: qkv_index + 1,
      rope_index: qkv_index + 2,
      attention_index: qkv_index + 3,
    })
  }
  if spans.is_empty() {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  ReadOnlyArray::from_array(spans)
}
