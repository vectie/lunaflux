///|
fn qwen_fused_test_plan() -> @plan.ModelPlan raise {
  let spec = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    layer_count=28,
    attention_head_count=16,
    kv_head_count=8,
    head_dimension=128,
    context_length=40960,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  let metadata = @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "3333333333333333333333333333333333333333333333333333333333333333",
    ),
  )
  @qwen3.build_paged(metadata, batch=@qwen3.Qwen3PagedBatchEnvelope::new(3))
}

///|
fn qwen_fused_test_blueprint(
  qwen_plan : @plan.ModelPlan,
) -> PagedKvExecutionBlueprint raise {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, plan, layout, target)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  let contracts = full_contract_set(plan, static_plan, memory, layout, target)
  let base = admit_paged_graph_execution_blueprint(
    plan,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    blueprint_artifacts(contracts),
  )
  let template = base.steps()[0]
  let steps : Array[PagedKvExecutionStep] = []
  for operation in qwen_plan.operations() {
    steps.push({
      ..template,
      operation_id: operation.id(),
      operation_kind: operation.kind(),
      kv_layer: operation.kv_layer(),
    })
  }
  {
    ..base,
    identity: qwen_plan.identity(),
    steps: ReadOnlyArray::from_array(steps),
  }
}

///|
fn qwen_fused_test_runtime(
  plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  include_residual~ : Bool = false,
  include_readonly~ : Bool = true,
  include_ingress~ : Bool = true,
  ingress_grid_y~ : Int32 = 32,
  phase_specific_attention~ : Bool = false,
  direct_compiler_decode~ : Bool = false,
  split_compiler_decode~ : Bool = false,
  full_ingress~ : Bool = false,
  step_rotary_cache~ : Bool = false,
  ingress_query_tile_rows? : Int32,
  ingress_heads_per_cta~ : Int32 = 1,
  ingress_row_variant? : @luna_fused_artifact_admission.ReusableFusedRuntimeRowVariant,
  blockwise_decode~ : Bool = false,
  contracting_decode~ : Bool = false,
  dual_score_decode~ : Bool = false,
  decode_execution? : @luna_fused_artifact_admission.DecodeExecutionDescriptor,
  mismatched_decode_law~ : Bool = false,
  ingress_grid_x~ : Int32 = -1,
  wide_query_prefill~ : Bool = false,
  wide_query_prefill_alias_selected~ : Bool = false,
  partitioned_prefill~ : Bool = false,
  partitioned_prefill_grid_x~ : Int32 = -1,
  deep_partitioned_prefill~ : Bool = false,
  partitioned_query_tile_rows~ : Int32 = 32,
  merge_query_tile_rows~ : Int32 = 32,
  prefill_threads~ : Int32 = 256,
  merge_threads~ : Int32 = -1,
  attention_bucket_routes? : Bytes,
) -> @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle raise {
  let ingress = @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
    abi=if full_ingress {
      if step_rotary_cache {
        AttentionIngressPreparedRotaryProductionV11
      } else {
        QwenQkvQkNormPositionedRopePagedKvWriteProductionV4
      }
    } else {
      QwenQkNormPositionedRopePagedKvWriteProductionV2
    },
    first_operation=if full_ingress { 2 } else { 3 },
    operation_count=if full_ingress { 4 } else { 3 },
    function_symbol=if full_ingress {
      if step_rotary_cache {
        "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3"
      } else {
        "qwen_full_ingress"
      }
    } else {
      "qwen_postprocess"
    },
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=if ingress_grid_x > 0 {
        ingress_grid_x
      } else if full_ingress {
        (blueprint.profile().max_query_tokens() - 1) /
        ingress_query_tile_rows.unwrap_or(16) +
        1
      } else {
        blueprint.profile().max_query_tokens()
      },
      grid_y=ingress_grid_y,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    module_bytes=b"qwen-postprocess-module",
    query_tile_rows?=ingress_query_tile_rows,
    heads_per_cta=ingress_heads_per_cta,
    row_variant?=ingress_row_variant,
  )
  let modules : Array[
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput,
  ] = []
  if include_residual {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=ResidualRmsNormProductionV2,
        first_operation=7,
        operation_count=2,
        function_symbol="residual_rmsnorm",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=blueprint.profile().max_query_tokens(),
          grid_y=1,
          grid_z=1,
          block_x=128,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        module_bytes=b"residual-module",
      ),
    )
  }
  if include_ingress {
    modules.push(ingress)
  }
  if include_readonly {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionReadOnlyProductionV2,
        query_tile_rows=16,
        first_operation=5,
        operation_count=1,
        function_symbol=if phase_specific_attention {
          "qwen_compiler_prefill"
        } else {
          "qwen_readonly"
        },
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=if phase_specific_attention {
            blueprint.profile().max_query_rows() +
            (
              blueprint.profile().max_query_tokens() -
              blueprint.profile().max_query_rows() +
              15
            ) /
            16
          } else {
            Int32::min(blueprint.profile().max_query_tokens(), 128)
          },
          grid_y=16,
          grid_z=1,
          block_x=if phase_specific_attention { prefill_threads } else { 128 },
          block_y=1,
          block_z=1,
          shared_memory_bytes=if phase_specific_attention { 31952 } else { 0 },
        ),
        module_bytes=b"qwen-readonly-module",
      ),
    )
  }
  if phase_specific_attention {
    let descriptor_geometry = match decode_execution {
      Some(descriptor) => {
        let (rows, stages) = descriptor.storage_geometry()
        Some(
          @luna_cuda_paged_attention_aot.blockwise_decode_execution_launch_geometry(
            @luna_attention_strategy.AttentionShape::new(
              query_heads=16,
              kv_heads=8,
              head_dimension=128,
              tokens_per_page=blueprint.layout().tokens_per_page(),
            ),
            8,
            descriptor.compute(),
            rows,
            stages,
          ),
        )
      }
      None => None
    }
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=if decode_execution is Some(_) {
          PagedAttentionReadOnlyExtendedDecodeProductionV11
        } else if dual_score_decode {
          PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10
        } else if contracting_decode {
          PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9
        } else if blockwise_decode {
          PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8
        } else {
          QwenPagedAttentionReadOnlyDecodeSplitProductionV3
        },
        first_operation=5,
        operation_count=1,
        function_symbol=if decode_execution is Some(descriptor) {
          descriptor.function_symbol()
        } else if dual_score_decode && !mismatched_decode_law {
          "lunaflux_attention_decode_tile_compiler_v1_dual_score_blockwise_f32_v2"
        } else if contracting_decode && !mismatched_decode_law {
          "lunaflux_attention_decode_tile_compiler_v1_blockwise_fma_f32_v3"
        } else if blockwise_decode != mismatched_decode_law {
          "lunaflux_attention_decode_tile_compiler_v1_blockwise_f32_v1"
        } else {
          "lunaflux_attention_decode_tile_compiler_v1"
        },
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=blueprint.profile().max_query_rows(),
          grid_y=if direct_compiler_decode { 16 } else { 8 },
          grid_z=1,
          block_x=if descriptor_geometry is Some(geometry) {
            geometry.2
          } else if direct_compiler_decode {
            32
          } else if split_compiler_decode {
            256
          } else {
            64
          },
          block_y=1,
          block_z=1,
          shared_memory_bytes=if descriptor_geometry is Some(geometry) {
            geometry.3.to_int()
          } else if direct_compiler_decode {
            0
          } else if blockwise_decode || contracting_decode || dual_score_decode {
            33040
          } else {
            34076
          },
        ),
        module_bytes=b"qwen-decode-module",
        decode_execution?,
      ),
    )
  }
  if wide_query_prefill {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillWideQueryProductionV5,
        query_tile_rows=if wide_query_prefill_alias_selected { 16 } else { 64 },
        first_operation=5,
        operation_count=1,
        function_symbol=if wide_query_prefill_alias_selected {
          "qwen_compiler_prefill"
        } else {
          "lunaflux_attention_prefill_tile_compiler_v1_c314"
        },
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=blueprint.profile().max_query_rows() +
            (
              blueprint.profile().max_query_tokens() -
              blueprint.profile().max_query_rows() +
              15
            ) /
            16,
          grid_y=16,
          grid_z=1,
          block_x=prefill_threads,
          block_y=1,
          block_z=1,
          shared_memory_bytes=if wide_query_prefill_alias_selected {
            31952
          } else {
            70032
          },
        ),
        module_bytes=if wide_query_prefill_alias_selected {
          b"qwen-readonly-module"
        } else {
          b"qwen-wide-prefill-module"
        },
      ),
    )
  }
  if partitioned_prefill {
    let shared_module = b"qwen-partitioned-prefill-module"
    let maximum_prefill_grid = blueprint.profile().max_query_rows() +
      (
        blueprint.profile().max_query_tokens() -
        blueprint.profile().max_query_rows() +
        15
      ) /
      16
    let prefill_grid = if partitioned_prefill_grid_x > 0 {
      partitioned_prefill_grid_x
    } else {
      maximum_prefill_grid
    }
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedPartialProductionV4,
        query_tile_rows=partitioned_query_tile_rows,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=prefill_grid,
          grid_y=16,
          grid_z=2,
          block_x=prefill_threads,
          block_y=1,
          block_z=1,
          shared_memory_bytes=31952,
        ),
        module_bytes=shared_module,
      ),
    )
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedMergeProductionV4,
        query_tile_rows=merge_query_tile_rows,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=prefill_grid,
          grid_y=16,
          grid_z=1,
          block_x=if merge_threads > 0 {
            merge_threads
          } else {
            prefill_threads
          },
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        module_bytes=shared_module,
      ),
    )
  }
  if deep_partitioned_prefill {
    let shared_module = b"qwen-deep-partitioned-prefill-module"
    let prefill_grid = if partitioned_prefill_grid_x > 0 {
      partitioned_prefill_grid_x
    } else {
      blueprint.profile().max_query_rows() +
      (
        blueprint.profile().max_query_tokens() -
        blueprint.profile().max_query_rows() +
        15
      ) /
      16
    }
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitioned8PartialProductionV6,
        query_tile_rows=32,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1_p8",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=prefill_grid,
          grid_y=16,
          grid_z=8,
          block_x=prefill_threads,
          block_y=1,
          block_z=1,
          shared_memory_bytes=31952,
        ),
        module_bytes=shared_module,
      ),
    )
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitioned8MergeProductionV6,
        query_tile_rows=32,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1_p8",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=prefill_grid,
          grid_y=16,
          grid_z=1,
          block_x=if merge_threads > 0 {
            merge_threads
          } else {
            prefill_threads
          },
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        module_bytes=shared_module,
      ),
    )
  }
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    attention_bucket_routes?,
    model_content_sha256=plan.identity().content().as_hex(),
    model_plan_sha256=plan.identity().plan().as_hex(),
    target=blueprint.target(),
    modules~,
  )
  @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
}

///|
fn qwen_fused_test_binding(
  space : PagedKvOperandSpace,
  offset : Int64,
  bytes : Int64,
) -> PagedKvOperandBinding {
  {
    role: Workspace,
    space,
    offset_bytes: offset,
    byte_count: bytes,
    alignment: 16L,
  }
}
