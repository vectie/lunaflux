///|
priv struct QwenFusedProductionSet {
  first_operation : Int
  postprocess : @luna_cuda_qknorm_rope_paged_kv_write_aot.QkNormRopePagedKvWriteCudaCandidate
  full_ingress : @luna_cuda_fused_parallel_aot.FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate
  attention_prefill_frontier : @luna_cuda_attention_tile_aot.CudaAttentionTileAotFrontier
  attention_prefill_partitioned : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier
  attention_decode : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation
  attention_decode_frontier : @luna_cuda_attention_tile_aot.CudaAttentionTileAotFrontier
  attention_read : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyProductionCudaCandidate
  attention_decode_split : @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitCudaBundle
}

///|
fn qwen_postprocess_activation_input(
  operand : @launch_contract.AotOperand,
) -> @launch_contract.AotOperand raise Qwen3Bf16CandidateExportCliError {
  guard operand.role() is ActivationOutput(reference) else {
    raise QwenIngressPreparation
  }
  @launch_contract.AotOperand::new(
    role=ActivationInput(reference),
    byte_count=operand.byte_count(),
    alignment=operand.alignment(),
  ) catch {
    _ => raise QwenIngressPreparation
  }
}

///|
fn qwen_postprocess_operands(
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk_norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
) -> Array[@launch_contract.AotOperand] raise Qwen3Bf16CandidateExportCliError {
  let operands : Array[@launch_contract.AotOperand] = []
  for operand in kv_write.operands() {
    if operand.role() is RuntimeInput(_) {
      operands.push(operand)
    }
  }
  for operand in qkv.operands() {
    if operand.role() is ActivationOutput(_) {
      // Projection writes this allocation, while postprocess reads it. Only
      // the ABI direction changes; extent and alignment remain admitted by
      // the projection lowering.
      operands.push(qwen_postprocess_activation_input(operand))
    }
  }
  for operand in qk_norm.operands() {
    if operand.role() is WeightInput(_) {
      operands.push(operand)
    }
  }
  for operand in rope.operands() {
    if operand.role() is ActivationOutput(_) {
      operands.push(operand)
    }
  }
  for operand in kv_write.operands() {
    if operand.role() is PersistentKvComponent(_, _) {
      operands.push(operand)
    }
  }
  if operands.length() != 12 {
    raise QwenIngressPreparation
  }
  operands
}

///|
fn prepare_qwen_fused_production_set(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  tuning? : (Bytes, String, String)? = None,
  resources? : (Bytes, String, String)? = None,
  decode_resources? : (Bytes, String, String)? = None,
  decode_tuning? : (Bytes, String, String)? = None,
  partitioned_resources? : (Bytes, String, String)? = None,
  query_metadata? : Bool = false,
) -> QwenFusedProductionSet raise Qwen3Bf16CandidateExportCliError {
  let candidates = set.candidates()
  for index in 0..<(candidates.length() - 3) {
    guard candidates[index] is Projection(qkv) &&
      candidates[index + 1] is Pointwise(qk_norm) &&
      qk_norm.family() == QkRmsNorm &&
      candidates[index + 2] is Pointwise(rope) &&
      rope.family() == PositionedRotary &&
      candidates[index + 3] is PagedAttention(attention) else {
      continue
    }
    let operation = plan.operations()[index + 3]
    if plan.operations()[index].kind() != QkvProjection ||
      plan.operations()[index + 1].kind() != QkRmsNorm ||
      plan.operations()[index + 2].kind() != RotaryEmbedding ||
      operation.kind() != CausalAttention {
      continue
    }
    let kv_operands : Array[@launch_contract.AotOperand] = []
    for operand in attention.operands() {
      match operand.role() {
        ActivationOutput(_) => ()
        _ => kv_operands.push(operand)
      }
    }
    let pointwise_limits = @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ) catch {
      _ => raise QwenKvWritePreparation
    }
    let kv_write = @luna_cuda_pointwise_aot.lower_auxiliary_kv_cache_write_cuda_aot(
      operation,
      set.profile(),
      layout,
      @catalog.AotKernelEntryPointId::from_stable_int(3901),
      set.target(),
      set.compiler_policy(),
      kv_operands,
      pointwise_limits,
    ) catch {
      _ => raise QwenKvWritePreparation
    }
    let semantic = @luna_fusion_plan.match_qkv_qknorm_positioned_rope_paged_kv_write(
      plan,
      qkv.operation_id(),
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let postprocess_limits = @luna_cuda_qknorm_rope_paged_kv_write_aot.QkNormRopePagedKvWriteAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_module_bytes=8388608L,
      max_receipt_bytes=65536,
      max_binding_bytes=65536,
      maximum_grid_x=4096,
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let postprocess = @luna_cuda_qknorm_rope_paged_kv_write_aot.lower_qknorm_rope_paged_kv_write_cuda_aot_candidate(
      plan,
      semantic,
      set.profile(),
      layout,
      set.target(),
      set.compiler_policy(),
      qwen_postprocess_operands(qkv, qk_norm, rope, kv_write),
      postprocess_limits,
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let full_ingress_qualification = @luna_cuda_fused_parallel_aot.lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_candidate(
      plan,
      layout,
      qkv,
      qk_norm,
      rope,
      kv_write,
      set.target(),
      set.compiler_policy(),
      @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
        maximum_source_bytes=1048576,
        maximum_recipe_bytes=65536,
        maximum_promotion_bytes=65536,
      ),
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let full_ingress = @luna_cuda_fused_parallel_aot.lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
      full_ingress_qualification,
      @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
        maximum_source_bytes=1048576,
        maximum_recipe_bytes=65536,
        maximum_promotion_bytes=65536,
      ),
    ) catch {
      _ => raise QwenIngressPreparation
    }
    let readonly_limits = @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyAotLimits::new(
      maximum_source_bytes=1048576,
      maximum_recipe_bytes=65536,
      maximum_canonical_bytes=65536,
    ) catch {
      _ => raise QwenReadOnlyPreparation
    }
    let readonly_qualification = @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_cuda_aot_candidate(
      attention,
      @catalog.AotKernelEntryPointId::from_stable_int(3902),
      readonly_limits,
    ) catch {
      _ => raise QwenReadOnlyPreparation
    }
    let attention_read = @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_production_cuda_aot_candidate(
      readonly_qualification, readonly_limits,
    ) catch {
      _ => raise QwenReadOnlyPreparation
    }
    let attention_decode_split = @luna_cuda_paged_attention_aot.lower_paged_attention_decode_split_cuda_aot_candidate(
      operation,
      set.profile(),
      layout,
      @catalog.AotKernelEntryPointId::from_stable_int(3903),
      @catalog.AotKernelEntryPointId::from_stable_int(3904),
      set.target(),
      set.compiler_policy(),
      attention.operands(),
      8,
      268435456L,
      @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
        max_source_bytes=1048576,
        max_recipe_bytes=65536,
      ),
      compiler_readonly=true,
    ) catch {
      _ => raise QwenDecodeSplitPreparation
    }
    let attention_prefill_frontier = prepare_qwen_compiler_prefill(
      plan,
      layout,
      set,
      index,
      index + 3,
      tuning~,
      resources~,
      query_metadata~,
    )
    let attention_prefill_partitioned = prepare_qwen_compiler_partitioned_prefill(
      set,
      attention_prefill_frontier.selected(),
      resources=partitioned_resources,
    )
    let attention_decode_frontier = prepare_qwen_compiler_decode(
      plan,
      layout,
      set,
      index,
      index + 3,
      resources=decode_resources,
      tuning=decode_tuning,
    )
    return {
      first_operation: index,
      postprocess,
      full_ingress,
      attention_prefill_frontier,
      attention_prefill_partitioned,
      attention_decode: attention_decode_frontier.selected(),
      attention_decode_frontier,
      attention_read,
      attention_decode_split,
    }
  }
  raise QwenFusionMissing
}

///|
fn prepare_residual_rmsnorm_candidate(
  plan : @plan.ModelPlan,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate raise Qwen3Bf16CandidateExportCliError {
  let candidates = set.candidates()
  for index in 0..<(candidates.length() - 1) {
    guard candidates[index] is Pointwise(residual) &&
      residual.family() == ResidualAdd &&
      candidates[index + 1] is Pointwise(norm) &&
      norm.family() == RmsNorm else {
      continue
    }
    return @luna_cuda_fused_parallel_aot.lower_fused_residual_rmsnorm_candidate(
      plan,
      residual,
      norm,
      set.target(),
      set.compiler_policy(),
      @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
        maximum_source_bytes=1048576,
        maximum_recipe_bytes=65536,
        maximum_promotion_bytes=65536,
      ),
    ) catch {
      error => {
        println("residual_fusion_detail=" + fused_parallel_error_name(error))
        raise ResidualFusionPreparation
      }
    }
  }
  raise CandidatePreparation
}
