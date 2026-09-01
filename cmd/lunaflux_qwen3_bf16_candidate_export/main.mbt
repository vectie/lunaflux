///|
fn load_authenticated_paged_plan(
  arguments : Qwen3CandidateExportArguments,
) -> @plan.ModelPlan raise Qwen3Bf16CandidateExportCliError {
  let root = @approved_fs.ApprovedRoot::open_absolute(arguments.model_root) catch {
    _ => raise ModelAdmission
  }
  let result : Result[@plan.ModelPlan, Qwen3Bf16CandidateExportCliError] = try {
    let file = root.open_file(
      @approved_fs.ApprovedRelativeLocator::new(arguments.config_locator),
    )
    let config = file.read_immutable_snapshot(maximum_bytes=1048576L)
    file.close()
    if @crypto.bytes_to_hex_string(@crypto.sha256(config)) !=
      arguments.config_sha256 {
      raise ModelAdmission
    }
    let metadata = @config_reader.parse_qwen3_bytes(
      config,
      @spec.ContentDigest::from_sha256(arguments.model_content_sha256),
      max_bytes=1048576,
      max_depth=16,
    )
    let batch = @qwen3.Qwen3PagedBatchEnvelope::new(arguments.max_query_rows)
    let plan = @qwen3.build_paged(metadata, batch~)
    let inspection = @qwen3_weights.inspect_paged_file(
      root,
      metadata,
      @approved_fs.ApprovedRelativeLocator::new(
        arguments.numeric_weights_locator,
      ),
      expected_artifact_digest=@numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
        arguments.numeric_weight_artifact_sha256,
      ),
      limits=@numeric_weight_file.NumericWeightFileLimits::new(
        max_file_bytes=3221225472L,
        max_header_bytes=16777216,
        transfer_chunk_bytes=4194304,
        max_tensor_count=310,
      ),
      alignment_bytes=256L,
      max_arena_bytes=3221225472L,
      expected_route_manifest_sha256=arguments.weight_route_manifest_sha256,
      batch~,
    )
    if plan.operations().length() != 283 ||
      plan.tensor_count() != 310 ||
      inspection.identity() != plan.identity() {
      raise ModelAdmission
    }
    Ok(plan)
  } catch {
    _ => Err(ModelAdmission)
  }
  let close = try {
    root.close()
    Ok(())
  } catch {
    _ => Err(ModelRootCleanup)
  }
  match (result, close) {
    (Ok(plan), Ok(_)) => plan
    (Err(error), _) | (_, Err(error)) => raise error
  }
}

///|
fn prepare_device_layout(
  plan : @plan.ModelPlan,
  arguments : Qwen3CandidateExportArguments,
) -> @device_layout.DeviceKvLayout raise Qwen3Bf16CandidateExportCliError {
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=arguments.tokens_per_page,
      total_page_count=arguments.total_page_count,
      max_arena_bytes=17179869184L,
    ),
  ) catch {
    _ => raise CandidatePreparation
  }
}

///|
fn prepare_candidate_set(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  arguments : Qwen3CandidateExportArguments,
) -> @luna_bf16_kernel_producer.LunaBf16CandidateSet raise Qwen3Bf16CandidateExportCliError {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=arguments.max_query_rows,
    max_query_tokens=arguments.max_query_tokens,
    max_page_table_entries=arguments.max_page_table_entries,
  ) catch {
    _ => raise CandidatePreparation
  }
  if profile.max_query_rows() != plan.shape_constraints().max_batch_rows() ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_query_tokens().to_int64() >
    profile.max_query_rows().to_int64() *
    plan.shape_constraints().max_sequence_tokens().to_int64() {
    raise CandidatePreparation
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  ) catch {
    _ => raise CandidatePreparation
  }
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=arguments.toolchain_sha256,
      major=arguments.compiler_major.reinterpret_as_int(),
      minor=arguments.compiler_minor.reinterpret_as_int(),
      patch=arguments.compiler_patch.reinterpret_as_int(),
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  ) catch {
    _ => raise CandidatePreparation
  }
  @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    compiler,
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=4096,
      max_source_bytes=16777216,
      max_recipe_bytes=16777216,
      max_total_bytes=1073741824,
    ),
  ) catch {
    _ => raise CandidatePreparation
  }
}

///|
priv struct QwenFusedProductionSet {
  first_operation : Int
  postprocess : @luna_cuda_qknorm_rope_paged_kv_write_aot.QkNormRopePagedKvWriteCudaCandidate
  attention_read : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyProductionCudaCandidate
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
    return { first_operation: index, postprocess, attention_read }
  }
  raise QwenFusionMissing
}

///|
async fn write_new_candidate_pair(
  root : String,
  source : Bytes,
  recipe : Bytes,
) -> Unit raise Qwen3Bf16CandidateExportCliError {
  @fs.mkdir(root, permission=0o700) catch {
    _ => raise Publication
  }
  @fs.write_file(
    root + "/kernel.cu",
    source,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise Publication
  }
  @fs.write_file(
    root + "/kernel.recipe",
    recipe,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise Publication
  }
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

///|
async fn run_qwen3_candidate_export(
  raw_arguments : ArrayView[String],
) -> Unit raise Qwen3Bf16CandidateExportCliError {
  let arguments = parse_qwen3_candidate_export_arguments(raw_arguments)
  let plan = load_authenticated_paged_plan(arguments)
  let layout = prepare_device_layout(plan, arguments)
  let set = prepare_candidate_set(plan, layout, arguments)
  let residual = prepare_residual_rmsnorm_candidate(plan, set)
  let production = @luna_cuda_fused_parallel_aot.lower_fused_residual_rmsnorm_production_candidate(
    residual,
    @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
      maximum_source_bytes=1048576,
      maximum_recipe_bytes=65536,
      maximum_promotion_bytes=65536,
    ),
  ) catch {
    _ => raise ResidualProductionPreparation
  }
  let qwen_fused = prepare_qwen_fused_production_set(plan, layout, set)
  let evidence = @luna_bf16_candidate_export.export_new(
    set,
    arguments.absolute_new_output,
  ) catch {
    _ => raise Publication
  }
  let fused_root = arguments.absolute_new_output + "/reusable-fused-residual"
  write_new_candidate_pair(
    fused_root,
    production.source_bytes(),
    production.recipe_bytes(),
  )
  let ingress_root = arguments.absolute_new_output + "/reusable-qwen-ingress"
  write_new_candidate_pair(
    ingress_root,
    qwen_fused.postprocess.source_bytes(),
    qwen_fused.postprocess.recipe_bytes(),
  )
  let readonly_root = arguments.absolute_new_output +
    "/reusable-qwen-readonly-attention"
  write_new_candidate_pair(
    readonly_root,
    qwen_fused.attention_read.source_bytes(),
    qwen_fused.attention_read.recipe_bytes(),
  )
  println("schema=lunaflux-qwen3-bf16-candidate-export-cli.v1")
  println("recipe=dense_qwen3_bf16_paged_aot_v12")
  println(
    "weight_route_manifest_sha256=" + arguments.weight_route_manifest_sha256,
  )
  println("model_content_sha256=" + evidence.model_content_sha256())
  println("model_plan_sha256=" + evidence.model_plan_sha256())
  println("target=" + evidence.target())
  println("max_query_rows=" + arguments.max_query_rows.to_string())
  println("max_query_tokens=" + arguments.max_query_tokens.to_string())
  println("operation_count=" + evidence.operation_count().to_string())
  println("candidate_root=" + arguments.absolute_new_output + "/candidate-root")
  println(
    "candidate_inventory=" +
    arguments.absolute_new_output +
    "/candidate.files.sha256#sha256=" +
    evidence.inventory_sha256(),
  )
  println("candidate_set_sha256=" + evidence.candidate_set_sha256())
  println("export_record_sha256=" + evidence.export_record_sha256())
  println("compiler_invoked=0")
  println("device_opened=0")
  println("runtime_authority=0")
  println("physical_readiness=0")
  println("reusable_fused_residual_source=" + fused_root + "/kernel.cu")
  println("reusable_fused_residual_recipe=" + fused_root + "/kernel.recipe")
  println(
    "reusable_fused_residual_candidate_sha256=" +
    production.candidate_digest().as_hex(),
  )
  println(
    "reusable_fused_residual_first_operation=" +
    residual.residual_fallback().operation_id().as_int().to_string(),
  )
  println(
    "reusable_fused_residual_function_symbol=" + production.function_symbol(),
  )
  let residual_launch = production.launch_dimensions()
  println(
    "reusable_fused_residual_launch=" +
    residual_launch.grid_x().to_string() +
    "," +
    residual_launch.grid_y().to_string() +
    "," +
    residual_launch.block_x().to_string() +
    "," +
    residual_launch.shared_memory_bytes().to_string(),
  )
  println("reusable_qwen_ingress_source=" + ingress_root + "/kernel.cu")
  println("reusable_qwen_ingress_recipe=" + ingress_root + "/kernel.recipe")
  println(
    "reusable_qwen_ingress_candidate_sha256=" +
    qwen_fused.postprocess.candidate_digest().as_hex(),
  )
  println(
    "reusable_qwen_ingress_first_operation=" +
    qwen_fused.first_operation.to_string(),
  )
  println(
    "reusable_qwen_ingress_function_symbol=" +
    qwen_fused.postprocess.function_symbol(),
  )
  let ingress_launch = qwen_fused.postprocess.launch_dimensions()
  println(
    "reusable_qwen_ingress_launch=" +
    ingress_launch.grid_x().to_string() +
    "," +
    ingress_launch.grid_y().to_string() +
    "," +
    ingress_launch.block_x().to_string() +
    "," +
    ingress_launch.shared_memory_bytes().to_string(),
  )
  println("reusable_qwen_readonly_source=" + readonly_root + "/kernel.cu")
  println("reusable_qwen_readonly_recipe=" + readonly_root + "/kernel.recipe")
  println(
    "reusable_qwen_readonly_candidate_sha256=" +
    qwen_fused.attention_read.candidate_digest().as_hex(),
  )
  println(
    "reusable_qwen_readonly_operation=" +
    (qwen_fused.first_operation + 3).to_string(),
  )
  println(
    "reusable_qwen_readonly_function_symbol=" +
    qwen_fused.attention_read.function_symbol(),
  )
  let readonly_launch = qwen_fused.attention_read.launch_dimensions()
  println(
    "reusable_qwen_readonly_launch=" +
    readonly_launch.grid_x().to_string() +
    "," +
    readonly_launch.grid_y().to_string() +
    "," +
    readonly_launch.block_x().to_string() +
    "," +
    readonly_launch.shared_memory_bytes().to_string(),
  )
}

///|
async fn main {
  try run_qwen3_candidate_export(@env.args()) catch {
    error => {
      println("outcome=failed")
      println("error=" + candidate_export_error_name(error))
      abort("Qwen3 BF16 candidate export failed")
    }
  } noraise {
    _ => ()
  }
}
