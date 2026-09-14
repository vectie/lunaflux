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
    projection_records=load_candidate_projection_records(arguments)[:],
  ) catch {
    _ => raise CandidatePreparation
  }
}
