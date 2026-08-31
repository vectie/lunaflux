///|
async fn qwen3_load_metadata(
  model_root : String,
  config_sha256 : String,
  model_content_sha256 : String,
) -> @qwen3_spec.Qwen3ModelMetadata raise Qwen3PhysicalError {
  qwen3_require(
    qwen3_is_sha256(model_content_sha256),
    "model digest is invalid",
  )
  let config = qwen3_read_pinned(
    model_root + "/config.json",
    config_sha256,
    1048576,
  )
  @config_reader.parse_qwen3_bytes(
    config,
    @spec.ContentDigest::from_sha256(model_content_sha256),
    max_bytes=1048576,
    max_depth=32,
  ) catch {
    _ => raise InvalidInput("Qwen3 config admission failed")
  }
}

///|
fn qwen3_candidate_inputs(
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  toolchain_sha256 : String,
  compiler_major : UInt,
  compiler_minor : UInt,
  compiler_patch : UInt,
) -> Qwen3PhysicalInputs raise Qwen3PhysicalError {
  qwen3_require(
    qwen3_is_sha256(toolchain_sha256),
    "toolchain digest is invalid",
  )
  qwen3_require(compiler_major > 0U, "compiler major version is invalid")
  let plan = @qwen3.build_paged(
    metadata,
    batch=@qwen3.Qwen3PagedBatchEnvelope::single_row(),
  ) catch {
    _ => raise CandidateMismatch("Qwen3 paged plan construction failed")
  }
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=16,
      total_page_count=8,
      max_arena_bytes=8589934592L,
    ),
  ) catch {
    _ => raise CandidateMismatch("Qwen3 KV layout construction failed")
  }
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=1,
    max_page_table_entries=8,
  ) catch {
    _ => raise CandidateMismatch("Qwen3 launch profile construction failed")
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  ) catch {
    _ => raise CandidateMismatch("sm120 target construction failed")
  }
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=toolchain_sha256,
      major=compiler_major.reinterpret_as_int(),
      minor=compiler_minor.reinterpret_as_int(),
      patch=compiler_patch.reinterpret_as_int(),
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  ) catch {
    _ => raise CandidateMismatch("compiler policy construction failed")
  }
  let candidates = @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
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
    _ => raise CandidateMismatch("Qwen3 BF16 candidate lowering failed")
  }
  qwen3_validate_candidates(metadata, candidates)
  { candidates, }
}

///|
fn qwen3_validate_candidates(
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  candidates : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> Unit raise Qwen3PhysicalError {
  let spec = metadata.spec()
  qwen3_require(
    candidates.operation_count() >= 13,
    "candidate graph is incomplete",
  )
  guard candidates.candidates()[2] is Projection(qkv) else {
    raise CandidateMismatch("operation 2 is not Qwen3 QKV projection")
  }
  qwen3_require(
    qkv.family() == QkvProjection &&
    qkv.input_width() == spec.hidden_size() &&
    qkv.intermediate_width() == spec.kv_projection_width() &&
    qkv.output_width() ==
    spec.query_projection_width() + 2 * spec.kv_projection_width(),
    "independent-width QKV lowering drifted",
  )
  guard candidates.candidates()[3] is Pointwise(qk) else {
    raise CandidateMismatch("operation 3 is not Q/K RMS normalization")
  }
  qwen3_require(
    qk.family() == QkRmsNorm &&
    candidates.candidate_key(3) == Some("op_00000003_qk_rms_norm") &&
    qk.operands().length() == 5,
    "capability-11 Q/K RMS normalization ABI drifted",
  )
}
