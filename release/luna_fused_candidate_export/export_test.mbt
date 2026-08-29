///|
fn fused_export_test_inputs(
  compute_major? : Int = 12,
) -> (
  @plan.ModelPlan,
  @device_layout.DeviceKvLayout,
  @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=256,
    hidden_size=128,
    intermediate_size=256,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    context_length=256,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let plan = @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("a".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=16,
      total_page_count=4,
      max_arena_bytes=67108864L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(7),
    max_query_rows=1,
    max_query_tokens=8,
    max_page_table_entries=4,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major~,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=compute_major > 0,
  )
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="b".repeat(64),
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let set = @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    compiler,
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=64,
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_total_bytes=33554432,
    ),
  )
  (plan, layout, set)
}

///|
test "qualification set is deterministic sm120 block128 and authority free" {
  let (plan, layout, reference) = fused_export_test_inputs()
  let first = prepare_qualification_set(plan, layout, reference)
  let second = prepare_qualification_set(plan, layout, reference)
  assert_eq(first.declaration_bytes(), second.declaration_bytes())
  assert_eq(first.inventory_bytes(), second.inventory_bytes())
  assert_eq(first.declaration_sha256(), second.declaration_sha256())
  assert_eq(first.inventory_sha256(), second.inventory_sha256())
  assert_false(first.manifest_bindable())
  assert_eq(first.qkv_candidate().launch_dimensions().block_x(), 128)
  assert_eq(first.residual_norm_candidate().launch_dimensions().block_x(), 128)
  let declaration = @utf8.decode(first.declaration_bytes())
  assert_true(declaration.contains("target=sm_120\n"))
  assert_true(declaration.contains("block_threads=128\n"))
  assert_true(declaration.contains("qualification_only=1\n"))
  assert_true(declaration.contains("manifest_bindable=0\n"))
  assert_true(declaration.contains("compiler_authority=0\n"))
  assert_true(declaration.contains("artifact_authority=0\n"))
  assert_true(declaration.contains("device_authority=0\n"))
  assert_true(declaration.contains("runtime_authority=0\n"))
  assert_true(declaration.contains("promotion_authority=0\n"))
}

///|
test "qualification set rejects non-sm120 reference candidates" {
  let (plan, layout, reference) = fused_export_test_inputs(compute_major=9)
  try prepare_qualification_set(plan, layout, reference) catch {
    FusedCandidateExportFailed(Qualification) => ()
    _ => fail("unexpected non-sm120 qualification rejection")
  } noraise {
    _ => fail("non-sm120 candidates gained qualification authority")
  }
}

///|
async test "export atomically publishes exact bytes and refuses overwrite" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-fused-candidate-export")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let (plan, layout, reference) = fused_export_test_inputs()
    let set = prepare_qualification_set(plan, layout, reference)
    let output = parent + "/candidate-set"
    let evidence = export_new(set, output)
    assert_eq(evidence.declaration_sha256(), set.declaration_sha256())
    assert_eq(evidence.inventory_sha256(), set.inventory_sha256())
    assert_eq(
      @fs.read_file(output + "/candidate-root/candidate-set.v1").binary(),
      set.declaration_bytes(),
    )
    assert_eq(
      @fs.read_file(output + "/candidate.files.sha256").binary(),
      set.inventory_bytes(),
    )
    assert_eq(
      @fs.readdir(output + "/candidate-root/sources", sort=true).length(),
      5,
    )
    assert_eq(
      @fs.readdir(output + "/candidate-root/recipes", sort=true).length(),
      5,
    )
    let record = @utf8.decode(@fs.read_file(output + "/export.v1").binary())
    assert_true(record.contains("qualification_only=1\n"))
    assert_true(record.contains("compiler_invoked=0\n"))
    assert_true(record.contains("artifact_authority=0\n"))
    assert_true(record.contains("device_opened=0\n"))
    assert_true(record.contains("runtime_authority=0\n"))
    try export_new(set, output) catch {
      _ => ()
    } noraise {
      _ => fail("published fused envelope was replaced")
    }
    assert_false(@fs.exists(parent + "/.candidate-set.lunaflux-fused-staging"))
  }
}
