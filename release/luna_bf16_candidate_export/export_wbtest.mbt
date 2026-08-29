///|
fn export_fixture_set() -> @luna_bf16_kernel_producer.LunaBf16CandidateSet raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=3000,
    hidden_size=16,
    intermediate_size=64,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=4,
    context_length=256,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let plan = @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256(
        "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30",
      ),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=8,
      total_page_count=1,
      max_arena_bytes=1048576L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=1,
    max_page_table_entries=1,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="3".repeat(64),
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
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
}

///|
async test "new export atomically publishes exact canonical candidate envelope" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-bf16-candidate-export")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let set = export_fixture_set()
    let output = parent + "/candidate-set"
    let evidence = export_new(set, output)
    assert_eq(
      evidence.model_content_sha256(),
      set.model_identity().content().as_hex(),
    )
    assert_eq(
      evidence.model_plan_sha256(),
      set.model_identity().plan().as_hex(),
    )
    assert_eq(evidence.target(), "sm_120")
    assert_eq(evidence.operation_count(), 21)
    assert_eq(evidence.candidate_set_sha256(), set.candidate_set_sha256())
    assert_eq(evidence.inventory_sha256(), set.inventory_sha256())
    assert_eq(
      @fs.read_file(output + "/candidate-root/candidate-set.v1").binary(),
      set.candidate_set_bytes(),
    )
    assert_eq(
      @fs.read_file(output + "/candidate.files.sha256").binary(),
      set.inventory_bytes(),
    )
    let record = @utf8.decode(@fs.read_file(output + "/export.v1").binary())
    assert_true(record.contains("generated_lowerings=1\n"))
    assert_true(record.contains("caller_source_authority=0\n"))
    assert_true(record.contains("compiler_invoked=0\n"))
    assert_true(record.contains("device_opened=0\n"))
    assert_true(record.contains("runtime_authority=0\n"))
    assert_eq(
      @fs.readdir(output + "/candidate-root/sources", sort=true).length(),
      21,
    )
    assert_eq(
      @fs.readdir(output + "/candidate-root/recipes", sort=true).length(),
      21,
    )
    for index in 0..<set.operation_count() {
      let key = set.candidate_key(index).unwrap()
      assert_eq(
        @fs.read_file(output + "/candidate-root/sources/" + key + ".cu").binary(),
        set.candidate_source_bytes(index).unwrap(),
      )
      assert_eq(
        @fs.read_file(output + "/candidate-root/recipes/" + key + ".recipe").binary(),
        set.candidate_recipe_bytes(index).unwrap(),
      )
    }
  }
}

///|
async test "export rejects replacement and preserves first published bytes" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-bf16-candidate-no-replace")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let set = export_fixture_set()
    let output = parent + "/candidate-set"
    ignore(export_new(set, output))
    let before = @fs.read_file(output + "/candidate.files.sha256").binary()
    try export_new(set, output) catch {
      ExportFailed(OutputPath) => ()
      _ => fail("unexpected existing-output rejection")
    } noraise {
      _ => fail("expected existing-output rejection")
    }
    assert_eq(
      @fs.read_file(output + "/candidate.files.sha256").binary(),
      before,
    )
    assert_false(@fs.exists(parent + "/.candidate-set.lunaflux-staging"))
  }
}

///|
async test "export rejects noncanonical unsafe and occupied staging paths" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-bf16-candidate-path")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let set = export_fixture_set()
    let invalid = [
      "relative-output",
      parent + "/../candidate-set",
      parent + "/candidate set",
      parent + "/candidate@set",
    ]
    for path in invalid {
      try export_new(set, path) catch {
        ExportFailed(OutputPath) => ()
        _ => fail("unexpected invalid-path rejection")
      } noraise {
        _ => fail("expected invalid-path rejection")
      }
    }
    let staging = parent + "/.candidate-set.lunaflux-staging"
    @fs.mkdir(staging, permission=0o700)
    try export_new(set, parent + "/candidate-set") catch {
      ExportFailed(OutputPath) => ()
      _ => fail("unexpected occupied-staging rejection")
    } noraise {
      _ => fail("expected occupied-staging rejection")
    }
    assert_true(@fs.exists(staging))
    assert_false(@fs.exists(parent + "/candidate-set"))
  }
}
