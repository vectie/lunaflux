///|
fn admission_weight_entry(
  name : String,
  shape : String,
  start : Int,
  end : Int,
) -> String {
  "\"\{name}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{start},\{end}]}"
}

///|
fn admission_weight_file() -> Bytes {
  let entries : Array[String] = [
    admission_weight_entry("model.embed_tokens.weight", "[8,4]", 0, 64),
    admission_weight_entry(
      "model.layers.0.input_layernorm.weight", "[4]", 64, 72,
    ),
    admission_weight_entry(
      "model.layers.0.self_attn.q_proj.weight", "[4,4]", 72, 104,
    ),
    admission_weight_entry(
      "model.layers.0.self_attn.k_proj.weight", "[2,4]", 104, 120,
    ),
    admission_weight_entry(
      "model.layers.0.self_attn.v_proj.weight", "[2,4]", 120, 136,
    ),
    admission_weight_entry(
      "model.layers.0.self_attn.o_proj.weight", "[4,4]", 136, 168,
    ),
    admission_weight_entry(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 168, 176,
    ),
    admission_weight_entry(
      "model.layers.0.mlp.gate_proj.weight", "[6,4]", 176, 224,
    ),
    admission_weight_entry(
      "model.layers.0.mlp.up_proj.weight", "[6,4]", 224, 272,
    ),
    admission_weight_entry(
      "model.layers.0.mlp.down_proj.weight", "[4,6]", 272, 320,
    ),
    admission_weight_entry("model.norm.weight", "[4]", 320, 328),
    admission_weight_entry("lm_head.weight", "[8,4]", 328, 392),
  ]
  let header = @utf8.encode("{" + entries.join(",") + "}")
  let output : Array[Byte] = []
  for shift in 0..<8 {
    output.push((header.length().to_uint64() >> (shift * 8)).to_byte())
  }
  for byte in header {
    output.push(byte)
  }
  for index in 0..<392 {
    output.push((index % 251).to_byte())
  }
  Bytes::from_array(output)
}

///|
fn admission_model(file : BytesView) -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=16,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(file)),
    ),
  )
}

///|
fn admission_execution_limits(
  operation_count : Int,
) -> @execution_manifest_file.ExecutionManifestFileLimits raise {
  let artifact = @artifact.KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=operation_count,
    max_module_bytes=1024L,
    max_total_module_bytes=1024L,
    max_function_symbol_bytes=32,
  )
  @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=65536L,
    max_json_depth=16,
    max_modules=1,
    max_operations=operation_count,
    device_memory=@device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=65536L,
    ),
    launch_contract=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=operation_count,
      max_operands_per_contract=16,
    ),
    artifact_file=@artifact_file.KernelArtifactFileLimits::new(
      max_manifest_bytes=65536L,
      max_json_depth=16,
      max_path_bytes=64,
      artifact~,
    ),
  )
}

///|
fn admission_bootstrap_limits() -> @device_step.DeviceBootstrapLimits raise {
  @device_step.DeviceBootstrapLimits::new(
    max_modules=1,
    max_entry_points=32,
    max_steps=32,
    max_operands_per_step=16,
    max_symbol_bytes=64,
    max_canonical_bytes=16384,
  )
}

///|
fn admission_sampling_source(
  sampling_runtime : @worker_wire.WorkerBootstrapSamplingRuntime,
) -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/model",
      kernel_root="/kernel",
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights.safetensors",
      config_digest_sha256="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="execution.json",
      manifest_digest_sha256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
      sampling_runtime~,
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
      tokens_per_page=4,
      total_page_count=8,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=4096,
      max_weight_file_bytes=4096L,
      max_weight_arena_bytes=4096L,
      max_activation_arena_bytes=65536L,
      max_kv_arena_bytes=65536L,
      max_execution_manifest_bytes=65536L,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
    ),
  )
}

///|
fn admission_startup(
  execution : @execution_manifest_file.PagedExecutionAdmission,
) -> @worker_wire.WorkerStartupContract raise {
  let manifest = @device_step.admit_device_worker_bootstrap(
    execution.blueprint(),
    execution.artifacts(),
    device_ordinal=0,
    admission_bootstrap_limits(),
  )
  manifest.worker_startup_contract(
    @worker_protocol.ModelPlanGeneration::first(),
    bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
    ),
    @worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=1,
      max_decode_rows=1,
      max_plan_rows=1,
      max_plan_tokens=4,
      max_plan_pages=2,
      max_capabilities=64,
      max_completion_slots=32,
      max_sequence_tokens=16,
      max_token_id=7,
    ),
    @inference.InferenceLimits::new(
      max_text_bytes=32,
      max_input_tokens=16,
      max_new_tokens=16,
      max_context_tokens=16,
      max_token_id=7,
      max_stop_token_ids=2,
      max_stop_strings=1,
      max_stop_string_bytes=8,
      max_trace_bytes=8,
      max_cache_scope_bytes=8,
      max_decoded_delta_bytes=8,
      max_deadline_millis=100UL,
      max_top_k=8,
      max_temperature=4.0,
    ),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn admission_startup_with_identity(
  base : @worker_wire.WorkerStartupContract,
  identity : @spec.ModelIdentity,
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity~,
    bootstrap_digest=base.bootstrap_digest(),
    bootstrap_source_digest=base.bootstrap_source_digest(),
    device_ordinal=base.device_ordinal(),
    model_generation_value=base.model_generation_value(),
    worker_limits=base.worker_limits(),
    inference_limits=base.inference_limits(),
    has_predecessor=base.has_predecessor(),
    predecessor_value=base.predecessor_value(),
  )
}

///|
async test "admission rejects swapped inspection and aggregate layout evidence" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-worker-admission")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = admission_weight_file()
    let module_bytes = b"\x01\x02\x03"
    let model = admission_model(file)
    let plan = @llama.build_paged(model)
    let target = @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    )
    let manifest = admission_manifest(plan, target, module_bytes)
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    @fs.write_file(
      "\{directory}/execution.json",
      @utf8.encode(manifest),
      create_mode=CreateNew,
    )
    @fs.write_file(
      "\{directory}/kernel.cubin",
      module_bytes,
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let locator = @approved_fs.ApprovedRelativeLocator::new(
      "weights.safetensors",
    )
    let file_limits = @device_materialize.DeviceWeightFileLimits::new(
      max_file_bytes=16384L,
      transfer_chunk_bytes=23,
      safetensors=@safetensors.HeaderLimits::new(
        max_header_bytes=8192,
        max_tensor_count=32,
        max_rank=2,
        max_name_length=128,
        max_payload_bytes=4096L,
        max_materialized_tensor_bytes=4096L,
      ),
    )
    let inspection64 = @device_materialize.inspect_file(
      root,
      model,
      plan,
      locator,
      limits=file_limits,
      alignment_bytes=64L,
      max_arena_bytes=4096L,
    )
    let inspection128 = @device_materialize.inspect_file(
      root,
      model,
      plan,
      locator,
      limits=file_limits,
      alignment_bytes=128L,
      max_arena_bytes=4096L,
    )
    assert_true(inspection64.layout() != inspection128.layout())
    let kv_layout = @device_layout.plan(
      plan.identity(),
      plan.kv_cache_geometry(),
      @device_layout.DeviceKvLimits::new(
        tokens_per_page=4,
        total_page_count=8,
        max_arena_bytes=16384L,
      ),
    )
    let step_limits = @device_step.DeviceStepLimits::new(
      max_rows=1,
      max_tokens=4,
      max_pages=2,
      max_sequence_tokens=16,
      max_token_id=7,
    )
    let execution_locator = @approved_fs.ApprovedRelativeLocator::new(
      "execution.json",
    )
    let expected_digest = @execution_manifest_file.ExecutionManifestDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(manifest))),
    )
    let execution_limits = admission_execution_limits(
      plan.operations().length(),
    )
    let execution64 = @execution_manifest_file.load_paged(
      root,
      execution_locator,
      expected_digest,
      plan,
      inspection64.layout(),
      kv_layout,
      target,
      step_limits,
      execution_limits,
    )
    let execution128 = @execution_manifest_file.load_paged(
      root,
      execution_locator,
      expected_digest,
      plan,
      inspection128.layout(),
      kv_layout,
      target,
      step_limits,
      execution_limits,
    )
    let startup64 = admission_startup(execution64)
    let startup128 = admission_startup(execution128)
    validate_authenticated_model_plan(model, startup64, execution64)
    let alternate_plan = @llama.build(model)
    let substituted_identity_startup = admission_startup_with_identity(
      startup64,
      alternate_plan.identity(),
    )
    assert_false(alternate_plan.identity() == plan.identity())
    try
      validate_authenticated_model_plan(
        model, substituted_identity_startup, execution64,
      )
    catch {
      Invalid(ModelIdentity) => ()
      _ => fail("unexpected injected startup identity error")
    } noraise {
      _ => fail("substituted rebuilt-plan identity was accepted")
    }
    let foreign_model = @spec.LlamaModelMetadata::from_verified_content(
      model.spec(),
      @spec.ContentDigest::from_sha256(
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
    )
    try
      validate_authenticated_model_plan(foreign_model, startup64, execution64)
    catch {
      Invalid(ModelIdentity) => ()
      _ => fail("unexpected substituted model-content error")
    } noraise {
      _ => fail("foreign model content was accepted for retained plan")
    }
    ignore(
      admit_plan(
        startup=startup64,
        model~,
        weight_inspection=inspection64,
        execution=execution64,
        bootstrap_limits=admission_bootstrap_limits(),
      ),
    )
    let sampling_policy = @device_step.admit_paged_embedded_cuda_greedy_sampling(
      plan,
      execution64.blueprint(),
      execution64.artifacts(),
    )
    assert_eq(
      admit_bootstrap_sampling_policy(
        admission_sampling_source(HostSamplingRuntime),
        plan,
        execution64,
      ),
      PagedHostSampling,
    )
    assert_eq(
      admit_bootstrap_sampling_policy(
        admission_sampling_source(EmbeddedCudaGreedyV1),
        plan,
        execution64,
      ),
      sampling_policy,
    )
    let greedy_worker = admit_plan(
      startup=startup64,
      model~,
      weight_inspection=inspection64,
      execution=execution64,
      bootstrap_limits=admission_bootstrap_limits(),
      sampling_policy~,
    )
    assert_eq(greedy_worker.sampling_policy, sampling_policy)
    ignore(
      admit_plan(
        startup=startup128,
        model~,
        weight_inspection=inspection128,
        execution=execution128,
        bootstrap_limits=admission_bootstrap_limits(),
      ),
    )
    try
      admit_plan(
        startup=startup64,
        model~,
        weight_inspection=inspection128,
        execution=execution64,
        bootstrap_limits=admission_bootstrap_limits(),
      )
    catch {
      Invalid(WeightEnvelope) => ()
      _ => fail("unexpected 128-to-64 evidence mismatch")
    } noraise {
      _ => fail("foreign inspection was paired with the 64-byte execution")
    }
    try
      admit_plan(
        startup=startup128,
        model~,
        weight_inspection=inspection64,
        execution=execution128,
        bootstrap_limits=admission_bootstrap_limits(),
      )
    catch {
      Invalid(WeightEnvelope) => ()
      _ => fail("unexpected 64-to-128 evidence mismatch")
    } noraise {
      _ => fail("foreign inspection was paired with the 128-byte execution")
    }
    root.close()
  }
}
