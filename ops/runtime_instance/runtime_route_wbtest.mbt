///|
fn legacy_cross_descriptor_fixture() -> String {
  (
    #|{"schema_version":"lunaflux.runtime.v1","model":{"config_locator":"missing-hostile-config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","weights_locator":"missing-hostile-weights.safetensors","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","max_batch_rows":1},"kernels":{"manifest_locator":"missing-hostile-execution.json","manifest_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"deployment_approved_aot_only","admitted_bootstrap_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":0,"supports_bf16":true,"supports_cublas_lt":false,"tokens_per_page":4,"total_page_count":8,"model_generation":1},"worker_limits":{"max_prefill_rows":1,"max_decode_rows":1,"max_plan_rows":1,"max_plan_tokens":8,"max_plan_pages":2,"max_capabilities":16,"max_completion_slots":1,"max_sequence_tokens":8,"max_token_id":7},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":8,"max_new_tokens":8,"max_context_tokens":8,"max_token_id":7,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096}}
  )
}

///|
fn i8_cross_descriptor_fixture() -> String {
  (
    #|{"schema_version":"lunaflux.runtime.i8.v2","model":{"config_locator":"missing-hostile-config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","numeric_weights_locator":"missing-hostile-numeric.safetensors","numeric_weight_artifact_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","max_batch_rows":1},"kernels":{"manifest_locator":"missing-hostile-execution-i8.json","manifest_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"production_symmetric_i8_weight_only_per_output_channel_f32_scale_f32_accumulate_bf16_output_v1","admitted_bootstrap_v3_sha256":"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":9,"total_memory_bytes":25769803776,"supports_bf16":true,"supports_cublas_lt":true,"tokens_per_page":16,"total_page_count":4096,"model_generation":1},"worker_limits":{"max_prefill_rows":1,"max_decode_rows":1,"max_plan_rows":1,"max_plan_tokens":8,"max_plan_pages":2,"max_capabilities":16,"max_completion_slots":1,"max_sequence_tokens":8,"max_token_id":7},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":8,"max_new_tokens":8,"max_context_tokens":8,"max_token_id":7,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096}}
  )
}

///|
fn mistral_cross_descriptor_fixture() -> String {
  (
    #|{"schema_version":"lunaflux.runtime.mistral_bf16.v1","model":{"family":"mistral","config_locator":"missing-hostile-config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","numeric_weights_locator":"missing-hostile-numeric.safetensors","numeric_weight_artifact_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","weight_manifest_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","sliding_window_tokens":4096,"max_batch_rows":1},"kernels":{"manifest_locator":"missing-hostile-mistral-execution.json","manifest_sha256":"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"deployment_approved_aot_only","admitted_bootstrap_sha256":"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":9,"supports_bf16":true,"supports_cublas_lt":true,"tokens_per_page":16,"total_page_count":4096,"model_generation":1},"worker_limits":{"max_prefill_rows":1,"max_decode_rows":1,"max_plan_rows":1,"max_plan_tokens":8,"max_plan_pages":2,"max_capabilities":16,"max_completion_slots":1,"max_sequence_tokens":8,"max_token_id":31},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":8,"max_new_tokens":8,"max_context_tokens":8,"max_token_id":31,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096}}
  )
}

///|
async fn assert_cross_descriptor_rejected_before_descendants(
  launch_schema : String,
  recipe_field : String,
  descriptor : String,
) -> Unit {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-runtime-route")
    let base = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(base, recursive=true))
    }
    let model = "\{base}/model"
    let kernel = "\{base}/kernel"
    let policy = "\{base}/policy"
    let launch_root = "\{base}/launch"
    @fs.mkdir(model)
    @fs.mkdir(kernel)
    @fs.mkdir(policy)
    @fs.mkdir(launch_root)
    let descriptor_bytes = @utf8.encode(descriptor)
    let descriptor_sha = @crypto.bytes_to_hex_string(
      @crypto.sha256(descriptor_bytes),
    )
    @fs.write_file(
      "\{model}/runtime.json",
      descriptor_bytes,
      create_mode=CreateNew,
    )
    // The named config, weights, manifest, policy, and executable deliberately
    // do not exist. A recipe/schema mismatch must stop at descriptor admission.
    let launch_text = "{\"schema\":\"\{launch_schema}\"\{recipe_field},\"model_root\":\"\{model}\",\"kernel_root\":\"\{kernel}\",\"policy_root\":\"\{policy}\",\"runtime_descriptor\":{\"locator\":\"runtime.json\",\"sha256\":\"\{descriptor_sha}\"},\"instance_policy\":{\"locator\":\"missing-hostile-policy.json\",\"sha256\":\"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"worker_executable\":{\"path\":\"\{base}/missing-hostile-worker\",\"sha256\":\"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"luna_approval\":{\"mode\":\"none\"}}"
    let launch_bytes = @utf8.encode(launch_text)
    let launch_sha = @crypto.bytes_to_hex_string(@crypto.sha256(launch_bytes))
    @fs.write_file(
      "\{launch_root}/lunaflux.launch.json",
      launch_bytes,
      create_mode=CreateNew,
    )
    let launch = @launch_file.load_argument(
      "\{launch_root}#sha256=\{launch_sha}",
      @launch_file.LaunchFileLimits::new(
        maximum_bytes=65536L,
        maximum_json_depth=16,
      ),
    )
    let model_root = @approved_fs.ApprovedRoot::open_absolute(model)
    let kernel_root = @approved_fs.ApprovedRoot::open_absolute(kernel)
    let outcome = try {
      ignore(
        load_runtime_descriptor(
          launch,
          model_root,
          kernel_root,
          @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
          @descriptor_file.RuntimeDescriptorDigest::from_sha256(descriptor_sha),
          @descriptor_file.RuntimeDescriptorFileLimits::new(
            max_descriptor_bytes=65536L,
            max_json_depth=16,
          ),
          None,
        ),
      )
      0
    } catch {
      RuntimeInstanceFailed(DescriptorAdmission) => 1
      _ => 2
    }
    kernel_root.close()
    model_root.close()
    match outcome {
      1 => ()
      2 => abort("cross-recipe descriptor failed at the wrong stage")
      _ => abort("cross-recipe descriptor was admitted through fallback")
    }
  }
}

///|
async test "v1 launch never falls back to the I8 descriptor loader" {
  assert_cross_descriptor_rejected_before_descendants(
    "lunaflux.launch.v1",
    "",
    i8_cross_descriptor_fixture(),
  )
}

///|
async test "v2 I8 launch never falls back to the legacy descriptor loader" {
  assert_cross_descriptor_rejected_before_descendants(
    "lunaflux.launch.v2",
    ",\"runtime_recipe\":\"dense_llama_i8_paged_aot_v6\"",
    legacy_cross_descriptor_fixture(),
  )
}

///|
async test "v2 Mistral launch cannot cross decode legacy or I8 descriptors" {
  for
    descriptor in [
      legacy_cross_descriptor_fixture(),
      i8_cross_descriptor_fixture(),
    ] {
    assert_cross_descriptor_rejected_before_descendants(
      "lunaflux.launch.v2", ",\"runtime_recipe\":\"dense_mistral_bf16_paged_aot_v7\"",
      descriptor,
    )
  }
}

///|
async test "legacy and I8 launches cannot cross decode a Mistral descriptor" {
  assert_cross_descriptor_rejected_before_descendants(
    "lunaflux.launch.v1",
    "",
    mistral_cross_descriptor_fixture(),
  )
  assert_cross_descriptor_rejected_before_descendants(
    "lunaflux.launch.v2",
    ",\"runtime_recipe\":\"dense_llama_i8_paged_aot_v6\"",
    mistral_cross_descriptor_fixture(),
  )
}

///|
fn assert_forced_route_failure_never_probes_other_loader(
  recipe : @launch_file.LaunchRuntimeRecipe,
  expected_legacy_calls : Int,
  expected_i8_calls : Int,
  expected_mistral_calls : Int,
  expected_tensor_parallel_calls : Int,
  expected_fp8_calls : Int,
) -> Unit {
  let legacy_calls = Ref(0)
  let i8_calls = Ref(0)
  let mistral_calls = Ref(0)
  let tensor_parallel_calls = Ref(0)
  let fp8_calls = Ref(0)
  try
    route_runtime_recipe(
      recipe,
      () => {
        legacy_calls.val += 1
        raise fail(DescriptorAdmission)
      },
      () => {
        i8_calls.val += 1
        raise fail(DescriptorAdmission)
      },
      () => {
        mistral_calls.val += 1
        raise fail(DescriptorAdmission)
      },
      () => {
        tensor_parallel_calls.val += 1
        raise fail(DescriptorAdmission)
      },
      () => {
        fp8_calls.val += 1
        raise fail(DescriptorAdmission)
      },
    )
  catch {
    RuntimeInstanceFailed(DescriptorAdmission) => ()
    _ => abort("forced selected-loader failure was not propagated exactly")
  } noraise {
    _ => abort("forced selected-loader failure was admitted")
  }
  if legacy_calls.val != expected_legacy_calls {
    abort("legacy loader call count drifted")
  }
  if i8_calls.val != expected_i8_calls {
    abort("I8 loader call count drifted")
  }
  if mistral_calls.val != expected_mistral_calls {
    abort("Mistral loader call count drifted")
  }
  if tensor_parallel_calls.val != expected_tensor_parallel_calls {
    abort("tensor-parallel loader call count drifted")
  }
  if fp8_calls.val != expected_fp8_calls {
    abort("FP8 loader call count drifted")
  }
}

///|
test "selected legacy loader failure never probes I8 fallback" {
  assert_forced_route_failure_never_probes_other_loader(
    DenseLlamaPagedAotV5,
    1,
    0,
    0,
    0,
    0,
  )
}

///|
test "selected I8 loader failure never probes legacy fallback" {
  assert_forced_route_failure_never_probes_other_loader(
    DenseLlamaI8PagedAotV6,
    0,
    1,
    0,
    0,
    0,
  )
}

///|
test "selected Mistral loader failure never probes legacy or I8 fallback" {
  assert_forced_route_failure_never_probes_other_loader(
    DenseMistralBf16PagedAotV7,
    0,
    0,
    1,
    0,
    0,
  )
}

///|
test "selected approved TP loader failure never probes another recipe" {
  assert_forced_route_failure_never_probes_other_loader(
    DenseLlamaTensorParallelPagedAotApprovedV8,
    0,
    0,
    0,
    1,
    0,
  )
}

///|
test "selected approved FP8 loader failure never probes another recipe" {
  assert_forced_route_failure_never_probes_other_loader(
    DenseLlamaFp8ReusablePagedAotApprovedV9,
    0,
    0,
    0,
    0,
    1,
  )
}
