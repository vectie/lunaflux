///|
fn release_test_limits() -> @runtime_instance.RuntimeInstanceFileLimits raise {
  @runtime_instance.RuntimeInstanceFileLimits::new(
    launch=@launch_file.LaunchFileLimits::new(
      maximum_bytes=65536L,
      maximum_json_depth=16,
    ),
    descriptor=@descriptor_file.RuntimeDescriptorFileLimits::new(
      max_descriptor_bytes=65536L,
      max_json_depth=16,
    ),
    policy=@instance_policy_file.InstancePolicyFileLimits::new(
      max_policy_bytes=65536L,
      max_json_depth=16,
    ),
    executable=@worker_executable_file.WorkerExecutableFileLimits::new(
      maximum_bytes=1048576L,
    ),
  )
}

///|
fn release_test_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
async test "release preflight rejects every unpinned or malformed deployment operand" {
  for
    argument in [
      "MODEL",
      "/absolute/deployment",
      "/absolute/deployment#sha256=ABC",
      "/absolute/deployment#sha256=" + "A".repeat(64),
      "/absolute/deployment/#sha256=" + "a".repeat(64),
    ] {
    try
      @runtime_instance.preflight_release(argument, release_test_limits())
    catch {
      @runtime_instance.RuntimeInstanceFailed(LaunchAdmission) => ()
      _ => abort("hostile deployment operand failed at the wrong release stage")
    } noraise {
      _ => abort("hostile deployment operand produced release evidence")
    }
  }
}

///|
async test "release preflight rejects malformed launch without descendant authority" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-release-blackbox-launch")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(temporary, recursive=true))
    }
    let launch = b"{}"
    @fs.write_file(
      "\{temporary}/lunaflux.launch.json",
      launch,
      create_mode=CreateNew,
    )
    let root = @fs.realpath(temporary)
    try
      @runtime_instance.preflight_release(
        "\{root}#sha256=\{release_test_sha256(launch)}",
        release_test_limits(),
      )
    catch {
      @runtime_instance.RuntimeInstanceFailed(LaunchAdmission) => ()
      _ => abort("malformed launch failed at the wrong release stage")
    } noraise {
      _ => abort("malformed launch produced release evidence")
    }
  }
}

///|
fn legacy_descriptor_at_i8_route() -> String {
  (
    #|{"schema_version":"lunaflux.runtime.v1","model":{"config_locator":"missing-config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","weights_locator":"missing.safetensors","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","max_batch_rows":1},"kernels":{"manifest_locator":"missing-execution.json","manifest_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"deployment_approved_aot_only","admitted_bootstrap_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":0,"supports_bf16":true,"supports_cublas_lt":false,"tokens_per_page":4,"total_page_count":8,"model_generation":1},"worker_limits":{"max_prefill_rows":1,"max_decode_rows":1,"max_plan_rows":1,"max_plan_tokens":8,"max_plan_pages":2,"max_capabilities":16,"max_completion_slots":1,"max_sequence_tokens":8,"max_token_id":7},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":8,"max_new_tokens":8,"max_context_tokens":8,"max_token_id":7,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096}}
  )
}

///|
async test "release preflight never cross-decodes an authenticated runtime recipe" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-release-blackbox-route")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(temporary, recursive=true))
    }
    let base = @fs.realpath(temporary)
    let model = "\{base}/model"
    let kernel = "\{base}/kernel"
    let policy = "\{base}/policy"
    let launch_root = "\{base}/launch"
    @fs.mkdir(model)
    @fs.mkdir(kernel)
    @fs.mkdir(policy)
    @fs.mkdir(launch_root)
    let descriptor = @utf8.encode(legacy_descriptor_at_i8_route())
    let descriptor_sha = release_test_sha256(descriptor)
    @fs.write_file("\{model}/runtime.json", descriptor, create_mode=CreateNew)
    let launch = @utf8.encode(
      "{\"schema\":\"lunaflux.launch.v2\",\"runtime_recipe\":\"dense_llama_i8_paged_aot_v6\",\"model_root\":\"\{model}\",\"kernel_root\":\"\{kernel}\",\"policy_root\":\"\{policy}\",\"runtime_descriptor\":{\"locator\":\"runtime.json\",\"sha256\":\"\{descriptor_sha}\"},\"instance_policy\":{\"locator\":\"missing-policy.json\",\"sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"worker_executable\":{\"path\":\"\{base}/missing-worker\",\"sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"luna_approval\":{\"mode\":\"none\"}}",
    )
    @fs.write_file(
      "\{launch_root}/lunaflux.launch.json",
      launch,
      create_mode=CreateNew,
    )
    try
      @runtime_instance.preflight_release(
        "\{launch_root}#sha256=\{release_test_sha256(launch)}",
        release_test_limits(),
      )
    catch {
      @runtime_instance.RuntimeInstanceFailed(DescriptorAdmission) => ()
      _ => abort("cross-recipe descriptor failed at the wrong release stage")
    } noraise {
      _ => abort("cross-recipe descriptor produced release evidence")
    }
  }
}
