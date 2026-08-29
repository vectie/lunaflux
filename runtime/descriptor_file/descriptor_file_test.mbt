///|
fn descriptor_fixture(
  kernel_manifest_locator? : String = "execution.json",
) -> String {
  let before =
    #|{"schema_version":"lunaflux.runtime.v1","model":{"config_locator":"config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","weights_locator":"model.safetensors","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","max_batch_rows":1},"kernels":{"manifest_locator":"
  let after =
    #|","manifest_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"deployment_approved_aot_only","admitted_bootstrap_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":0,"supports_bf16":true,"supports_cublas_lt":false,"tokens_per_page":4,"total_page_count":8,"model_generation":1},"worker_limits":{"max_prefill_rows":1,"max_decode_rows":1,"max_plan_rows":1,"max_plan_tokens":8,"max_plan_pages":2,"max_capabilities":16,"max_completion_slots":1,"max_sequence_tokens":8,"max_token_id":7},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":8,"max_new_tokens":8,"max_context_tokens":8,"max_token_id":7,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096}}
  before + kernel_manifest_locator + after
}

///|
fn descriptor_digest(
  text : String,
) -> @descriptor_file.RuntimeDescriptorDigest raise {
  @descriptor_file.RuntimeDescriptorDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(text))),
  )
}

///|
async fn with_descriptor_roots(
  text : String,
  body : (String, @approved_fs.ApprovedRoot, @approved_fs.ApprovedRoot) -> Unit raise @descriptor_file.RuntimeDescriptorFileError,
) -> Unit {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-runtime-descriptor")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    @fs.write_file(
      "\{directory}/runtime.json",
      @utf8.encode(text),
      create_mode=CreateNew,
    )
    let label = @fs.realpath(directory)
    let model_root = @approved_fs.ApprovedRoot::open_absolute(label)
    let kernel_root = @approved_fs.ApprovedRoot::open_absolute(label)
    body(label, model_root, kernel_root)
    if !kernel_root.is_closed() {
      kernel_root.close()
    }
    if !model_root.is_closed() {
      model_root.close()
    }
  }
}

///|
async test "descriptor digest mismatch fails before parsing and retains roots" {
  let text = "not json"
  with_descriptor_roots(text, (label, model_root, kernel_root) => {
    let wrong = @descriptor_file.RuntimeDescriptorDigest::from_sha256(
      "f123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )
    try
      @descriptor_file.load(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        wrong,
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=4096L,
          max_json_depth=16,
        ),
      )
    catch {
      DigestMismatch => ()
      _ => abort("unexpected descriptor digest failure")
    } noraise {
      _ => abort("mismatched descriptor digest was admitted")
    }
    if model_root.is_closed() || kernel_root.is_closed() {
      abort("descriptor loader consumed caller roots")
    }
  })
}

///|
async test "descriptor rejects duplicate and unknown fields before descendants" {
  let canonical = descriptor_fixture()
  let hostile : Array[(String, String)] = [
    (
      canonical.replace(
        old="\"schema_version\":\"lunaflux.runtime.v1\"",
        new="\"schema_version\":\"lunaflux.runtime.v1\",\"schema_version\":\"lunaflux.runtime.v1\"",
      ),
      "duplicate",
    ),
    (
      canonical.replace(
        old="\"max_batch_rows\":1",
        new="\"max_batch_rows\":1,\"unknown\":0",
      ),
      "unknown",
    ),
  ]
  for item in hostile {
    let (text, kind) = item
    with_descriptor_roots(text, (label, model_root, kernel_root) => {
      try
        @descriptor_file.load(
          model_root,
          kernel_root,
          label,
          label,
          try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
          descriptor_digest(text),
          @descriptor_file.RuntimeDescriptorFileLimits::new(
            max_descriptor_bytes=16384L,
            max_json_depth=16,
          ),
        )
      catch {
        @descriptor_file.DuplicateField(_) if kind == "duplicate" => ()
        @descriptor_file.UnknownField(area=Model, index=_) if kind == "unknown" =>
          ()
        _ => abort("unexpected strict descriptor schema failure")
      } noraise {
        _ => abort("hostile descriptor schema was admitted")
      }
    })
  }
}

///|
async test "descriptor bounds and ambient kernel locator fail closed" {
  let canonical = descriptor_fixture()
  with_descriptor_roots(canonical, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        descriptor_digest(canonical),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=32L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.FileTooLarge(maximum=32L) => ()
      _ => abort("unexpected descriptor byte-bound failure")
    } noraise {
      _ => abort("oversized descriptor was admitted")
    }
  })
  let ambient = descriptor_fixture(kernel_manifest_locator="../execution.json")
  with_descriptor_roots(ambient, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        descriptor_digest(ambient),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.AdmissionRejected(KernelConfig) => ()
      _ => abort("unexpected ambient-kernel-locator failure")
    } noraise {
      _ => abort("ambient kernel locator was admitted")
    }
  })
}

///|
async test "valid descriptor reaches first missing pinned descendant" {
  let canonical = descriptor_fixture()
  with_descriptor_roots(canonical, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        descriptor_digest(canonical),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.AdmissionRejected(ModelConfig) => ()
      _ => abort("valid descriptor did not reach model-config admission")
    } noraise {
      _ => abort("descriptor without model files was admitted")
    }
  })
}

///|
async test "descriptor v2 graph ceiling is strict positive and versioned" {
  let v2 = descriptor_fixture()
    .replace(old="lunaflux.runtime.v1", new="lunaflux.runtime.v2")
    .replace(
      old="\"max_total_module_bytes\":4096",
      new="\"max_total_module_bytes\":4096,\"max_graph_capture_bytes\":8192",
    )
  with_descriptor_roots(v2, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        descriptor_digest(v2),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.AdmissionRejected(ModelConfig) => ()
      _ => abort("valid v2 ceiling failed before model admission")
    } noraise {
      _ => abort("descriptor without model files was admitted")
    }
  })
  let zero = v2.replace(
    old="\"max_graph_capture_bytes\":8192",
    new="\"max_graph_capture_bytes\":0",
  )
  with_descriptor_roots(zero, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        descriptor_digest(zero),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.AdmissionRejected(GraphMemory) => ()
      _ => abort("unexpected zero v2 graph-ceiling failure")
    } noraise {
      _ => abort("zero graph ceiling was admitted")
    }
  })
}
