///|
fn i8_descriptor_digest(
  text : String,
) -> @descriptor_file.RuntimeDescriptorDigest raise {
  @descriptor_file.RuntimeDescriptorDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(text))),
  )
}

///|
fn i8_public_descriptor_fixture() -> String {
  descriptor_fixture()
  .replace(old="lunaflux.runtime.v1", new="lunaflux.runtime.i8.v2")
  .replace(
    old="\"weights_locator\":\"model.safetensors\",\"content_sha256\":\"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
    new="\"content_sha256\":\"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"numeric_weights_locator\":\"model.numeric.safetensors\",\"numeric_weight_artifact_sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
  )
  .replace(
    old="\"policy\":\"deployment_approved_aot_only\",\"admitted_bootstrap_sha256\"",
    new="\"policy\":\"production_symmetric_i8_weight_only_per_output_channel_f32_scale_f32_accumulate_bf16_output_v1\",\"admitted_bootstrap_v3_sha256\"",
  )
  .replace(
    old="\"manifest_locator\":\"execution.json\"",
    new="\"manifest_locator\":\"execution-i8-v4.json\"",
  )
  .replace(
    old="\"compute_major\":8,\"compute_minor\":0,\"supports_bf16\":true,\"supports_cublas_lt\":false",
    new="\"compute_major\":8,\"compute_minor\":9,\"total_memory_bytes\":25769803776,\"supports_bf16\":true,\"supports_cublas_lt\":true",
  )
}

///|
async fn with_hostile_i8_descendants(
  text : String,
  body : (String, @approved_fs.ApprovedRoot, @approved_fs.ApprovedRoot) -> Unit raise @descriptor_file.RuntimeDescriptorFileError,
) -> Unit {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-runtime-i8-preflight")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    @fs.write_file(
      "\{directory}/runtime.json",
      @utf8.encode(text),
      create_mode=CreateNew,
    )
    @fs.write_file(
      "\{directory}/config.json",
      b"hostile-model-config",
      create_mode=CreateNew,
    )
    @fs.write_file(
      "\{directory}/model.numeric.safetensors",
      b"hostile-numeric-weights",
      create_mode=CreateNew,
    )
    @fs.write_file(
      "\{directory}/execution-i8-v4.json",
      b"hostile-execution-manifest",
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
fn assert_i8_preflight_error(kind : Int, error : Error) -> Unit {
  match (kind, error) {
    (0, @descriptor_file.AdmissionRejected(ModelConfig)) => ()
    (1, @descriptor_file.AdmissionRejected(ModelIdentity)) => ()
    (2, @descriptor_file.AdmissionRejected(NumericWeightInspection)) => ()
    (3, @descriptor_file.AdmissionRejected(ExecutionManifest)) => ()
    (4, @descriptor_file.AdmissionRejected(BootstrapDigest)) => ()
    (
      5,
      @descriptor_file.InvalidField(
        area=Kernels,
        issue=UnsupportedValue,
        index=2
      ),
    ) => ()
    (6, @descriptor_file.AdmissionRejected(ModelPlan)) => ()
    _ => abort("unexpected I8 scalar/lexical preflight failure kind=\{kind}")
  }
}

///|
async test "I8 scalar and lexical failures precede hostile descendants" {
  let canonical = i8_public_descriptor_fixture()
  let cases : Array[(String, Int)] = [
    (
      canonical.replace(
        old="\"config_locator\":\"config.json\"",
        new="\"config_locator\":\"../config.json\"",
      ),
      0,
    ),
    (
      canonical.replace(
        old="1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        new="X123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
      1,
    ),
    (
      canonical.replace(
        old="\"numeric_weights_locator\":\"model.numeric.safetensors\"",
        new="\"numeric_weights_locator\":\"../model.numeric.safetensors\"",
      ),
      2,
    ),
    (
      canonical.replace(
        old="\"manifest_locator\":\"execution-i8-v4.json\"",
        new="\"manifest_locator\":\"../execution-i8-v4.json\"",
      ),
      3,
    ),
    (
      canonical.replace(
        old="2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        new="X123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
      3,
    ),
    (
      canonical.replace(
        old="3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        new="X123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
      4,
    ),
    (
      canonical.replace(
        old="production_symmetric_i8_weight_only_per_output_channel_f32_scale_f32_accumulate_bf16_output_v1",
        new="deployment_approved_aot_only",
      ),
      5,
    ),
    (
      canonical.replace(old="\"max_batch_rows\":1", new="\"max_batch_rows\":2"),
      6,
    ),
  ]
  for _, item in cases {
    let (text, kind) = item
    with_hostile_i8_descendants(text, (label, model_root, kernel_root) => {
      try
        @descriptor_file.load_i8_v2(
          model_root,
          kernel_root,
          label,
          label,
          try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
          i8_descriptor_digest(text),
          @descriptor_file.RuntimeDescriptorFileLimits::new(
            max_descriptor_bytes=16384L,
            max_json_depth=16,
          ),
        )
      catch {
        error => assert_i8_preflight_error(kind, error)
      } noraise {
        _ => abort("hostile I8 scalar/lexical input reached descendants")
      }
    })
  }
}

///|
async test "I8 schema rejects duplicate keys, byte overflow, and depth before descendants" {
  let canonical = i8_public_descriptor_fixture()
  let duplicate = canonical.replace(
    old="\"schema_version\":\"lunaflux.runtime.i8.v2\"",
    new="\"schema_version\":\"lunaflux.runtime.i8.v2\",\"schema_version\":\"lunaflux.runtime.i8.v2\"",
  )
  with_hostile_i8_descendants(duplicate, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load_i8_v2(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        i8_descriptor_digest(duplicate),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.DuplicateField(_) => ()
      _ => abort("unexpected I8 duplicate-key failure")
    } noraise {
      _ => abort("duplicate I8 descriptor key reached descendants")
    }
  })
  with_hostile_i8_descendants(canonical, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load_i8_v2(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        i8_descriptor_digest(canonical),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=32L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.FileTooLarge(maximum=32L) => ()
      _ => abort("unexpected I8 descriptor byte-limit failure")
    } noraise {
      _ => abort("oversized I8 descriptor reached descendants")
    }
  })
  with_hostile_i8_descendants(canonical, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load_i8_v2(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        i8_descriptor_digest(canonical),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=1,
        ),
      )
    catch {
      @descriptor_file.JsonDepthExceeded(maximum=1) => ()
      _ => abort("unexpected I8 descriptor depth-limit failure")
    } noraise {
      _ => abort("over-depth I8 descriptor reached descendants")
    }
  })
}

///|
async test "I8 runtime v2 rejects legacy schema before descendants" {
  let legacy = descriptor_fixture()
  with_descriptor_roots(legacy, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load_i8_v2(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        i8_descriptor_digest(legacy),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.InvalidField(area=Root, issue=UnsupportedValue, index=0) =>
        ()
      _ => abort("unexpected legacy/I8 schema-crossing failure")
    } noraise {
      _ => abort("legacy runtime descriptor entered I8 v2 loader")
    }
    if model_root.is_closed() || kernel_root.is_closed() {
      abort("I8 descriptor loader consumed caller roots")
    }
  })
}

///|
async test "I8 runtime v2 validates device facts before model descendants" {
  let canonical = i8_public_descriptor_fixture()
  let unsupported = canonical.replace(
    old="\"compute_minor\":9",
    new="\"compute_minor\":0",
  )
  with_descriptor_roots(unsupported, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load_i8_v2(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        i8_descriptor_digest(unsupported),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.AdmissionRejected(DeviceCapability) => ()
      _ => abort("unexpected I8 device-fact failure")
    } noraise {
      _ => abort("unsupported I8 device facts reached model descendants")
    }
  })
}

///|
async test "valid I8 runtime v2 reaches first missing pinned model config" {
  let canonical = i8_public_descriptor_fixture()
  with_descriptor_roots(canonical, (label, model_root, kernel_root) => {
    try
      @descriptor_file.load_i8_v2(
        model_root,
        kernel_root,
        label,
        label,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        i8_descriptor_digest(canonical),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.AdmissionRejected(ModelConfig) => ()
      _ => abort("valid I8 descriptor did not reach model-config admission")
    } noraise {
      _ => abort("I8 descriptor without model files was admitted")
    }
  })
}
