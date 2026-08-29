///|
fn fixture_metadata(digit : Char) -> @spec.LlamaModelMetadata raise {
  let digest = digit.to_string() +
    "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=8,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(digest),
  )
}

///|
fn append_tensor(
  entries : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  offset : Ref[Int64],
) -> Unit {
  let start = offset.val
  let end = shape.fold(init=2L, (size, dimension) => size * dimension) + start
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype="BF16",
      shape~,
      data_start=start,
      data_end=end,
    ),
  )
  offset.val = end
}

///|
fn fixture_weight_layout(
  metadata : @spec.LlamaModelMetadata,
) -> @device_materialize.DeviceWeightLayout raise {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let offset = Ref(0L)
  append_tensor(entries, "model.embed_tokens.weight", [8L, 4L], offset)
  append_tensor(entries, "model.layers.0.input_layernorm.weight", [4L], offset)
  append_tensor(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [4L, 4L],
    offset,
  )
  append_tensor(
    entries,
    "model.layers.0.self_attn.k_proj.weight",
    [2L, 4L],
    offset,
  )
  append_tensor(
    entries,
    "model.layers.0.self_attn.v_proj.weight",
    [2L, 4L],
    offset,
  )
  append_tensor(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [4L, 4L],
    offset,
  )
  append_tensor(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [4L],
    offset,
  )
  append_tensor(
    entries,
    "model.layers.0.mlp.gate_proj.weight",
    [6L, 4L],
    offset,
  )
  append_tensor(entries, "model.layers.0.mlp.up_proj.weight", [6L, 4L], offset)
  append_tensor(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [4L, 6L],
    offset,
  )
  append_tensor(entries, "model.norm.weight", [4L], offset)
  append_tensor(entries, "lm_head.weight", [8L, 4L], offset)
  let tensors = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=16,
    payload_bytes=offset.val,
    limits=@safetensors.HeaderLimits::new(
      max_header_bytes=4096,
      max_tensor_count=32,
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=4096L,
      max_materialized_tensor_bytes=4096L,
    ),
  )
  @device_materialize.plan_layout(
    @llama_weights.bind(metadata, @llama.build(metadata), tensors),
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
}

///|
fn fixture_layout(
  plan : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=8,
      max_arena_bytes=8192L,
    ),
  )
}

///|
fn fixture_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn fixture_step_limits(rows? : Int = 1) -> @device_step.DeviceStepLimits raise {
  @device_step.DeviceStepLimits::new(
    max_rows=rows,
    max_tokens=4,
    max_pages=2,
    max_sequence_tokens=8,
    max_token_id=7,
  )
}

///|
fn fixture_limits(
  operations : Int,
  entry_points? : Int = operations,
) -> ExecutionManifestFileLimits raise {
  let artifact = @artifact.KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=entry_points,
    max_module_bytes=1024L,
    max_total_module_bytes=1024L,
    max_function_symbol_bytes=32,
  )
  ExecutionManifestFileLimits::new(
    max_manifest_bytes=65536L,
    max_json_depth=16,
    max_modules=1,
    max_operations=operations,
    device_memory=@device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=65536L,
    ),
    launch_contract=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=operations,
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
fn fixture_module_bytes() -> Bytes {
  b"\x01\x02\x03"
}

///|
fn fixture_kind_index(
  kinds : ArrayView[@plan.OperationKind],
  expected : @plan.OperationKind,
) -> Int? {
  for index, kind in kinds {
    if kind == expected {
      return Some(index)
    }
  }
  None
}

///|
fn fixture_manifest(
  plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
) -> String {
  let digest = @crypto.bytes_to_hex_string(
    @crypto.sha256(fixture_module_bytes()),
  )
  let mut entries = ""
  let mut operations = ""
  let kinds : Array[@plan.OperationKind] = []
  for index, operation in plan.operations() {
    if index > 0 {
      operations += ","
    }
    let stable = match fixture_kind_index(kinds, operation.kind()) {
      Some(kind_index) => kind_index + 1
      None => {
        kinds.push(operation.kind())
        let kind_index = kinds.length()
        if kind_index > 1 {
          entries += ","
        }
        entries += "{\"family_id\":" +
          kind_index.to_string() +
          ",\"entry_point_id\":" +
          kind_index.to_string() +
          ",\"symbol\":\"kernel_" +
          kind_index.to_string() +
          "\"}"
        kind_index
      }
    }
    let workspace = operation.workspace_bytes()
    operations += "{\"operation_id\":" +
      operation.id().as_int().to_string() +
      ",\"module_index\":0,\"family_id\":" +
      stable.to_string() +
      ",\"entry_point_id\":" +
      stable.to_string() +
      ",\"workspace_bytes\":" +
      workspace.to_string() +
      ",\"workspace_alignment\":" +
      (if workspace > 0L { "16" } else { "1" }) +
      ",\"dimensions\":{\"grid_x\":1,\"grid_y\":1,\"grid_z\":1," +
      "\"block_x\":32,\"block_y\":1,\"block_z\":1," +
      "\"shared_memory_bytes\":0}}"
  }
  "{\"schema_version\":2,\"model_identity\":{\"content_sha256\":\"" +
  plan.identity().content().as_hex() +
  "\",\"plan_sha256\":\"" +
  plan.identity().plan().as_hex() +
  "\"},\"device_target\":{" +
  "\"compute_major\":" +
  target.compute_major().to_string() +
  ",\"compute_minor\":" +
  target.compute_minor().to_string() +
  ",\"supports_bf16\":true,\"supports_cublas_lt\":false}," +
  "\"catalog_version\":3,\"profile\":{\"id\":1},\"modules\":[{" +
  "\"sha256\":\"" +
  digest +
  "\",\"path\":\"kernel.cubin\"," +
  "\"entry_points\":[" +
  entries +
  "]}],\"operations\":[" +
  operations +
  "]}"
}

///|
fn fixture_digest(value : StringView) -> ExecutionManifestDigest raise {
  ExecutionManifestDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(value))),
  )
}

///|
async fn write_fixture_file(
  root : String,
  name : String,
  bytes : Bytes,
) -> Unit {
  @fs.write_file("\{root}/\{name}", bytes, create_mode=CreateNew)
}
