///|
fn bridge_append_tensor(
  entries : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  position : Ref[Int64],
) -> Unit {
  let mut elements = 1L
  for dimension in shape {
    elements = elements * dimension
  }
  let bytes = elements * 2L
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype="BF16",
      shape~,
      data_start=position.val,
      data_end=position.val + bytes,
    ),
  )
  position.val = position.val + bytes
}

///|
fn bridge_fixture(
  content? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> (@plan.ModelPlan, @llama_weights.LlamaWeightBindings) raise {
  let vocabulary = 16
  let hidden = 8
  let intermediate = 12
  let layers = 1
  let attention_heads = 4
  let kv_heads = 2
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=vocabulary,
    hidden_size=hidden,
    intermediate_size=intermediate,
    layer_count=layers,
    attention_head_count=attention_heads,
    kv_head_count=kv_heads,
    context_length=32,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(content),
  )
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let position = Ref(0L)
  bridge_append_tensor(
    entries,
    "model.embed_tokens.weight",
    [16L, 8L],
    position,
  )
  bridge_append_tensor(
    entries,
    "model.layers.0.input_layernorm.weight",
    [8L],
    position,
  )
  bridge_append_tensor(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [8L, 8L],
    position,
  )
  for name in ["k_proj", "v_proj"] {
    bridge_append_tensor(
      entries,
      "model.layers.0.self_attn.\{name}.weight",
      [4L, 8L],
      position,
    )
  }
  bridge_append_tensor(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [8L, 8L],
    position,
  )
  bridge_append_tensor(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [8L],
    position,
  )
  for name in ["gate_proj", "up_proj"] {
    bridge_append_tensor(
      entries,
      "model.layers.0.mlp.\{name}.weight",
      [12L, 8L],
      position,
    )
  }
  bridge_append_tensor(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [8L, 12L],
    position,
  )
  bridge_append_tensor(entries, "model.norm.weight", [8L], position)
  bridge_append_tensor(entries, "lm_head.weight", [16L, 8L], position)
  let tensors = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=1024,
    payload_bytes=position.val,
    limits=@safetensors.HeaderLimits::new(
      max_header_bytes=4096,
      max_tensor_count=entries.length(),
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=position.val,
      max_materialized_tensor_bytes=position.val,
    ),
  )
  let plan = @llama.build(metadata)
  (plan, @llama_weights.bind(metadata, plan, tensors))
}

///|
fn bridge_rank_plan(
  model : @plan.ModelPlan,
  bindings : @llama_weights.LlamaWeightBindings,
  rank : Int,
  world_size : Int,
) -> @tensor_parallel_plan.TensorParallelRankPlan raise {
  @llama_tensor_parallel.build_rank_plan(
    model,
    bindings,
    rank~,
    world_size~,
    alignment_bytes=16L,
    max_arena_bytes=1048576L,
  )
}
