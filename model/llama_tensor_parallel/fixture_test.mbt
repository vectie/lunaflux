///|
fn tp_append_raw_tensor(
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
fn tp_fixture(
  vocabulary? : Int = 16,
  hidden? : Int = 8,
  intermediate? : Int = 12,
  layers? : Int = 1,
  attention_heads? : Int = 4,
  kv_heads? : Int = 2,
  content? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> (@plan.ModelPlan, @llama_weights.LlamaWeightBindings) raise {
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
  let model = @spec.LlamaModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(content),
  )
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let position = Ref(0L)
  tp_append_raw_tensor(
    entries,
    "model.embed_tokens.weight",
    [vocabulary.to_int64(), hidden.to_int64()],
    position,
  )
  let kv_width = kv_heads * (hidden / attention_heads)
  for layer in 0..<layers {
    let prefix = "model.layers.\{layer}"
    tp_append_raw_tensor(
      entries,
      "\{prefix}.input_layernorm.weight",
      [hidden.to_int64()],
      position,
    )
    tp_append_raw_tensor(
      entries,
      "\{prefix}.self_attn.q_proj.weight",
      [hidden.to_int64(), hidden.to_int64()],
      position,
    )
    for name in ["k_proj", "v_proj"] {
      tp_append_raw_tensor(
        entries,
        "\{prefix}.self_attn.\{name}.weight",
        [kv_width.to_int64(), hidden.to_int64()],
        position,
      )
    }
    tp_append_raw_tensor(
      entries,
      "\{prefix}.self_attn.o_proj.weight",
      [hidden.to_int64(), hidden.to_int64()],
      position,
    )
    tp_append_raw_tensor(
      entries,
      "\{prefix}.post_attention_layernorm.weight",
      [hidden.to_int64()],
      position,
    )
    for name in ["gate_proj", "up_proj"] {
      tp_append_raw_tensor(
        entries,
        "\{prefix}.mlp.\{name}.weight",
        [intermediate.to_int64(), hidden.to_int64()],
        position,
      )
    }
    tp_append_raw_tensor(
      entries,
      "\{prefix}.mlp.down_proj.weight",
      [hidden.to_int64(), intermediate.to_int64()],
      position,
    )
  }
  tp_append_raw_tensor(
    entries,
    "model.norm.weight",
    [hidden.to_int64()],
    position,
  )
  tp_append_raw_tensor(
    entries,
    "lm_head.weight",
    [vocabulary.to_int64(), hidden.to_int64()],
    position,
  )
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
  let plan = @llama.build(model)
  (plan, @llama_weights.bind(model, plan, tensors))
}

///|
fn tp_plan(
  model : @plan.ModelPlan,
  bindings : @llama_weights.LlamaWeightBindings,
  rank : Int,
  world_size : Int,
) -> @tensor_parallel_plan.TensorParallelRankPlan raise LlamaTensorParallelError {
  build_rank_plan(
    model,
    bindings,
    rank~,
    world_size~,
    alignment_bytes=16L,
    max_arena_bytes=1048576L,
  )
}
