///|
fn expect(
  object : Map[String, Json],
  key : String,
  value : Json,
) -> Unit raise ConfigError {
  guard object.get(key) == Some(value) else {
    raise InvalidConfig("unsupported or missing field: " + key)
  }
}

///|
fn optional(
  object : Map[String, Json],
  key : String,
  value : Json,
) -> Unit raise ConfigError {
  if object.get(key) is Some(actual) && actual != value {
    raise InvalidConfig("unsupported optional field: " + key)
  }
}

///|
fn fields(
  object : Map[String, Json],
  expected : Map[String, Json],
  extra : Array[String],
) -> Unit raise ConfigError {
  for key, value in expected {
    expect(object, key, value)
  }
  for key, _ in object {
    guard expected.contains(key) || extra.contains(key) else {
      raise InvalidConfig("unknown semantic field: " + key)
    }
  }
}

///|
fn child(
  object : Map[String, Json],
  key : String,
) -> Map[String, Json] raise ConfigError {
  guard object.get(key) is Some(Object(value)) else {
    raise InvalidConfig("missing object: " + key)
  }
  value
}

///|
fn integer(object : Map[String, Json], key : String) -> Int32 raise ConfigError {
  guard object.get(key) is Some(Number(value, ..)) &&
    value > 0.0 &&
    value <= 2147483647.0 &&
    value.trunc() == value else {
    raise InvalidConfig("invalid dimension: " + key)
  }
  value.to_int()
}

///|
fn text_geometry(text : Map[String, Json]) -> @hybrid.Geometry raise {
  let expected : Map[String, Json] = {
    "attention_bias": false,
    "attention_dropout": 0,
    "attn_output_gate": true,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "moe_intermediate_size": 512,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "output_router_logits": false,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 0.000001,
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 512,
    "vocab_size": 248144,
    "model_type": "qwen3_5_moe_text",
    "tie_word_embeddings": false,
  }
  fields(text, expected, ["layer_types", "rope_parameters"])
  let layers : Json = Json(
    Array::makei(40, i => {
      if i % 4 == 3 {
        "full_attention"
      } else {
        "linear_attention"
      }
    }),
  )
  expect(text, "layer_types", layers)
  fields(
    child(text, "rope_parameters"),
    {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default",
    },
    [],
  )
  @hybrid.Geometry::new(
    hidden_width=integer(text, "hidden_size"),
    vocabulary_size=integer(text, "vocab_size"),
    context_length=integer(text, "max_position_embeddings"),
    layer_count=integer(text, "num_hidden_layers"),
    full_interval=integer(text, "full_attention_interval"),
    full={
      query_heads: integer(text, "num_attention_heads"),
      kv_heads: integer(text, "num_key_value_heads"),
      head_width: integer(text, "head_dim"),
      output_gate: true,
    },
    recurrent={
      key_heads: integer(text, "linear_num_key_heads"),
      key_width: integer(text, "linear_key_head_dim"),
      value_heads: integer(text, "linear_num_value_heads"),
      value_width: integer(text, "linear_value_head_dim"),
      convolution_width: integer(text, "linear_conv_kernel_dim"),
    },
    experts={
      routed: integer(text, "num_experts"),
      selected: integer(text, "num_experts_per_tok"),
      intermediate_width: integer(text, "moe_intermediate_size"),
      shared_width: integer(text, "shared_expert_intermediate_size"),
    },
  )
}

///|
fn decode(root : Map[String, Json]) -> Spec raise {
  fields(
    root,
    {
      "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
      "dtype": "bfloat16",
      "model_type": "minicpmv4_7",
      "tie_word_embeddings": false,
      "use_cache": true,
      "insert_layer_id": 6,
      "mrope_mode": "canvas",
      "image_token_id": 248056,
      "video_token_id": 248057,
      "image_start_id": 248078,
      "image_end_id": 248079,
      "slice_start_id": 248088,
      "slice_end_id": 248089,
      "newline_id": 198,
      "version": 4.7,
    },
    [
      "text_config", "vision_config", "transformers_version", "downsample_mode",
      "merge_kernel_size", "merger_times",
    ],
  )
  guard root.get("transformers_version") is Some(String(_)) else {
    raise InvalidConfig("missing source library version")
  }
  optional(root, "merge_kernel_size", Json([2, 2]))
  optional(root, "merger_times", Json(1))
  let downsample = match root.get("downsample_mode") {
    Some(String("4x")) => FourX
    Some(String("16x")) => SixteenX
    _ => raise InvalidConfig("unsupported visual downsample mode")
  }
  let vision = child(root, "vision_config")
  fields(
    vision,
    {
      "attention_dropout": 0,
      "hidden_act": "gelu_pytorch_tanh",
      "hidden_size": 1152,
      "image_size": 980,
      "intermediate_size": 4304,
      "layer_norm_eps": 0.000001,
      "model_type": "minicpmv4_7_vision",
      "num_attention_heads": 16,
      "num_channels": 3,
      "num_hidden_layers": 27,
      "patch_size": 14,
    },
    ["insert_layer_id", "window_kernel_size"],
  )
  optional(vision, "insert_layer_id", Json(6))
  optional(vision, "window_kernel_size", Json([2, 2]))
  {
    decoder: text_geometry(child(root, "text_config")),
    vision: {
      hidden_width: integer(vision, "hidden_size"),
      intermediate_width: integer(vision, "intermediate_size"),
      layers: integer(vision, "num_hidden_layers"),
      heads: integer(vision, "num_attention_heads"),
      channels: integer(vision, "num_channels"),
      position_grid_side: integer(vision, "image_size") /
      integer(vision, "patch_size"),
      patch_size: integer(vision, "patch_size"),
      compression_after_layer: integer(root, "insert_layer_id"),
      window_side: 2,
      merge_rounds: 1,
    },
    markers: {
      image: 248056,
      video: 248057,
      image_start: 248078,
      image_end: 248079,
      slice_start: 248088,
      slice_end: 248089,
      newline: 198,
    },
    rotary: {
      theta: 10000000.0,
      partial_factor: 0.25,
      temporal_sections: 11,
      height_sections: 11,
      width_sections: 10,
    },
    downsample,
  }
}

///|
/// Parses only the selected model semantics. No source code, file access,
/// model weights, cryptography, device allocation or execution is involved.
pub fn Spec::parse(document : BytesView) -> Spec raise {
  guard document.length() <= 65536 else {
    raise InvalidConfig("config exceeds byte limit")
  }
  let text = @utf8.decode(document)
  @json_guard.reject_duplicate_object_keys(text, max_depth=16)
  guard @json.parse(text) is Object(root) else {
    raise InvalidConfig("expected model object")
  }
  decode(root)
}
