///|
fn is_known_root_field(name : StringView) -> Bool {
  match name {
    "architectures"
    | "image_token_id"
    | "model_type"
    | "quantization_config"
    | "text_config"
    | "tie_word_embeddings"
    | "transformers_version"
    | "video_token_id"
    | "vision_config"
    | "vision_end_token_id"
    | "vision_start_token_id" => true
    _ => false
  }
}

///|
fn is_known_text_field(name : StringView) -> Bool {
  match name {
    "attention_bias"
    | "attention_dropout"
    | "attn_output_gate"
    | "bos_token_id"
    | "dtype"
    | "eos_token_id"
    | "full_attention_interval"
    | "head_dim"
    | "hidden_act"
    | "hidden_size"
    | "initializer_range"
    | "layer_types"
    | "linear_conv_kernel_dim"
    | "linear_key_head_dim"
    | "linear_num_key_heads"
    | "linear_num_value_heads"
    | "linear_value_head_dim"
    | "mamba_ssm_dtype"
    | "max_position_embeddings"
    | "model_type"
    | "moe_intermediate_size"
    | "mtp_num_hidden_layers"
    | "mtp_use_dedicated_embeddings"
    | "num_attention_heads"
    | "num_experts"
    | "num_experts_per_tok"
    | "num_hidden_layers"
    | "num_key_value_heads"
    | "output_router_logits"
    | "pad_token_id"
    | "partial_rotary_factor"
    | "rms_norm_eps"
    | "rope_parameters"
    | "router_aux_loss_coef"
    | "shared_expert_intermediate_size"
    | "tie_word_embeddings"
    | "use_cache"
    | "vocab_size" => true
    _ => false
  }
}

///|
fn is_known_rope_field(name : StringView) -> Bool {
  match name {
    "mrope_interleaved"
    | "mrope_section"
    | "partial_rotary_factor"
    | "rope_theta"
    | "rope_type" => true
    _ => false
  }
}

///|
fn is_known_quantization_field(name : StringView) -> Bool {
  match name {
    "activation_scheme"
    | "fmt"
    | "modules_to_not_convert"
    | "quant_method"
    | "weight_block_size" => true
    _ => false
  }
}

///|
fn validate_known_fields(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
) -> Unit raise Qwen35ConfigError {
  let mut index = 0
  for name, _ in object {
    let known = match area {
      Root => is_known_root_field(name)
      Text => is_known_text_field(name)
      Rope => is_known_rope_field(name)
      Quantization => is_known_quantization_field(name)
    }
    if !known {
      raise UnknownField(area~, index~)
    }
    index += 1
  }
}

///|
fn expect_single_architecture(
  object : Map[String, Json],
) -> Unit raise Qwen35ConfigError {
  match require_value(object, Root, "architectures", Architectures) {
    Array(values) => {
      if values.length() != 1 {
        raise invalid(Root, Architectures, UnsupportedValue)
      }
      match values[0] {
        String("Qwen3_5MoeForConditionalGeneration") => ()
        String(_) => raise invalid(Root, Architectures, UnsupportedValue)
        _ => raise invalid(Root, Architectures, InvalidArrayElement)
      }
    }
    _ => raise invalid(Root, Architectures, ExpectedArray)
  }
}

///|
fn expect_int_array(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
  expected : ArrayView[Int],
) -> Unit raise Qwen35ConfigError {
  match require_value(object, area, name, field) {
    Array(values) => {
      if values.length() != expected.length() {
        raise invalid(area, field, UnsupportedValue)
      }
      for index in 0..<values.length() {
        let value = exact_int64(values[index], area, field, index~)
        if value != Int64::from_int(expected[index]) {
          raise invalid(area, field, UnsupportedValue, index~)
        }
      }
    }
    _ => raise invalid(area, field, ExpectedArray)
  }
}

///|
fn validate_layer_types(
  text : Map[String, Json],
) -> Unit raise Qwen35ConfigError {
  match require_value(text, Text, "layer_types", LayerTypes) {
    Array(values) => {
      if values.length() != 40 {
        raise invalid(Text, LayerTypes, UnsupportedValue)
      }
      for index in 0..<values.length() {
        let expected = if index % 4 == 3 {
          "full_attention"
        } else {
          "linear_attention"
        }
        match values[index] {
          String(value) if value == expected => ()
          String(_) => raise invalid(Text, LayerTypes, UnsupportedValue, index~)
          _ => raise invalid(Text, LayerTypes, InvalidArrayElement, index~)
        }
      }
    }
    _ => raise invalid(Text, LayerTypes, ExpectedArray)
  }
}

///|
fn validate_modules_not_converted(
  quantization : Map[String, Json],
) -> Unit raise Qwen35ConfigError {
  match
    require_value(
      quantization,
      Quantization,
      "modules_to_not_convert",
      ModulesNotConverted,
    ) {
    Array(values) => {
      let mut has_embedding = false
      let mut has_lm_head = false
      for index in 0..<values.length() {
        match values[index] {
          String("model.embed_tokens") => has_embedding = true
          String("lm_head") => has_lm_head = true
          String(_) => ()
          _ =>
            raise invalid(
              Quantization,
              ModulesNotConverted,
              InvalidArrayElement,
              index~,
            )
        }
      }
      if !has_embedding || !has_lm_head {
        raise invalid(Quantization, ModulesNotConverted, MissingRequiredEntry)
      }
    }
    _ => raise invalid(Quantization, ModulesNotConverted, ExpectedArray)
  }
}

///|
fn validate_rope(rope : Map[String, Json]) -> Unit raise Qwen35ConfigError {
  validate_known_fields(rope, Rope)
  expect_bool(rope, Rope, "mrope_interleaved", RopeInterleaved, true)
  expect_int_array(rope, Rope, "mrope_section", RopeSections, [11, 11, 10])
  expect_number(rope, Rope, "partial_rotary_factor", PartialRotaryFactor, 0.25)
  expect_number(rope, Rope, "rope_theta", RopeTheta, 10000000.0)
  expect_string(rope, Rope, "rope_type", RopeType, "default")
}

///|
fn validate_quantization(
  quantization : Map[String, Json],
) -> Unit raise Qwen35ConfigError {
  validate_known_fields(quantization, Quantization)
  expect_string(
    quantization,
    Quantization,
    "activation_scheme",
    ActivationScheme,
    "dynamic",
  )
  expect_string(quantization, Quantization, "fmt", Fp8Format, "e4m3")
  expect_string(
    quantization,
    Quantization,
    "quant_method",
    QuantizationMethod,
    "fp8",
  )
  expect_int_array(
    quantization,
    Quantization,
    "weight_block_size",
    WeightBlockSize,
    [128, 128],
  )
  validate_modules_not_converted(quantization)
}
