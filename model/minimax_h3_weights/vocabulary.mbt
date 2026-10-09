///|
fn bounded_index(text : StringView, upper : Int) -> Bool {
  let value = @string.parse_int(text, base=10) catch { _ => return false }
  value >= 0 && value < upper
}

///|
fn append_weight_bias(names : Array[String], prefix : String) -> Unit {
  names.push("\{prefix}.weight")
  names.push("\{prefix}.bias")
}

///|
fn expected_denoiser_names() -> Array[String] {
  let names : Array[String] = []
  for
    prefix in [
      "audio_proj_in", "audio_proj_out", "context_embedder", "proj_in", "proj_out",
      "time_embedder.linear_1", "time_embedder.linear_2", "norm_out.linear",
    ] {
    append_weight_bias(names, prefix)
  }
  names.push("norm_out.norm.weight")
  names.push("token_refiner.final_norm.weight")
  for layer in 0..<2 {
    let prefix = "token_refiner.blocks.\{layer}"
    for
      suffix in [
        "attn.norm_k.weight", "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight",
        "attn.to_q.weight", "attn.to_v.weight", "ff.net.0.proj.weight", "ff.net.2.weight",
        "norm1.weight", "norm2.weight",
      ] {
      names.push("\{prefix}.\{suffix}")
    }
  }
  for layer in 0..<50 {
    let prefix = "transformer_blocks.\{layer}"
    for
      suffix in [
        "adaln_proj.linear.bias", "adaln_proj.linear.weight", "attn.norm_k.weight",
        "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight", "attn.to_q.weight",
        "attn.to_v.weight", "ff.net.0.proj.weight", "ff.net.2.weight", "norm1.weight",
        "norm2.weight",
      ] {
      names.push("\{prefix}.\{suffix}")
    }
  }
  names
}

///|
fn append_vision_merger(names : Array[String], prefix : String) -> Unit {
  for
    suffix in [
      "norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight",
      "linear_fc2.bias",
    ] {
    names.push("\{prefix}.\{suffix}")
  }
}

///|
fn expected_text_names() -> Array[String] {
  let names : Array[String] = [
    "lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight",
    "model.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.bias",
    "model.visual.pos_embed.weight",
  ]
  for layer in 0..<64 {
    let prefix = "model.language_model.layers.\{layer}"
    for
      suffix in [
        "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight",
        "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight",
        "mlp.up_proj.weight", "mlp.down_proj.weight",
      ] {
      names.push("\{prefix}.\{suffix}")
    }
  }
  for layer in 0..<27 {
    let prefix = "model.visual.blocks.\{layer}"
    for
      suffix in [
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn.qkv.weight",
        "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias", "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
      ] {
      names.push("\{prefix}.\{suffix}")
    }
  }
  append_vision_merger(names, "model.visual.merger")
  for index in 0..<3 {
    append_vision_merger(names, "model.visual.deepstack_merger_list.\{index}")
  }
  names
}

///|
fn parts_match(
  parts : Array[StringView],
  offset : Int,
  expected : Array[String],
) -> Bool {
  if parts.length() != offset + expected.length() {
    return false
  }
  for index, value in expected {
    if parts[offset + index] != value {
      return false
    }
  }
  true
}

///|
fn denoiser_fixed(name : StringView) -> MiniMaxH3TensorRole? {
  match name {
    "audio_proj_in.weight"
    | "audio_proj_in.bias"
    | "audio_proj_out.weight"
    | "audio_proj_out.bias"
    | "context_embedder.weight"
    | "context_embedder.bias"
    | "proj_in.weight"
    | "proj_in.bias"
    | "proj_out.weight"
    | "proj_out.bias" => Some(DenoiserProjection)
    "time_embedder.linear_1.weight"
    | "time_embedder.linear_1.bias"
    | "time_embedder.linear_2.weight"
    | "time_embedder.linear_2.bias" => Some(DenoiserTimeEmbedding)
    "norm_out.linear.weight"
    | "norm_out.linear.bias"
    | "norm_out.norm.weight"
    | "token_refiner.final_norm.weight" => Some(DenoiserNormalization)
    _ => None
  }
}

///|
fn denoiser_block(parts : Array[StringView]) -> MiniMaxH3TensorRole? {
  if parts.length() < 4 ||
    parts[0] != "transformer_blocks" ||
    !bounded_index(parts[1], 50) {
    return None
  }
  if parts_match(parts, 2, ["adaln_proj", "linear", "weight"]) ||
    parts_match(parts, 2, ["adaln_proj", "linear", "bias"]) ||
    parts_match(parts, 2, ["norm1", "weight"]) ||
    parts_match(parts, 2, ["norm2", "weight"]) {
    return Some(DenoiserNormalization)
  }
  if parts_match(parts, 2, ["attn", "norm_k", "weight"]) ||
    parts_match(parts, 2, ["attn", "norm_q", "weight"]) ||
    parts_match(parts, 2, ["attn", "to_k", "weight"]) ||
    parts_match(parts, 2, ["attn", "to_q", "weight"]) ||
    parts_match(parts, 2, ["attn", "to_v", "weight"]) ||
    parts_match(parts, 2, ["attn", "to_out", "0", "weight"]) {
    return Some(DenoiserAttention)
  }
  if parts_match(parts, 2, ["ff", "net", "0", "proj", "weight"]) ||
    parts_match(parts, 2, ["ff", "net", "2", "weight"]) {
    return Some(DenoiserFeedForward)
  }
  None
}

///|
fn refiner_block(parts : Array[StringView]) -> MiniMaxH3TensorRole? {
  if parts.length() < 5 ||
    parts[0] != "token_refiner" ||
    parts[1] != "blocks" ||
    !bounded_index(parts[2], 2) {
    return None
  }
  if parts_match(parts, 3, ["norm1", "weight"]) ||
    parts_match(parts, 3, ["norm2", "weight"]) {
    return Some(DenoiserNormalization)
  }
  if parts_match(parts, 3, ["attn", "norm_k", "weight"]) ||
    parts_match(parts, 3, ["attn", "norm_q", "weight"]) ||
    parts_match(parts, 3, ["attn", "to_k", "weight"]) ||
    parts_match(parts, 3, ["attn", "to_q", "weight"]) ||
    parts_match(parts, 3, ["attn", "to_v", "weight"]) ||
    parts_match(parts, 3, ["attn", "to_out", "0", "weight"]) {
    return Some(DenoiserAttention)
  }
  if parts_match(parts, 3, ["ff", "net", "0", "proj", "weight"]) ||
    parts_match(parts, 3, ["ff", "net", "2", "weight"]) {
    return Some(DenoiserFeedForward)
  }
  None
}

///|
fn classify_denoiser(name : String) -> MiniMaxH3TensorRole? {
  match denoiser_fixed(name) {
    Some(role) => Some(role)
    None => {
      let parts = name.split(".").to_array()
      match denoiser_block(parts) {
        Some(role) => Some(role)
        None => refiner_block(parts)
      }
    }
  }
}

///|
fn is_text_identifier(value : StringView) -> Bool {
  match value {
    "lm_head"
    | "model"
    | "language_model"
    | "embed_tokens"
    | "norm"
    | "layers"
    | "input_layernorm"
    | "post_attention_layernorm"
    | "self_attn"
    | "q_proj"
    | "k_proj"
    | "v_proj"
    | "o_proj"
    | "q_norm"
    | "k_norm"
    | "mlp"
    | "gate_proj"
    | "up_proj"
    | "down_proj"
    | "visual"
    | "patch_embed"
    | "proj"
    | "pos_embed"
    | "blocks"
    | "norm1"
    | "norm2"
    | "attn"
    | "qkv"
    | "linear_fc1"
    | "linear_fc2"
    | "merger"
    | "deepstack_merger_list"
    | "weight"
    | "bias" => true
    _ => bounded_index(value, 64)
  }
}

///|
fn valid_text_path(name : String) -> Bool {
  let parts = name.split(".").to_array()
  if parts.length() < 2 ||
    !(name == "lm_head.weight" ||
    name.has_prefix("model.language_model.") ||
    name.has_prefix("model.visual.")) {
    return false
  }
  for part in parts {
    if !is_text_identifier(part) {
      return false
    }
  }
  parts[parts.length() - 1] == "weight" || parts[parts.length() - 1] == "bias"
}

///|
fn classify_text(name : String) -> MiniMaxH3TensorRole? {
  if !expected_text_names().contains(name) || !valid_text_path(name) {
    return None
  }
  if name == "lm_head.weight" || name.contains("embed_tokens") {
    Some(TextEmbedding)
  } else if name.has_prefix("model.language_model.") {
    if name.contains("self_attn") {
      Some(TextAttention)
    } else if name.contains("mlp") {
      Some(TextFeedForward)
    } else {
      Some(TextNormalization)
    }
  } else if name.contains("patch_embed") || name.contains("pos_embed") {
    Some(VisionPatchEmbedding)
  } else if name.contains("attn") {
    Some(VisionAttention)
  } else if name.contains("mlp") || name.contains("linear_fc") {
    Some(VisionFeedForward)
  } else {
    Some(VisionNormalization)
  }
}

///|
fn classify_video_vae(name : String) -> MiniMaxH3TensorRole? {
  video_vae_role(name)
}

///|
fn classify_audio_vae(name : String) -> MiniMaxH3TensorRole? {
  audio_vae_role(name)
}

///|
fn classify_tensor(
  component : MiniMaxH3WeightComponent,
  name : String,
) -> MiniMaxH3TensorRole? {
  match component {
    Fl2vaDenoiser | Ref2vaDenoiser => classify_denoiser(name)
    TextConditioner => classify_text(name)
    VideoVae => classify_video_vae(name)
    AudioVae => classify_audio_vae(name)
  }
}
