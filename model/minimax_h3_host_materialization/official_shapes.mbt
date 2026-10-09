///|
fn shape_is(actual : ReadOnlyArray[Int64], expected : ArrayView[Int64]) -> Bool {
  actual == ReadOnlyArray::from_array(expected)
}

///|
fn index_in(value : StringView, upper : Int) -> Bool {
  let parsed = @string.parse_int(value, base=10) catch { _ => return false }
  parsed >= 0 && parsed < upper
}

///|
fn suffix_is(
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
fn language_shape(name : String, shape : ReadOnlyArray[Int64]) -> Bool {
  match name {
    "lm_head.weight" | "model.language_model.embed_tokens.weight" =>
      shape_is(shape, [151936L, 5120L])
    "model.language_model.norm.weight" => shape_is(shape, [5120L])
    _ => {
      let parts = name.split(".").to_array()
      if parts.length() < 6 ||
        parts[0] != "model" ||
        parts[1] != "language_model" ||
        parts[2] != "layers" ||
        !index_in(parts[3], 64) {
        return false
      }
      if suffix_is(parts, 4, ["input_layernorm", "weight"]) ||
        suffix_is(parts, 4, ["post_attention_layernorm", "weight"]) {
        shape_is(shape, [5120L])
      } else if suffix_is(parts, 4, ["self_attn", "q_proj", "weight"]) {
        shape_is(shape, [8192L, 5120L])
      } else if suffix_is(parts, 4, ["self_attn", "k_proj", "weight"]) ||
        suffix_is(parts, 4, ["self_attn", "v_proj", "weight"]) {
        shape_is(shape, [1024L, 5120L])
      } else if suffix_is(parts, 4, ["self_attn", "o_proj", "weight"]) {
        shape_is(shape, [5120L, 8192L])
      } else if suffix_is(parts, 4, ["self_attn", "q_norm", "weight"]) ||
        suffix_is(parts, 4, ["self_attn", "k_norm", "weight"]) {
        shape_is(shape, [128L])
      } else if suffix_is(parts, 4, ["mlp", "gate_proj", "weight"]) ||
        suffix_is(parts, 4, ["mlp", "up_proj", "weight"]) {
        shape_is(shape, [25600L, 5120L])
      } else if suffix_is(parts, 4, ["mlp", "down_proj", "weight"]) {
        shape_is(shape, [5120L, 25600L])
      } else {
        false
      }
    }
  }
}

///|
fn merger_shape(
  parts : Array[StringView],
  offset : Int,
  shape : ReadOnlyArray[Int64],
) -> Bool {
  if suffix_is(parts, offset, ["norm", "weight"]) ||
    suffix_is(parts, offset, ["norm", "bias"]) {
    shape_is(shape, [1152L])
  } else if suffix_is(parts, offset, ["linear_fc1", "weight"]) {
    shape_is(shape, [4608L, 4608L])
  } else if suffix_is(parts, offset, ["linear_fc1", "bias"]) {
    shape_is(shape, [4608L])
  } else if suffix_is(parts, offset, ["linear_fc2", "weight"]) {
    shape_is(shape, [5120L, 4608L])
  } else if suffix_is(parts, offset, ["linear_fc2", "bias"]) {
    shape_is(shape, [5120L])
  } else {
    false
  }
}

///|
fn vision_shape(name : String, shape : ReadOnlyArray[Int64]) -> Bool {
  match name {
    "model.visual.patch_embed.proj.weight" =>
      shape_is(shape, [1152L, 3L, 2L, 16L, 16L])
    "model.visual.patch_embed.proj.bias" => shape_is(shape, [1152L])
    "model.visual.pos_embed.weight" => shape_is(shape, [2304L, 1152L])
    _ => {
      let parts = name.split(".").to_array()
      if parts.length() >= 6 &&
        parts[0] == "model" &&
        parts[1] == "visual" &&
        parts[2] == "blocks" &&
        index_in(parts[3], 27) {
        if suffix_is(parts, 4, ["norm1", "weight"]) ||
          suffix_is(parts, 4, ["norm1", "bias"]) ||
          suffix_is(parts, 4, ["norm2", "weight"]) ||
          suffix_is(parts, 4, ["norm2", "bias"]) {
          shape_is(shape, [1152L])
        } else if suffix_is(parts, 4, ["attn", "qkv", "weight"]) {
          shape_is(shape, [3456L, 1152L])
        } else if suffix_is(parts, 4, ["attn", "qkv", "bias"]) {
          shape_is(shape, [3456L])
        } else if suffix_is(parts, 4, ["attn", "proj", "weight"]) {
          shape_is(shape, [1152L, 1152L])
        } else if suffix_is(parts, 4, ["attn", "proj", "bias"]) {
          shape_is(shape, [1152L])
        } else if suffix_is(parts, 4, ["mlp", "linear_fc1", "weight"]) {
          shape_is(shape, [4304L, 1152L])
        } else if suffix_is(parts, 4, ["mlp", "linear_fc1", "bias"]) {
          shape_is(shape, [4304L])
        } else if suffix_is(parts, 4, ["mlp", "linear_fc2", "weight"]) {
          shape_is(shape, [1152L, 4304L])
        } else if suffix_is(parts, 4, ["mlp", "linear_fc2", "bias"]) {
          shape_is(shape, [1152L])
        } else {
          false
        }
      } else if parts.length() >= 5 &&
        parts[0] == "model" &&
        parts[1] == "visual" &&
        parts[2] == "merger" {
        merger_shape(parts, 3, shape)
      } else if parts.length() >= 6 &&
        parts[0] == "model" &&
        parts[1] == "visual" &&
        parts[2] == "deepstack_merger_list" &&
        index_in(parts[3], 3) {
        merger_shape(parts, 4, shape)
      } else {
        false
      }
    }
  }
}

///|
fn exact_official_shape(
  component : @minimax_h3_weights.MiniMaxH3WeightComponent,
  name : String,
  shape : ReadOnlyArray[Int64],
) -> Bool {
  match component {
    Fl2vaDenoiser | Ref2vaDenoiser => true
    TextConditioner => language_shape(name, shape) || vision_shape(name, shape)
    VideoVae =>
      @minimax_h3_weights.MiniMaxH3VideoVaeSchema::official_v1().matches_tensor(
        name,
        Float32,
        shape,
      )
    AudioVae =>
      @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1().matches_tensor(
        name,
        Float32,
        shape,
      )
  }
}
