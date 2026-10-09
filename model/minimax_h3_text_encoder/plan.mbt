///|
pub(all) suberror TextEncoderError {
  InvalidRows
  InvalidEpsilon
  InvalidLayer
  InvalidToken
  VisualTokensRequireEncoder
} derive(Eq)

///|
/// The H3 extractor stops after layer 49, before the language-model final norm.
/// Epsilon comes from the admitted Qwen3-VL text configuration, not DiT config.
pub struct TextEncoderPlan {
  rows : Int
  epsilon : Double
}

///|
pub fn TextEncoderPlan::new(
  rows~ : Int,
  epsilon~ : Double,
) -> Self raise TextEncoderError {
  guard rows > 0 && rows <= 262144 else { raise InvalidRows }
  guard epsilon > 0.0 && epsilon <= 1.0 && !epsilon.is_nan() else {
    raise InvalidEpsilon
  }
  { rows, epsilon, }
}

///|
pub fn TextEncoderPlan::selected_hidden_state(_self : Self) -> Int {
  50
}

///|
pub fn TextEncoderPlan::output_bytes(self : Self) -> Int64 {
  self.rows.to_int64() * 5120L * 2L
}

///|
/// Tokenization/chat-template handling happens upstream. Reject image/video
/// placeholders rather than silently running them without visual embeddings.
pub fn TextEncoderPlan::validate_text_tokens(
  self : Self,
  ids : ArrayView[Int],
  image_token~ : Int,
  video_token~ : Int,
) -> Unit raise TextEncoderError {
  guard ids.length() == self.rows else { raise InvalidRows }
  for id in ids {
    guard id >= 0 && id < 151936 else { raise InvalidToken }
    guard id != image_token && id != video_token else {
      raise VisualTokensRequireEncoder
    }
  }
}

///|
pub struct WeightRegion {
  name : String
  offset : Int64
  bytes : Int64
}

///|
pub struct LayerWeights {
  layer : Int
  regions : ReadOnlyArray[WeightRegion]
  byte_count : Int64
}

///|
pub fn embedding_region() -> WeightRegion {
  {
    name: "model.language_model.embed_tokens.weight",
    offset: 0L,
    bytes: 151936L * 5120L * 2L,
  }
}

///|
/// Regions: input norm, Q, K, V, Q norm, K norm, O, post norm,
/// up, gate, down. Up/gate are adjacent in the shared staged-MLP ABI order.
pub fn layer_weights(layer : Int) -> LayerWeights raise TextEncoderError {
  guard layer >= 0 && layer < 50 else { raise InvalidLayer }
  let prefix = "model.language_model.layers.\{layer}."
  let shapes = [
    ("input_layernorm.weight", 5120L),
    ("self_attn.q_proj.weight", 8192L * 5120L),
    ("self_attn.k_proj.weight", 1024L * 5120L),
    ("self_attn.v_proj.weight", 1024L * 5120L),
    ("self_attn.q_norm.weight", 128L),
    ("self_attn.k_norm.weight", 128L),
    ("self_attn.o_proj.weight", 5120L * 8192L),
    ("post_attention_layernorm.weight", 5120L),
    ("mlp.up_proj.weight", 25600L * 5120L),
    ("mlp.gate_proj.weight", 25600L * 5120L),
    ("mlp.down_proj.weight", 5120L * 25600L),
  ]
  let regions : Array[WeightRegion] = []
  let mut offset = 0L
  for (name, elements) in shapes {
    let bytes = elements * 2L
    regions.push({ name: prefix + name, offset, bytes, })
    offset += bytes
  }
  { layer, regions: ReadOnlyArray::from_array(regions), byte_count: offset, }
}

///|
/// Checked startup workspace, shared across all 50 layers; no KV decode cache.
pub fn TextEncoderPlan::workspace_regions(
  self : Self,
) -> ReadOnlyArray[(Int64, Int64)] {
  let widths = [5120, 5120, 8192, 1024, 1024, 8192, 5120, 25600]
  let out : Array[(Int64, Int64)] = []
  let mut offset = 0L
  for width in widths {
    let bytes = self.rows.to_int64() * width.to_int64() * 2L
    out.push((offset, bytes))
    offset += bytes
  }
  ReadOnlyArray::from_array(out)
}
