///|
pub(all) suberror EncoderConfigError {
  InvalidDocument
  InvalidArchitecture
  UnsupportedRotary
  InvalidRequest
  ByteBudgetExceeded
} derive(Eq)

///|
/// Values are read from the admitted encoder config, never from DiT defaults.
pub struct EncoderConfig {
  epsilon : Double
  theta : Double
  max_positions : Int
  vocabulary_size : Int
  image_token : Int
  video_token : Int
  vision_start_token : Int
  vision_end_token : Int
  mrope_height : Int
  mrope_width : Int
}

///|
fn required(
  o : Map[String, Json],
  key : String,
) -> Json raise EncoderConfigError {
  match o.get(key) {
    Some(v) => v
    None => raise InvalidArchitecture
  }
}

///|
fn object(v : Json) -> Map[String, Json] raise EncoderConfigError {
  guard v is Object(o) else { raise InvalidArchitecture }
  o
}

///|
fn number(
  o : Map[String, Json],
  key : String,
) -> Double raise EncoderConfigError {
  guard required(o, key) is Number(v, ..) && !v.is_nan() && !v.is_inf() else {
    raise InvalidArchitecture
  }
  v
}

///|
fn integer(o : Map[String, Json], key : String) -> Int raise EncoderConfigError {
  let v = number(o, key)
  guard v >= 0.0 && v <= 2147483647.0 && v == v.trunc() else {
    raise InvalidArchitecture
  }
  v.to_int()
}

///|
/// Parse checkpoint Qwen3-VL-32B config. Only the default full-head rotary
/// contract is supported; dynamic/YaRN scaling is never silently discarded.
/// For text-only input all three mRoPE axes have identical positions.
pub fn parse_config(
  bytes : Bytes,
  max_bytes~ : Int,
) -> EncoderConfig raise EncoderConfigError {
  guard max_bytes > 0 && bytes.length() <= max_bytes else {
    raise ByteBudgetExceeded
  }
  let document = @utf8.decode(bytes) catch { _ => raise InvalidDocument }
  @json_guard.reject_duplicate_object_keys(document, max_depth=16) catch {
    _ => raise InvalidDocument
  }
  let root = object(@json.parse(document) catch { _ => raise InvalidDocument })
  guard required(root, "model_type") is String("qwen3_vl") else {
    raise InvalidArchitecture
  }
  let text = object(required(root, "text_config"))
  guard required(text, "model_type") is String("qwen3_vl_text") else {
    raise InvalidArchitecture
  }
  let dtype = match text.get("dtype") {
    Some(v) => v
    None => required(text, "torch_dtype")
  }
  guard dtype is String("bfloat16") else { raise InvalidArchitecture }
  for
    (name, value) in [
      ("hidden_size", 5120),
      ("intermediate_size", 25600),
      ("num_hidden_layers", 64),
      ("num_attention_heads", 64),
      ("num_key_value_heads", 8),
      ("head_dim", 128),
      ("vocab_size", 151936),
    ] {
    guard integer(text, name) == value else { raise InvalidArchitecture }
  }
  guard required(text, "hidden_act") is String("silu") &&
    required(text, "attention_bias") is False else {
    raise InvalidArchitecture
  }
  if text.get("mlp_bias") is Some(v) && !(v is False) {
    raise InvalidArchitecture
  }
  if text.get("partial_rotary_factor") is Some(v) && v != Json::number(1.0) {
    raise UnsupportedRotary
  }
  let epsilon = number(text, "rms_norm_eps")
  let max_positions = integer(text, "max_position_embeddings")
  let rotary = match (text.get("rope_parameters"), text.get("rope_scaling")) {
    (Some(v), None) => object(v)
    (None, Some(v)) => object(v)
    _ => raise UnsupportedRotary
  }
  let rope_type = match rotary.get("rope_type") {
    Some(v) => v
    None => required(rotary, "type")
  }
  guard rope_type is String("default") else { raise UnsupportedRotary }
  if rotary.get("type") is Some(v) && !(v is String("default")) {
    raise UnsupportedRotary
  }
  for key, _ in rotary {
    guard key == "rope_type" ||
      key == "type" ||
      key == "rope_theta" ||
      key == "mrope_section" ||
      key == "mrope_interleaved" else {
      raise UnsupportedRotary
    }
  }
  let theta = match rotary.get("rope_theta") {
    Some(_) => {
      guard text.get("rope_theta") is None else { raise UnsupportedRotary }
      number(rotary, "rope_theta")
    }
    None => number(text, "rope_theta")
  }
  let sections = match rotary.get("mrope_section") {
    Some(Array(a)) if a.length() == 3 => a
    None => [Json::number(24.0), Json::number(20.0), Json::number(20.0)]
    _ => raise UnsupportedRotary
  }
  let counts : Array[Int] = []
  for v in sections {
    guard v is Number(n, ..) && n >= 0.0 && n <= 64.0 && n == n.trunc() else {
      raise UnsupportedRotary
    }
    counts.push(n.to_int())
  }
  guard counts[0] + counts[1] + counts[2] == 64 &&
    counts[1] <= 21 &&
    counts[2] <= 21 else {
    raise UnsupportedRotary
  }
  if rotary.get("mrope_interleaved") is Some(v) && !(v is True) {
    raise UnsupportedRotary
  }
  let image_token = integer(root, "image_token_id")
  let video_token = integer(root, "video_token_id")
  let vision_start_token = integer(root, "vision_start_token_id")
  let vision_end_token = integer(root, "vision_end_token_id")
  guard epsilon > 0.0 &&
    epsilon <= 1.0 &&
    theta > 1.0 &&
    theta <= 3.4028234e38 &&
    max_positions > 0 &&
    image_token < 151936 &&
    video_token < 151936 &&
    image_token != video_token &&
    vision_start_token < 151936 &&
    vision_end_token < 151936 &&
    vision_start_token != vision_end_token &&
    image_token != vision_start_token &&
    image_token != vision_end_token &&
    video_token != vision_start_token &&
    video_token != vision_end_token else {
    raise InvalidArchitecture
  }
  {
    epsilon,
    theta,
    max_positions,
    vocabulary_size: integer(text, "vocab_size"),
    image_token,
    video_token,
    vision_start_token,
    vision_end_token,
    mrope_height: counts[1],
    mrope_width: counts[2],
  }
}

///|
pub fn EncoderConfig::plan(
  self : Self,
  ids : ArrayView[Int],
) -> TextEncoderPlan raise {
  guard ids.length() <= self.max_positions else { raise InvalidRequest }
  let plan = TextEncoderPlan::new(rows=ids.length(), epsilon=self.epsilon)
  plan.validate_text_tokens(
    ids,
    image_token=self.image_token,
    video_token=self.video_token,
  )
  plan
}
