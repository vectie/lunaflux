///|
fn validate_limits(
  max_bytes : Int,
  max_depth : Int,
) -> Unit raise MiniMaxH3ConfigError {
  if max_bytes <= 0 || max_depth <= 0 {
    raise InvalidLimit
  }
}

///|
fn validate_utf8_size(
  input : StringView,
  max_bytes : Int,
) -> Unit raise MiniMaxH3ConfigError {
  let mut index = 0
  let mut bytes = 0
  while index < input.length() {
    let unit = input[index]
    let (width, next) = if unit <= 0x7f {
      (1, index + 1)
    } else if unit <= 0x7ff {
      (2, index + 1)
    } else if unit.is_leading_surrogate() {
      if index + 1 >= input.length() ||
        !input[index + 1].is_trailing_surrogate() {
        raise InvalidUtf8
      }
      (4, index + 2)
    } else if unit.is_trailing_surrogate() {
      raise InvalidUtf8
    } else {
      (3, index + 1)
    }
    if bytes > max_bytes - width {
      raise ByteLimitExceeded
    }
    bytes += width
    index = next
  }
}

///|
fn parse_object(
  document : StringView,
  max_depth : Int,
) -> Map[String, Json] raise MiniMaxH3ConfigError {
  @json_guard.reject_duplicate_object_keys(document, max_depth~) catch {
    InvalidDepthLimit => raise InvalidLimit
    DepthLimitExceeded(_) => raise NestingLimitExceeded
    DuplicateObjectKey(_) => raise DuplicateField
    MalformedStructure => raise MalformedJson
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedJson }
  guard parsed is Object(root) else { raise NonObjectRoot }
  root
}

///|
fn required(
  object : Map[String, Json],
  name : String,
) -> Json raise MiniMaxH3ConfigError {
  match object.get(name) {
    Some(value) => value
    None => raise MissingField
  }
}

///|
fn string_field(
  object : Map[String, Json],
  name : String,
) -> String raise MiniMaxH3ConfigError {
  match required(object, name) {
    String(value) => value
    _ => raise InvalidField
  }
}

///|
fn int_field(
  object : Map[String, Json],
  name : String,
) -> Int raise MiniMaxH3ConfigError {
  match required(object, name) {
    Number(value, repr=_) if !value.is_nan() &&
      !value.is_inf() &&
      value == value.trunc() &&
      value >= -2147483648.0 &&
      value <= 2147483647.0 => value.to_int()
    _ => raise InvalidField
  }
}

///|
fn number_field(
  object : Map[String, Json],
  name : String,
) -> Double raise MiniMaxH3ConfigError {
  match required(object, name) {
    Number(value, repr=_) if !value.is_nan() && !value.is_inf() => value
    _ => raise InvalidField
  }
}

///|
fn object_field(
  object : Map[String, Json],
  name : String,
) -> Map[String, Json] raise MiniMaxH3ConfigError {
  match required(object, name) {
    Object(value) => value
    _ => raise InvalidField
  }
}

///|
fn exact_int_array(
  object : Map[String, Json],
  name : String,
  expected : ArrayView[Int],
) -> Unit raise MiniMaxH3ConfigError {
  guard required(object, name) is Array(values) else { raise InvalidField }
  if values.length() != expected.length() {
    raise InvalidField
  }
  for index, expected_value in expected {
    match values[index] {
      Number(value, repr=_) if value == expected_value.to_double() => ()
      _ => raise InvalidField
    }
  }
}

///|
fn exact_nested_int_array(
  object : Map[String, Json],
  name : String,
  expected : ArrayView[Array[Int]],
) -> Unit raise MiniMaxH3ConfigError {
  guard required(object, name) is Array(rows) else { raise InvalidField }
  if rows.length() != expected.length() {
    raise InvalidField
  }
  for row_index, expected_row in expected {
    guard rows[row_index] is Array(values) else { raise InvalidField }
    if values.length() != expected_row.length() {
      raise InvalidField
    }
    for value_index, expected_value in expected_row {
      match values[value_index] {
        Number(value, repr=_) if value == expected_value.to_double() => ()
        _ => raise InvalidField
      }
    }
  }
}

///|
fn component_pair(
  object : Map[String, Json],
  name : String,
  owner : String,
  class_name : String,
) -> Unit raise MiniMaxH3ConfigError {
  guard required(object, name) is Array(values) else { raise InvalidField }
  match values {
    [String(actual_owner), String(actual_class)] if actual_owner == owner &&
      actual_class == class_name => ()
    _ => raise InvalidField
  }
}

///|
fn validate_known(
  object : Map[String, Json],
  allowed : ArrayView[String],
) -> Unit raise MiniMaxH3ConfigError {
  let mut index = 0
  for name, _ in object {
    if name == "auto_map" ||
      name == "trust_remote_code" ||
      name == "custom_pipeline" {
      raise UnsupportedExecutableMetadata
    }
    if !allowed.any(value => value == name) {
      raise UnknownField(index~)
    }
    index += 1
  }
}

///|
fn decode_pipeline(
  object : Map[String, Json],
) -> MiniMaxH3PipelineIndex raise MiniMaxH3ConfigError {
  validate_known(object, [
    "_class_name", "_diffusers_version", "text_encoder", "tokenizer", "video_vae",
    "audio_vae", "scheduler", "transformer", "processor", "_minimax_h3",
  ])
  if string_field(object, "_class_name") != "MiniMaxH3Pipeline" {
    raise WrongDocument(PipelineIndex)
  }
  component_pair(
    object, "text_encoder", "transformers", "MiniMaxH3Qwen3VLHFEncoder",
  )
  component_pair(object, "tokenizer", "transformers", "Qwen2TokenizerFast")
  component_pair(object, "video_vae", "diffusers", "MiniMaxH3VideoVAE")
  component_pair(object, "audio_vae", "diffusers", "MiniMaxH3AudioVAE")
  component_pair(object, "transformer", "diffusers", "MiniMaxH3DiTModel")
  component_pair(object, "processor", "transformers", "Qwen3VLProcessor")
  match required(object, "scheduler") {
    Null => ()
    _ => raise InvalidField
  }
  let family = object_field(object, "_minimax_h3")
  validate_known(family, [
    "schema_version", "partition", "tasks", "task_aliases", "sigma_shift_scales",
  ])
  let schema_version = int_field(family, "schema_version")
  if schema_version != 1 {
    raise InvalidField
  }
  let variant = match string_field(family, "partition") {
    "fl2va" => @minimax_h3_spec.MiniMaxH3Variant::Fl2va
    "ref2va" => Ref2va
    _ => raise InvalidField
  }
  guard required(family, "tasks") is Array(tasks) else { raise InvalidField }
  match (variant, tasks) {
    (Fl2va, [String("t2va"), String("fl2va")]) => ()
    (Ref2va, [String("ref2va")]) => ()
    _ => raise InvalidField
  }
  guard required(family, "task_aliases") is Object(aliases) &&
    aliases.is_empty() else {
    raise InvalidField
  }
  let shifts = object_field(family, "sigma_shift_scales")
  validate_known(shifts, ["video", "audio"])
  let video_shift = number_field(shifts, "video")
  let audio_shift = number_field(shifts, "audio")
  if video_shift != 12.0 || audio_shift != 3.0 {
    raise InvalidField
  }
  { variant, schema_version, video_shift, audio_shift, }
}

///|
fn decode_transformer(
  object : Map[String, Json],
  variant : @minimax_h3_spec.MiniMaxH3Variant,
  content : @spec.ContentDigest,
) -> @minimax_h3_spec.MiniMaxH3ModelMetadata raise MiniMaxH3ConfigError {
  let architecture = string_field(object, "_class_name")
  let spec = if architecture == "MiniMaxH3Transformer3DModel" {
    validate_known(object, [
      "_class_name", "_diffusers_version", "num_attention_heads", "attention_head_dim",
      "hidden_size", "num_layers", "num_refiner_layers", "ffn_dim", "in_channels",
      "audio_in_channels", "patch_size", "text_dim", "freq_dim", "time_embed_hidden_dim",
      "time_embed_dim", "rope_freq_dim", "rope_theta", "norm_eps", "qk_norm_eps",
      "final_norm_eps",
    ])
    exact_int_array(object, "patch_size", [1, 2, 2])
    @minimax_h3_spec.MiniMaxH3TransformerSpec::new_exact(
      variant~,
      checkpoint_format=DiffusersV1,
      schema_version=1,
      hidden_size=int_field(object, "hidden_size"),
      layer_count=int_field(object, "num_layers"),
      refiner_layer_count=int_field(object, "num_refiner_layers"),
      attention_head_count=int_field(object, "num_attention_heads"),
      attention_head_dimension=int_field(object, "attention_head_dim"),
      ffn_dimension=int_field(object, "ffn_dim"),
      visual_latent_channels=int_field(object, "in_channels"),
      audio_latent_channels=int_field(object, "audio_in_channels"),
      patch_temporal=1,
      patch_height=2,
      patch_width=2,
      text_dimension=int_field(object, "text_dim"),
      frequency_dimension=int_field(object, "freq_dim"),
      time_embedding_hidden_dimension=int_field(object, "time_embed_hidden_dim"),
      time_embedding_dimension=int_field(object, "time_embed_dim"),
      adaln_output_features=96768,
      final_adaln_output_features=10752,
      rope_frequency_dimension=int_field(object, "rope_freq_dim"),
      rope_theta=number_field(object, "rope_theta"),
      norm_epsilon=number_field(object, "norm_eps"),
      query_key_norm_epsilon=number_field(object, "qk_norm_eps"),
      final_norm_epsilon=number_field(object, "final_norm_eps"),
      weight_dtype=BFloat16,
    ) catch {
      _ => raise InvalidField
    }
  } else if architecture == "MiniMaxH3DiTModel" {
    validate_known(object, [
      "_class_name", "_diffusers_version", "hidden_size", "num_layers", "token_refiner_num_layers",
      "num_attention_heads", "attention_head_dim", "ffn_hidden_size", "latents_dim",
      "audio_latents_dim", "patch_size", "text_dim", "timestep_input_dim", "time_embed_hidden_size",
      "time_embed_dim", "adaln_out_features", "final_adaln_out_features", "rope_inv_freq_len",
      "eps", "norm_eps", "qk_norm_eps", "final_norm_eps",
    ])
    exact_int_array(object, "patch_size", [1, 2, 2])
    let (norm_epsilon, query_key_norm_epsilon, final_norm_epsilon) = original_transformer_epsilons(
      object,
    )
    @minimax_h3_spec.MiniMaxH3TransformerSpec::new_exact(
      variant~,
      checkpoint_format=OriginalV1,
      schema_version=1,
      hidden_size=int_field(object, "hidden_size"),
      layer_count=int_field(object, "num_layers"),
      refiner_layer_count=int_field(object, "token_refiner_num_layers"),
      attention_head_count=int_field(object, "num_attention_heads"),
      attention_head_dimension=int_field(object, "attention_head_dim"),
      ffn_dimension=int_field(object, "ffn_hidden_size"),
      visual_latent_channels=int_field(object, "latents_dim"),
      audio_latent_channels=int_field(object, "audio_latents_dim"),
      patch_temporal=1,
      patch_height=2,
      patch_width=2,
      text_dimension=int_field(object, "text_dim"),
      frequency_dimension=int_field(object, "timestep_input_dim"),
      time_embedding_hidden_dimension=int_field(
        object, "time_embed_hidden_size",
      ),
      time_embedding_dimension=int_field(object, "time_embed_dim"),
      adaln_output_features=int_field(object, "adaln_out_features"),
      final_adaln_output_features=int_field(object, "final_adaln_out_features"),
      rope_frequency_dimension=int_field(object, "rope_inv_freq_len"),
      rope_theta=10000.0,
      norm_epsilon~,
      query_key_norm_epsilon~,
      final_norm_epsilon~,
      weight_dtype=BFloat16,
    ) catch {
      _ => raise InvalidField
    }
  } else {
    raise WrongDocument(Transformer)
  }
  @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(spec, content)
}

///|
fn decode_video_vae(
  object : Map[String, Json],
) -> MiniMaxH3VideoVaeConfig raise MiniMaxH3ConfigError {
  validate_known(object, [
    "_class_name", "_diffusers_version", "in_channels", "out_channels", "latent_channels",
    "block_out_channels", "layers_per_block", "spatial_downsample_factors", "temporal_downsample_factors",
    "norm_num_groups", "norm_eps", "spatial_padding_mode", "decoder_num_layers",
    "decoder_num_attention_heads", "decoder_attention_head_dim", "decoder_num_register_tokens",
    "decoder_ffn_mult", "decoder_rope_theta", "decoder_rope_dim_ratio", "decoder_norm_eps",
    "clip_length", "token_drop", "latents_mean", "latents_std",
  ])
  if string_field(object, "_class_name") != "AutoencoderKLMiniMaxH3" {
    raise WrongDocument(VideoVae)
  }
  exact_int_array(object, "block_out_channels", [128, 256, 256, 512, 512, 1024])
  exact_int_array(object, "spatial_downsample_factors", [2, 2, 2, 2, 1, 1])
  exact_int_array(object, "temporal_downsample_factors", [1, 2, 2, 1, 1, 1])
  validate_official_video_vae_scaling(object)
  if int_field(object, "in_channels") != 3 ||
    int_field(object, "out_channels") != 3 ||
    int_field(object, "latent_channels") != 24 ||
    int_field(object, "clip_length") != 17 ||
    int_field(object, "token_drop") != 3 ||
    int_field(object, "layers_per_block") != 2 ||
    int_field(object, "norm_num_groups") != 32 ||
    number_field(object, "norm_eps") != 0.000001 ||
    string_field(object, "spatial_padding_mode") != "reflect" ||
    int_field(object, "decoder_num_layers") != 36 ||
    int_field(object, "decoder_num_attention_heads") != 32 ||
    int_field(object, "decoder_attention_head_dim") != 64 ||
    int_field(object, "decoder_num_register_tokens") != 4 ||
    int_field(object, "decoder_ffn_mult") != 4 ||
    number_field(object, "decoder_rope_theta") != 100.0 ||
    number_field(object, "decoder_rope_dim_ratio") != 0.75 ||
    number_field(object, "decoder_norm_eps") != 0.00001 {
    raise InvalidField
  }
  {
    architecture_schema_version: 1,
    input_channels: 3,
    output_channels: 3,
    latent_channels: 24,
    block_out_channels: ReadOnlyArray::from_array([
      128, 256, 256, 512, 512, 1024,
    ]),
    layers_per_block: 2,
    spatial_downsample_factors: ReadOnlyArray::from_array([2, 2, 2, 2, 1, 1]),
    temporal_downsample_factors: ReadOnlyArray::from_array([1, 2, 2, 1, 1, 1]),
    norm_group_count: 32,
    norm_epsilon: 0.000001,
    clip_length: 17,
    token_drop: 3,
    spatial_compression: 16,
    temporal_compression: 4,
    decoder_layer_count: 36,
    decoder_attention_head_count: 32,
    decoder_attention_head_dimension: 64,
    decoder_register_token_count: 4,
    decoder_ffn_multiplier: 4,
    decoder_rope_theta: 100.0,
    decoder_rope_dimension_ratio: 0.75,
    decoder_norm_epsilon: 0.00001,
    latent_means: official_video_vae_f32(official_video_vae_latent_means_f64()),
    latent_standard_deviations: official_video_vae_f32(
      official_video_vae_latent_standard_deviations_f64(),
    ),
  }
}

///|
fn decode_audio_vae(
  object : Map[String, Json],
) -> MiniMaxH3AudioVaeConfig raise MiniMaxH3ConfigError {
  validate_known(object, [
    "_class_name", "_diffusers_version", "encoder_dim", "encoder_rates", "latent_dim",
    "latent_channels", "decoder_dim", "decoder_rates", "decoder_kernel_sizes", "num_attention_heads",
    "resblock_kernel_sizes", "resblock_dilation_sizes", "sampling_rate", "latents_mean",
    "latents_std",
  ])
  if string_field(object, "_class_name") != "AutoencoderKLMiniMaxH3Audio" {
    raise WrongDocument(AudioVae)
  }
  exact_int_array(object, "encoder_rates", [2, 4, 4, 5, 5])
  exact_int_array(object, "decoder_rates", [5, 5, 2, 2, 2, 2, 2])
  exact_int_array(object, "decoder_kernel_sizes", [9, 9, 4, 4, 4, 4, 4])
  exact_int_array(object, "resblock_kernel_sizes", [3, 7, 11])
  exact_nested_int_array(object, "resblock_dilation_sizes", [
    [1, 3, 5],
    [1, 3, 5],
    [1, 3, 5],
  ])
  validate_official_audio_vae_scaling(object)
  if int_field(object, "encoder_dim") != 64 ||
    int_field(object, "latent_channels") != 32 ||
    int_field(object, "sampling_rate") != 32000 ||
    int_field(object, "latent_dim") != 2048 ||
    int_field(object, "decoder_dim") != 1024 ||
    int_field(object, "num_attention_heads") != 8 {
    raise InvalidField
  }
  official_audio_vae_config()
}

///|
fn decode_scheduler(
  object : Map[String, Json],
  document : MiniMaxH3ConfigDocument,
) -> MiniMaxH3SchedulerConfig raise MiniMaxH3ConfigError {
  validate_known(object, ["_class_name", "_diffusers_version", "shift"])
  if string_field(object, "_class_name") != "MiniMaxH3Scheduler" {
    raise WrongDocument(document)
  }
  let shift = number_field(object, "shift")
  let expected = match document {
    VideoScheduler => 12.0
    AudioScheduler => 3.0
    _ => raise WrongDocument(document)
  }
  if shift != expected {
    raise InvalidField
  }
  { document, shift, }
}

///|
fn bounded_object(
  document : StringView,
  max_bytes : Int,
  max_depth : Int,
) -> Map[String, Json] raise MiniMaxH3ConfigError {
  validate_limits(max_bytes, max_depth)
  validate_utf8_size(document, max_bytes)
  parse_object(document, max_depth)
}
