///|
/// Interpret original wrapper/source JSON as model data. The known auto_map is
/// never imported; original architecture defaults belong to this model adapter.
pub fn parse_video_vae_checkpoint(
  wrapper : StringView,
  source_config : StringView,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> MiniMaxH3VideoVaeConfig raise MiniMaxH3ConfigError {
  let original = bounded_object(wrapper, max_bytes, max_depth)
  if string_field(original, "_class_name") != "MiniMaxH3VideoVAE" {
    raise WrongDocument(VideoVae)
  }
  let object : Map[String, Json] = Map([])
  for name, value in original {
    if name == "auto_map" {
      guard value is Object(mapping) &&
        mapping.length() == 1 &&
        mapping.get("AutoModel")
        is Some(String("minimax_h3_video_vae.MiniMaxH3VideoVAE")) else {
        raise UnsupportedExecutableMetadata
      }
    } else {
      object[name] = value
    }
  }
  validate_known(object, [
    "_class_name", "_diffusers_version", "mode", "source_path", "source_class_name",
    "source_safetensors_path", "vae_clip_length", "vae_token_drop", "vae_encoder_tiling",
    "vae_decoder_tiling", "vae_parallel_tiling", "vae_tile_size", "vae_tile_overlap_min",
    "vae_encoder_parallel", "vae_decoder_parallel", "vae_chunk_dim", "latent_channels",
    "latents_mean", "latents_std",
  ])
  guard string_field(object, "mode") == "standalone" &&
    string_field(object, "source_path") == "source" &&
    string_field(object, "source_class_name") == "AutoencoderKLLegacy" &&
    string_field(object, "source_safetensors_path") == "model.safetensors" &&
    int_field(object, "latent_channels") == 24 &&
    int_field(object, "vae_clip_length") == 17 &&
    int_field(object, "vae_token_drop") == 3 &&
    int_field(object, "vae_tile_size") == 256 &&
    int_field(object, "vae_tile_overlap_min") == 64 &&
    int_field(object, "vae_chunk_dim") == -1 else {
    raise InvalidField
  }
  for
    name in ["vae_encoder_tiling", "vae_decoder_tiling", "vae_parallel_tiling"] {
    guard int_field(object, name) == 1 else { raise InvalidField }
  }
  for name in ["vae_encoder_parallel", "vae_decoder_parallel"] {
    guard int_field(object, name) == 0 else { raise InvalidField }
  }
  validate_official_video_vae_scaling(object)
  let source = bounded_object(source_config, max_bytes, max_depth)
  validate_known(source, [
    "_class_name", "_diffusers_version", "causal_decoder", "causal_encoder", "ch",
    "ch_mult", "embed_dim", "in_channels", "num_res_blocks", "num_res_blocks_decoder",
    "out_ch", "padding_mode", "padding_mode_t", "pixel_norm_type", "scaling_factor",
    "shift_factor", "space_down", "space_up", "time_down", "time_up", "use_3d_conv",
    "use_t_isolated_gn", "use_vit_decoder", "vae_ratio", "vae_ratio_t", "vit_decoder_kwargs",
    "z_channels", "zq_ch_decoder", "zq_ch_encoder",
  ])
  guard string_field(source, "_class_name") == "AutoencoderKLLegacy" &&
    required(source, "causal_decoder") is False &&
    required(source, "causal_encoder") is True &&
    required(source, "use_3d_conv") is True &&
    required(source, "use_t_isolated_gn") is True &&
    required(source, "use_vit_decoder") is True &&
    int_field(source, "ch") == 128 &&
    int_field(source, "embed_dim") == 24 &&
    int_field(source, "z_channels") == 24 &&
    int_field(source, "in_channels") == 3 &&
    int_field(source, "out_ch") == 3 &&
    int_field(source, "num_res_blocks") == 2 &&
    int_field(source, "vae_ratio") == 16 &&
    int_field(source, "vae_ratio_t") == 4 &&
    string_field(source, "padding_mode") == "reflect" &&
    string_field(source, "pixel_norm_type") == "imagenet" &&
    number_field(source, "scaling_factor") == 1.0 &&
    number_field(source, "shift_factor") == 0.0 else {
    raise InvalidField
  }
  for
    name in [
      "num_res_blocks_decoder", "padding_mode_t", "time_up", "zq_ch_decoder", "zq_ch_encoder",
    ] {
    guard required(source, name) is Null else { raise InvalidField }
  }
  exact_int_array(source, "ch_mult", [1, 2, 2, 4, 4, 8])
  exact_int_array(source, "space_down", [2, 2, 2, 2, 1, 1])
  exact_int_array(source, "space_up", [1, 2, 2, 2, 2, 1])
  exact_int_array(source, "time_down", [1, 2, 2, 1, 1, 1])
  let decoder = object_field(source, "vit_decoder_kwargs")
  validate_known(decoder, [
    "dim_head", "ffn_activation_fn", "ffn_use_gated", "heads", "norm_affine", "norm_type",
    "num_layers", "qk_norm_affine", "qk_norm_type", "rope_dim_ratio", "rope_theta",
  ])
  guard int_field(decoder, "dim_head") == 64 &&
    int_field(decoder, "heads") == 32 &&
    int_field(decoder, "num_layers") == 36 &&
    string_field(decoder, "ffn_activation_fn") == "silu" &&
    string_field(decoder, "norm_type") == "rms_norm" &&
    string_field(decoder, "qk_norm_type") == "rms_norm" &&
    required(decoder, "ffn_use_gated") is True &&
    required(decoder, "norm_affine") is True &&
    required(decoder, "qk_norm_affine") is False &&
    number_field(decoder, "rope_dim_ratio") == 0.75 &&
    number_field(decoder, "rope_theta") == 100.0 else {
    raise InvalidField
  }
  official_video_vae_config()
}
