///|
/// Resolve the original AudioVAE's three data-only configuration documents.
/// The known auto_map is descriptive only: no module is imported or executed.
/// `source_config` accepts only this model's flat `model_config` integer map,
/// not general YAML or executable extensions.
pub fn parse_audio_vae_checkpoint(
  wrapper : StringView,
  metadata : StringView,
  source_config : StringView,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> MiniMaxH3AudioVaeConfig raise MiniMaxH3ConfigError {
  let object = bounded_object(wrapper, max_bytes, max_depth)
  if string_field(object, "_class_name") != "MiniMaxH3AudioVAE" {
    raise WrongDocument(AudioVae)
  }
  let numeric : Map[String, Json] = Map([])
  for name, value in object {
    if name == "auto_map" {
      guard value is Object(mapping) &&
        mapping.length() == 1 &&
        mapping.get("AutoModel")
        is Some(String("minimax_h3_audio_vae.MiniMaxH3AudioVAE")) else {
        raise UnsupportedExecutableMetadata
      }
    } else {
      numeric[name] = value
    }
  }
  validate_known(numeric, [
    "_class_name", "_diffusers_version", "output_channel", "sample_rate", "source_config_path",
    "source_safetensors_path", "source_metadata_path", "latent_channels", "latents_mean",
    "latents_std",
  ])
  guard int_field(numeric, "output_channel") == 2 &&
    int_field(numeric, "sample_rate") == 32000 &&
    int_field(numeric, "latent_channels") == 32 &&
    string_field(numeric, "source_config_path") == "config.yaml" &&
    string_field(numeric, "source_safetensors_path") == "model.safetensors" &&
    string_field(numeric, "source_metadata_path") == "metadata.json" else {
    raise InvalidField
  }
  validate_official_audio_vae_scaling(numeric)
  let root = bounded_object(metadata, max_bytes, max_depth)
  validate_known(root, ["metadata"])
  let body = object_field(root, "metadata")
  validate_known(body, ["kwargs"])
  let args = object_field(body, "kwargs")
  validate_known(args, [
    "attn_proj", "decoder_dim", "decoder_rates", "decoder_type", "encoder_dim", "encoder_rates",
    "latent_dim", "sample_rate", "vae_latent_channels",
  ])
  guard required(args, "attn_proj") is True &&
    string_field(args, "decoder_type") == "bigvgan" &&
    int_field(args, "decoder_dim") == 1024 &&
    int_field(args, "encoder_dim") == 64 &&
    int_field(args, "latent_dim") == 2048 &&
    int_field(args, "sample_rate") == 32000 &&
    int_field(args, "vae_latent_channels") == 32 else {
    raise InvalidField
  }
  exact_int_array(args, "decoder_rates", [5, 5, 2, 2, 2, 2, 2])
  exact_int_array(args, "encoder_rates", [2, 4, 4, 5, 5])
  validate_audio_source_config(source_config, max_bytes)
  official_audio_vae_config()
}

///|
fn validate_audio_source_config(
  source : StringView,
  maximum : Int,
) -> Unit raise MiniMaxH3ConfigError {
  validate_utf8_size(source, maximum)
  let values : Map[String, Int] = Map([])
  let mut header = false
  for line in source.to_string().split("\n") {
    let trimmed = line.trim()
    if trimmed.is_empty() || trimmed.has_prefix("#") {
      continue
    }
    if !header {
      guard trimmed == "model_config:" && !line.has_prefix(" ") else {
        raise InvalidField
      }
      header = true
      continue
    }
    guard line.has_prefix("  ") && trimmed.split(":").to_array() is [key, raw] else {
      raise InvalidField
    }
    let name = key.trim().to_string()
    guard !values.contains(name) else { raise DuplicateField }
    let value = @string.parse_int(raw.trim()) catch { _ => raise InvalidField }
    values[name] = value
  }
  guard header &&
    values.length() == 4 &&
    values.get("sr") == Some(32000) &&
    values.get("decoder_dim") == Some(1024) &&
    values.get("audio_channel") == Some(1) &&
    values.get("vae_latent_channels") == Some(32) else {
    raise InvalidField
  }
}
