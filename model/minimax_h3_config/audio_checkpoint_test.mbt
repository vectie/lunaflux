///|
fn original_audio_wrapper() -> String {
  "{\"_class_name\":\"MiniMaxH3AudioVAE\",\"auto_map\":{\"AutoModel\":\"minimax_h3_audio_vae.MiniMaxH3AudioVAE\"},\"output_channel\":2,\"sample_rate\":32000,\"source_config_path\":\"config.yaml\",\"source_safetensors_path\":\"model.safetensors\",\"source_metadata_path\":\"metadata.json\",\"latent_channels\":32,\"latents_mean\":\{audio_latents_mean()},\"latents_std\":\{audio_latents_std()}}"
}

///|
fn original_audio_metadata() -> String {
  (
    #|{"metadata":{"kwargs":{"attn_proj":true,"decoder_dim":1024,"decoder_rates":[5,5,2,2,2,2,2],"decoder_type":"bigvgan","encoder_dim":64,"encoder_rates":[2,4,4,5,5],"latent_dim":2048,"sample_rate":32000,"vae_latent_channels":32}}}
  )
}

///|
fn original_audio_yaml() -> String {
  (
    #|model_config:
    #|  sr: 32000
    #|  decoder_dim: 1024
    #|  audio_channel: 1
    #|  vae_latent_channels: 32
  )
}

///|
test "original AudioVAE data documents resolve the native numeric contract" {
  let parsed = @minimax_h3_config.parse_audio_vae_checkpoint(
    original_audio_wrapper(),
    original_audio_metadata(),
    original_audio_yaml(),
    max_bytes=8192,
    max_depth=8,
  )
  assert_eq(parsed.decoder_dimension(), 1024)
  assert_eq(parsed.decoder_rates(), [5, 5, 2, 2, 2, 2, 2])
  assert_eq(parsed.output_channels(), 2)
  assert_eq(parsed.sampling_rate(), 32000)
  assert_eq(parsed.latent_means().length(), 32)
}

///|
test "original AudioVAE rejects mismatched wrapper metadata and source dimensions" {
  for
    (wrapper, metadata, yaml) in [
      (
        original_audio_wrapper().replace(
          old="minimax_h3_audio_vae.MiniMaxH3AudioVAE",
          new="other.Executable",
        ),
        original_audio_metadata(),
        original_audio_yaml(),
      ),
      (
        original_audio_wrapper().replace(old="32000", new="44100"),
        original_audio_metadata(),
        original_audio_yaml(),
      ),
      (
        original_audio_wrapper(),
        original_audio_metadata().replace(old="bigvgan", new="other"),
        original_audio_yaml(),
      ),
      (
        original_audio_wrapper(),
        original_audio_metadata().replace(old="1024", new="512"),
        original_audio_yaml(),
      ),
      (
        original_audio_wrapper(),
        original_audio_metadata(),
        original_audio_yaml().replace(old="1024", new="512"),
      ),
      (
        original_audio_wrapper(),
        original_audio_metadata(),
        original_audio_yaml() + "\n  sr: 32000",
      ),
      (
        original_audio_wrapper(),
        original_audio_metadata(),
        original_audio_yaml() + "\n  extension: 1",
      ),
    ] {
    try
      @minimax_h3_config.parse_audio_vae_checkpoint(
        wrapper,
        metadata,
        yaml,
        max_bytes=8192,
        max_depth=8,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("inconsistent original AudioVAE admitted")
    }
  }
}
