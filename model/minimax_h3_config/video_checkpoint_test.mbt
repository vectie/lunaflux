///|
fn original_video_wrapper() -> String {
  "{\"_class_name\":\"MiniMaxH3VideoVAE\",\"auto_map\":{\"AutoModel\":\"minimax_h3_video_vae.MiniMaxH3VideoVAE\"},\"mode\":\"standalone\",\"source_path\":\"source\",\"source_class_name\":\"AutoencoderKLLegacy\",\"vae_clip_length\":17,\"vae_token_drop\":3,\"vae_encoder_tiling\":1,\"vae_decoder_tiling\":1,\"vae_parallel_tiling\":1,\"vae_tile_size\":256,\"vae_tile_overlap_min\":64,\"vae_encoder_parallel\":0,\"vae_decoder_parallel\":0,\"vae_chunk_dim\":-1,\"source_safetensors_path\":\"model.safetensors\",\"latent_channels\":24,\"latents_mean\":\{video_latents_mean()},\"latents_std\":\{video_latents_std()}}"
}

///|
fn original_video_source() -> String {
  (
    #|{"_class_name":"AutoencoderKLLegacy","causal_decoder":false,"causal_encoder":true,"ch":128,"ch_mult":[1,2,2,4,4,8],"embed_dim":24,"in_channels":3,"num_res_blocks":2,"num_res_blocks_decoder":null,"out_ch":3,"padding_mode":"reflect","padding_mode_t":null,"pixel_norm_type":"imagenet","scaling_factor":1.0,"shift_factor":0.0,"space_down":[2,2,2,2,1,1],"space_up":[1,2,2,2,2,1],"time_down":[1,2,2,1,1,1],"time_up":null,"use_3d_conv":true,"use_t_isolated_gn":true,"use_vit_decoder":true,"vae_ratio":16,"vae_ratio_t":4,"vit_decoder_kwargs":{"dim_head":64,"ffn_activation_fn":"silu","ffn_use_gated":true,"heads":32,"norm_affine":true,"norm_type":"rms_norm","num_layers":36,"qk_norm_affine":false,"qk_norm_type":"rms_norm","rope_dim_ratio":0.75,"rope_theta":100.0},"z_channels":24,"zq_ch_decoder":null,"zq_ch_encoder":null}
  )
}

///|
test "original VideoVAE resolves the same decoder and latent numeric contract" {
  let video = @minimax_h3_config.parse_video_vae_checkpoint(
    original_video_wrapper(),
    original_video_source(),
    max_bytes=8192,
    max_depth=8,
  )
  assert_eq(video.decoder_layer_count(), 36)
  assert_eq(video.decoder_hidden_size(), 2048)
  assert_eq(video.decoder_register_token_count(), 4)
  assert_eq(video.decoder_rope_dimension_ratio(), 0.75)
  assert_eq(video.spatial_compression(), 16)
  assert_eq(video.temporal_compression(), 4)
  assert_eq(video.latent_channels(), 24)
  assert_eq(
    video.latent_means()[0].reinterpret_as_uint(),
    Float::from_double(0.858090341091156).reinterpret_as_uint(),
  )
}

///|
test "original VideoVAE detects source and wrapper semantic mismatches" {
  for
    (wrapper, source) in [
      (
        original_video_wrapper().replace(
          old="vae_token_drop\":3",
          new="vae_token_drop\":2",
        ),
        original_video_source(),
      ),
      (
        original_video_wrapper().replace(
          old="minimax_h3_video_vae.MiniMaxH3VideoVAE",
          new="other.Code",
        ),
        original_video_source(),
      ),
      (
        original_video_wrapper(),
        original_video_source().replace(
          old="causal_decoder\":false",
          new="causal_decoder\":true",
        ),
      ),
      (
        original_video_wrapper(),
        original_video_source().replace(
          old="num_layers\":36",
          new="num_layers\":24",
        ),
      ),
      (
        original_video_wrapper(),
        original_video_source().replace(
          old="qk_norm_affine\":false",
          new="qk_norm_affine\":true",
        ),
      ),
      (
        original_video_wrapper(),
        original_video_source().replace(
          old="z_channels\":24",
          new="z_channels\":24,\"z_channels\":24",
        ),
      ),
    ] {
    try
      @minimax_h3_config.parse_video_vae_checkpoint(
        wrapper,
        source,
        max_bytes=8192,
        max_depth=8,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("inconsistent original VideoVAE admitted")
    }
  }
}
