///|
fn h3_digest() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  )
}

///|
fn pipeline_index(partition : String, tasks : String) -> String {
  "{\"_class_name\":\"MiniMaxH3Pipeline\",\"_diffusers_version\":\"0.32.2\",\"text_encoder\":[\"transformers\",\"MiniMaxH3Qwen3VLHFEncoder\"],\"tokenizer\":[\"transformers\",\"Qwen2TokenizerFast\"],\"video_vae\":[\"diffusers\",\"MiniMaxH3VideoVAE\"],\"audio_vae\":[\"diffusers\",\"MiniMaxH3AudioVAE\"],\"scheduler\":null,\"transformer\":[\"diffusers\",\"MiniMaxH3DiTModel\"],\"processor\":[\"transformers\",\"Qwen3VLProcessor\"],\"_minimax_h3\":{\"schema_version\":1,\"partition\":\"\{partition}\",\"tasks\":\{tasks},\"task_aliases\":{},\"sigma_shift_scales\":{\"video\":12.0,\"audio\":3.0}}}"
}

///|
fn transformer_config() -> String {
  "{\"_class_name\":\"MiniMaxH3Transformer3DModel\",\"_diffusers_version\":\"0.36.0.dev0\",\"num_attention_heads\":56,\"attention_head_dim\":128,\"hidden_size\":5376,\"num_layers\":50,\"num_refiner_layers\":2,\"ffn_dim\":14336,\"in_channels\":24,\"audio_in_channels\":32,\"patch_size\":[1,2,2],\"text_dim\":5120,\"freq_dim\":256,\"time_embed_hidden_dim\":5376,\"time_embed_dim\":2688,\"rope_freq_dim\":16,\"rope_theta\":10000.0,\"norm_eps\":1e-5,\"qk_norm_eps\":1e-5,\"final_norm_eps\":1e-5}"
}

///|
fn original_transformer_config() -> String {
  "{\"_class_name\":\"MiniMaxH3DiTModel\",\"_diffusers_version\":\"0.32.2\",\"hidden_size\":5376,\"num_layers\":50,\"token_refiner_num_layers\":2,\"num_attention_heads\":56,\"attention_head_dim\":128,\"ffn_hidden_size\":14336,\"latents_dim\":24,\"audio_latents_dim\":32,\"patch_size\":[1,2,2],\"text_dim\":5120,\"timestep_input_dim\":256,\"time_embed_hidden_size\":5376,\"time_embed_dim\":2688,\"adaln_out_features\":96768,\"final_adaln_out_features\":10752,\"rope_inv_freq_len\":16,\"eps\":1e-5}"
}

///|
fn video_latents_mean() -> String {
  "[0.858090341091156,-0.9606591463088989,1.0661640167236328,-0.5090325474739075,-0.2727581858634949,-1.3675414323806763,-0.2553254961967468,-0.26907554268836975,-0.5376840829849243,-0.0464097298681736,0.6657370328903198,0.19690127670764923,-0.5460608005523682,-0.4035342037677765,-0.23683024942874908,0.25928452610969543,-0.30133944749832153,0.211341992020607,-1.1206848621368408,0.3581933379173279,-0.04225143790245056,0.2604829967021942,0.22864092886447906,0.7056031823158264]"
}

///|
fn video_latents_std() -> String {
  "[1.2223774194717407,1.2767263650894165,1.6831774711608887,1.7549455165863037,1.5636216402053833,2.194143533706665,0.9653137922286987,1.0569885969161987,0.841948926448822,0.7729952931404114,1.8955937623977661,0.946841835975647,0.7996809482574463,0.44988900423049927,0.7197399735450745,0.6936293244361877,2.961095094680786,2.7694199085235596,3.0496184825897217,2.1088054180145264,3.276226282119751,3.1627357006073,2.2816812992095947,2.6127843856811523]"
}

///|
fn audio_latents_mean() -> String {
  "[-0.020211687488382354,0.3876466479950502,-0.04398279799186767,-0.28591514936373,0.08179686214561671,-0.35782641352446604,0.040623809960919084,-0.01552534501956604,-0.223362481667332,0.1821006842509091,0.2941778783780663,-0.07901167601970885,-0.056815072777201,-0.3699028221860095,-0.31616315591624855,0.5905951377425391,-0.052139568068853864,0.013673160263486295,-0.03691647864630577,0.09732660653298163,-0.3394662328788498,-0.30685677538541667,-0.24504598907458763,-0.034698524462007344,0.02868032184767538,-0.21217779266454084,-0.1678263169941987,0.3221287889040614,-0.1223055851554907,0.4356604928128464,-0.0502599202236253,0.3979258376211797]"
}

///|
fn audio_latents_std() -> String {
  "[1.6895524230479284,2.76263727217653,1.7945344281264435,1.6801681847309828,1.6390226546605453,2.7788298348882177,1.7659090095747236,1.6199757612137327,2.6336525640336896,1.8539356672817833,2.5056497896915633,1.811019237886178,1.9579657790720237,1.6685498243529284,1.4922469314453364,3.298670198067373,1.9491804496832168,1.8720003270431442,1.8334080103291832,1.6488070416529093,1.6176957696319716,1.9131449234774398,1.5695245398428617,1.6943659940415912,1.8318420762504692,1.5540637421583379,1.9344930328968526,1.599198216109855,1.718045989838149,1.6307219190837705,1.8661226051202384,1.5613768203168363]"
}

///|
test "family parser admits exact FL2VA and Ref2VA pipeline indexes" {
  let fl = @minimax_h3_config.parse_pipeline_index(
    pipeline_index("fl2va", "[\"t2va\",\"fl2va\"]"),
    max_bytes=4096,
    max_depth=8,
  )
  let reference = @minimax_h3_config.parse_pipeline_index(
    pipeline_index("ref2va", "[\"ref2va\"]"),
    max_bytes=4096,
    max_depth=8,
  )
  assert_eq(fl.variant(), Fl2va)
  assert_eq(reference.variant(), Ref2va)
  assert_eq(fl.video_shift(), 12.0)
  assert_eq(reference.audio_shift(), 3.0)
}

///|
test "family parser constructs content-bound exact transformer metadata" {
  let metadata = @minimax_h3_config.parse_transformer(
    transformer_config(),
    variant=Ref2va,
    content_digest=h3_digest(),
    max_bytes=4096,
    max_depth=8,
  )
  assert_eq(metadata.spec().variant(), Ref2va)
  assert_eq(metadata.spec().attention_width(), 7168)
  assert_eq(metadata.content_digest(), h3_digest())
  let original = @minimax_h3_config.parse_transformer(
    original_transformer_config(),
    variant=Fl2va,
    content_digest=h3_digest(),
    max_bytes=4096,
    max_depth=8,
  )
  assert_eq(original.spec().checkpoint_format(), OriginalV1)
  assert_eq(original.spec().adaln_output_features(), 96768)
}

///|
test "family parser admits exact VAE and scheduler geometry" {
  let video = "{\"_class_name\":\"AutoencoderKLMiniMaxH3\",\"_diffusers_version\":\"0.36.0.dev0\",\"in_channels\":3,\"out_channels\":3,\"latent_channels\":24,\"block_out_channels\":[128,256,256,512,512,1024],\"layers_per_block\":2,\"spatial_downsample_factors\":[2,2,2,2,1,1],\"temporal_downsample_factors\":[1,2,2,1,1,1],\"norm_num_groups\":32,\"norm_eps\":1e-6,\"spatial_padding_mode\":\"reflect\",\"decoder_num_layers\":36,\"decoder_num_attention_heads\":32,\"decoder_attention_head_dim\":64,\"decoder_num_register_tokens\":4,\"decoder_ffn_mult\":4,\"decoder_rope_theta\":100.0,\"decoder_rope_dim_ratio\":0.75,\"decoder_norm_eps\":1e-5,\"clip_length\":17,\"token_drop\":3,\"latents_mean\":\{video_latents_mean()},\"latents_std\":\{video_latents_std()}}"
  let audio = "{\"_class_name\":\"AutoencoderKLMiniMaxH3Audio\",\"_diffusers_version\":\"0.36.0.dev0\",\"encoder_dim\":64,\"encoder_rates\":[2,4,4,5,5],\"latent_dim\":2048,\"latent_channels\":32,\"decoder_dim\":1024,\"decoder_rates\":[5,5,2,2,2,2,2],\"decoder_kernel_sizes\":[9,9,4,4,4,4,4],\"num_attention_heads\":8,\"resblock_kernel_sizes\":[3,7,11],\"resblock_dilation_sizes\":[[1,3,5],[1,3,5],[1,3,5]],\"sampling_rate\":32000,\"latents_mean\":\{audio_latents_mean()},\"latents_std\":\{audio_latents_std()}}"
  let video_config = @minimax_h3_config.parse_video_vae(
    video,
    max_bytes=8192,
    max_depth=8,
  )
  let audio_config = @minimax_h3_config.parse_audio_vae(
    audio,
    max_bytes=8192,
    max_depth=8,
  )
  assert_eq(audio_config.latent_means().length(), 32)
  assert_eq(audio_config.latent_standard_deviations().length(), 32)
  assert_eq(
    audio_config.latent_means()[0].reinterpret_as_uint(),
    Float::from_double(-0.020211687488382354).reinterpret_as_uint(),
  )
  assert_eq(
    audio_config.latent_standard_deviations()[31].reinterpret_as_uint(),
    Float::from_double(1.5613768203168363).reinterpret_as_uint(),
  )
  assert_eq(video_config.spatial_compression(), 16)
  assert_eq(video_config.temporal_compression(), 4)
  assert_eq(video_config.architecture_schema_version(), 1)
  assert_eq(video_config.block_out_channels(), [128, 256, 256, 512, 512, 1024])
  assert_eq(video_config.decoder_hidden_size(), 2048)
  assert_eq(video_config.decoder_ffn_dimension(), 8192)
  assert_eq(video_config.decoder_register_token_count(), 4)
  assert_eq(video_config.decoder_rope_dimension_ratio(), 0.75)
  assert_eq(video_config.latent_means().length(), 24)
  assert_eq(video_config.latent_standard_deviations().length(), 24)
  assert_eq(
    video_config.latent_means()[0].reinterpret_as_uint(),
    Float::from_double(0.858090341091156).reinterpret_as_uint(),
  )
  assert_eq(
    video_config.latent_standard_deviations()[23].reinterpret_as_uint(),
    Float::from_double(2.6127843856811523).reinterpret_as_uint(),
  )
  try
    @minimax_h3_config.parse_video_vae(
      video.replace(old="0.858090341091156", new="0.0"),
      max_bytes=8192,
      max_depth=8,
    )
  catch {
    InvalidField => ()
    _ => fail("unexpected latent-scaling rejection")
  } noraise {
    _ => fail("shifted video-VAE latent scaling admitted")
  }
  assert_eq(audio_config.latent_channels(), 32)
  assert_eq(audio_config.latent_rate(), 40)
  assert_eq(audio_config.architecture_schema_version(), 1)
  assert_eq(audio_config.encoder_rates(), [2, 4, 4, 5, 5])
  assert_eq(audio_config.decoder_kernel_sizes(), [9, 9, 4, 4, 4, 4, 4])
  assert_eq(audio_config.attention_head_count(), 8)
  try
    @minimax_h3_config.parse_audio_vae(
      audio.replace(old="-0.020211687488382354", new="0.0"),
      max_bytes=8192,
      max_depth=8,
    )
  catch {
    InvalidField => ()
    _ => fail("unexpected audio latent-scaling rejection")
  } noraise {
    _ => fail("shifted AudioVae latent scaling admitted")
  }
  let scheduler = @minimax_h3_config.parse_scheduler(
    "{\"_class_name\":\"MiniMaxH3Scheduler\",\"_diffusers_version\":\"0.36.0.dev0\",\"shift\":12.0}",
    document_kind=VideoScheduler,
    max_bytes=512,
    max_depth=4,
  )
  assert_eq(scheduler.shift(), 12.0)
}

///|
test "family parser rejects duplicate unknown executable and shifted configs" {
  try
    @minimax_h3_config.parse_transformer(
      transformer_config().replace(
        old="\"hidden_size\":5376",
        new="\"hidden_size\":5376,\"hidden_size\":5376",
      ),
      variant=Fl2va,
      content_digest=h3_digest(),
      max_bytes=4096,
      max_depth=8,
    )
  catch {
    DuplicateField => ()
    _ => fail("unexpected duplicate-field result")
  } noraise {
    _ => fail("duplicate transformer field was admitted")
  }
  try
    @minimax_h3_config.parse_transformer(
      transformer_config().replace(
        old="\"_diffusers_version\":\"0.36.0.dev0\"",
        new="\"auto_map\":{}",
      ),
      variant=Fl2va,
      content_digest=h3_digest(),
      max_bytes=4096,
      max_depth=8,
    )
  catch {
    UnsupportedExecutableMetadata => ()
    _ => fail("unexpected executable-metadata result")
  } noraise {
    _ => fail("executable transformer metadata was admitted")
  }
  try
    @minimax_h3_config.parse_scheduler(
      "{\"_class_name\":\"MiniMaxH3Scheduler\",\"_diffusers_version\":\"0.36.0.dev0\",\"shift\":4.0}",
      document_kind=AudioScheduler,
      max_bytes=512,
      max_depth=4,
    )
  catch {
    InvalidField => ()
    _ => fail("unexpected scheduler-shift result")
  } noraise {
    _ => fail("wrong audio scheduler shift was admitted")
  }
}
