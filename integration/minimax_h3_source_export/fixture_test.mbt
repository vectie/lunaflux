///|
fn audio_config() -> @config.MiniMaxH3AudioVaeConfig raise {
  let means = "[-0.020211687488382354,0.3876466479950502,-0.04398279799186767,-0.28591514936373,0.08179686214561671,-0.35782641352446604,0.040623809960919084,-0.01552534501956604,-0.223362481667332,0.1821006842509091,0.2941778783780663,-0.07901167601970885,-0.056815072777201,-0.3699028221860095,-0.31616315591624855,0.5905951377425391,-0.052139568068853864,0.013673160263486295,-0.03691647864630577,0.09732660653298163,-0.3394662328788498,-0.30685677538541667,-0.24504598907458763,-0.034698524462007344,0.02868032184767538,-0.21217779266454084,-0.1678263169941987,0.3221287889040614,-0.1223055851554907,0.4356604928128464,-0.0502599202236253,0.3979258376211797]"
  let std = "[1.6895524230479284,2.76263727217653,1.7945344281264435,1.6801681847309828,1.6390226546605453,2.7788298348882177,1.7659090095747236,1.6199757612137327,2.6336525640336896,1.8539356672817833,2.5056497896915633,1.811019237886178,1.9579657790720237,1.6685498243529284,1.4922469314453364,3.298670198067373,1.9491804496832168,1.8720003270431442,1.8334080103291832,1.6488070416529093,1.6176957696319716,1.9131449234774398,1.5695245398428617,1.6943659940415912,1.8318420762504692,1.5540637421583379,1.9344930328968526,1.599198216109855,1.718045989838149,1.6307219190837705,1.8661226051202384,1.5613768203168363]"
  @config.parse_audio_vae(
    "{\"_class_name\":\"AutoencoderKLMiniMaxH3Audio\",\"encoder_dim\":64,\"encoder_rates\":[2,4,4,5,5],\"latent_dim\":2048,\"latent_channels\":32,\"decoder_dim\":1024,\"decoder_rates\":[5,5,2,2,2,2,2],\"decoder_kernel_sizes\":[9,9,4,4,4,4,4],\"num_attention_heads\":8,\"resblock_kernel_sizes\":[3,7,11],\"resblock_dilation_sizes\":[[1,3,5],[1,3,5],[1,3,5]],\"sampling_rate\":32000,\"latents_mean\":\{means},\"latents_std\":\{std}}",
    max_bytes=8192,
    max_depth=8,
  )
}

///|
fn text_config() -> @text.EncoderConfig raise {
  @text.parse_config(
    @utf8.encode(
      (
        #|{"model_type":"qwen3_vl","image_token_id":151655,"video_token_id":151656,"vision_start_token_id":151652,"vision_end_token_id":151653,"text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"head_dim":128,"hidden_act":"silu","hidden_size":5120,"intermediate_size":25600,"max_position_embeddings":262144,"num_attention_heads":64,"num_hidden_layers":64,"num_key_value_heads":8,"rms_norm_eps":1e-6,"rope_scaling":{"mrope_interleaved":true,"mrope_section":[24,20,20],"rope_type":"default"},"rope_theta":5000000,"vocab_size":151936}}
      ),
    ),
    max_bytes=4096,
  )
}

///|
test "complete offline export emits every schedule variant and deterministic source inventory" {
  let metadata = @spec.MiniMaxH3ModelMetadata::from_verified_content(
    @spec.MiniMaxH3TransformerSpec::official(
      variant=Fl2va,
      checkpoint_format=DiffusersV1,
    ),
    @identity.ContentDigest::from_sha256(String::make(64, '7')),
  )
  let model = @h3.build(metadata)
  let request = model
    .shapes()
    .resolve_request(
      height=32,
      width=32,
      requested_frames=120,
      inference_steps=5,
    )
  let plan = @loop.PreparedPlan::new(
    model,
    request,
    max_latent_elements=100000000L,
  )
  let sequence = @sequence.Sequence::from_request(
    model,
    request,
    text_rows=3,
    conditioning=[],
    max_rows=10000,
  )
  let text = @text.TextEncoderPlan::new(rows=3, epsilon=0.000001)
  let audio = audio_config()
  let config = text_config()
  let vision = @vision.VisionPlan::new(
    [{ time: 1, height: 2, width: 2, }],
    max_workspace_bytes=10000000L,
    max_output_bytes=10000000L,
  )
  fn make_export() -> @minimax_h3_source_export.SourceExport raise {
    @minimax_h3_source_export.build(
      metadata,
      plan,
      sequence,
      audio,
      text,
      text_config=config,
      visual_noise=0.0,
      audio_noise=0.0,
      vision_text_rows=[1],
      max_metadata_bytes=1000000L,
      max_workspace_bytes=1000000000L,
      tile_size=32,
      minimum_overlap=0,
      max_tiles=16,
      vision~,
    )
  }
  let first = make_export()
  let second = make_export()
  let audio_only = @minimax_h3_source_export.build_audio(metadata, plan, audio)
  assert_eq(audio_only.units.length(), 17)
  for entry in audio_only.units {
    guard first.units[:].filter(u => u.name == entry.name) is [full] else {
      fail("audio component absent from full export")
    }
    assert_eq(full.source, entry.source)
    assert_eq(full.symbols, entry.symbols)
    assert_eq(full.geometry, entry.geometry)
  }
  let decoder_sources = @block.build_audio_decoder(
    @vae.bind_audio_decoder_upsample_stages(metadata, audio, request)[:],
  )
  let decoder = @audio_checkpoint.CheckpointAudioDecoder::new(
    decoder_sources,
    maximum_workspace_bytes=2147483648L,
  )
  let memory = decoder.memory()
  assert_eq(
    memory.input_bytes,
    2L * 32L * request.audio_latent_frames().to_int64() * 4L,
  )
  assert_eq(
    memory.output_bytes,
    2L * request.audio_latent_frames().to_int64() * 800L * 4L,
  )
  assert_eq(
    memory.total_bytes,
    memory.frontend_bytes +
    memory.stage_bytes[:].fold(init=0L, (a, b) => a + b) +
    memory.chain_bytes +
    memory.bridge_bytes +
    memory.tail_bytes +
    memory.cache_bytes +
    memory.input_bytes +
    memory.output_bytes,
  )
  let exact = @audio_checkpoint.CheckpointAudioDecoder::new(
    decoder_sources,
    maximum_workspace_bytes=memory.total_bytes,
  )
  assert_eq(exact.memory().total_bytes, memory.total_bytes)
  exact.close()
  decoder.close()
  try
    @audio_checkpoint.CheckpointAudioDecoder::new(
      decoder_sources,
      maximum_workspace_bytes=memory.total_bytes - 1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("decoder aggregate admitted one byte beyond limit")
  }
  assert_eq(first.manifest(), second.manifest())
  assert_true(first.timestep_counts.contains(1))
  assert_true(first.timestep_counts.contains(2))
  for count in first.timestep_counts {
    for prefix in ["dit_n", "timestep_n", "output_n"] {
      assert_true(first.units.any(u => u.name == prefix + count.to_string()))
    }
  }
  for
    name in [
      "vision", "flow_video", "flow_audio", "video_pack", "text", "conditioning",
      "refiner", "input", "video_decode", "audio_input", "audio_front", "audio_tail",
    ] {
    assert_true(first.units.any(u => u.name == name))
  }
  for i in 0..<7 {
    assert_true(first.units.any(u => u.name == "audio_up_\{i}"))
    assert_true(first.units.any(u => u.name == "audio_amp_\{i}"))
  }
  for i, entry in first.units {
    assert_eq(entry.source, second.units[i].source)
  }
  let media = first.with_reference_audio(
    ordinal=0,
    samples=800,
    max_scratch_bytes=1000000000L,
  )
  assert_eq(media.units.length(), first.units.length() + 1)
  try
    media.with_reference_audio(
      ordinal=0,
      samples=800,
      max_scratch_bytes=1000000000L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("duplicate module accepted")
  }
  let media = media.with_reference_video(
    ordinal=0,
    frames=1,
    height=32,
    width=32,
    image=true,
    max_tiles=1,
    max_scratch_bytes=2000000000L,
  )
  assert_eq(media.units.length(), first.units.length() + 2)
  let wrong = @sequence.Sequence::new(
    hidden_width=model.denoiser().hidden_size(),
    text_rows=3,
    video_rows=1,
    audio_rows=request.audio_token_rows(),
    conditioning=[],
    max_rows=10000,
  )
  try
    @minimax_h3_source_export.build(
      metadata,
      plan,
      wrong,
      audio,
      text,
      text_config=config,
      visual_noise=0.0,
      audio_noise=0.0,
      vision_text_rows=[],
      max_metadata_bytes=1000000L,
      max_workspace_bytes=1000000000L,
      tile_size=32,
      minimum_overlap=0,
      max_tiles=16,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("same text but wrong target geometry accepted")
  }
  let wrong_epsilon = @text.TextEncoderPlan::new(rows=3, epsilon=0.00001)
  try
    @minimax_h3_source_export.build(
      metadata,
      plan,
      sequence,
      audio,
      wrong_epsilon,
      text_config=config,
      visual_noise=0.0,
      audio_noise=0.0,
      vision_text_rows=[],
      max_metadata_bytes=1000000L,
      max_workspace_bytes=1000000000L,
      tile_size=32,
      minimum_overlap=0,
      max_tiles=16,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("DiT epsilon accepted as text epsilon")
  }
  for indices in [[], [1, 1], [0, 1]] {
    try
      @minimax_h3_source_export.build(
        metadata,
        plan,
        sequence,
        audio,
        text,
        text_config=config,
        visual_noise=0.0,
        audio_noise=0.0,
        vision_text_rows=indices,
        max_metadata_bytes=1000000L,
        max_workspace_bytes=1000000000L,
        tile_size=32,
        minimum_overlap=0,
        max_tiles=16,
        vision~,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("vision output row mismatch accepted")
    }
  }
}
