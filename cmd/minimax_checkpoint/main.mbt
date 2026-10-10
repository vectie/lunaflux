///|
fn read(
  root : @approved_fs.ApprovedRoot,
  name : String,
  maximum : Int64,
) -> Bytes raise {
  let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(name))
  defer file.close()
  file.read_immutable_snapshot(maximum_bytes=maximum)
}

///|
fn component(
  name : String,
  variant : @h3.MiniMaxH3Variant,
) -> (@weights.MiniMaxH3WeightComponent, String) raise {
  match name {
    "transformer" =>
      (
        if variant == Fl2va {
          Fl2vaDenoiser
        } else {
          Ref2vaDenoiser
        },
        "transformer",
      )
    "text_encoder" => (TextConditioner, "text_encoder")
    "video_vae" => (VideoVae, "video_vae/source")
    "audio_vae" => (AudioVae, "audio_vae")
    _ => fail("unknown H3 component")
  }
}

///|
async fn main {
  guard @env.args() is [_, mode, root_input, .. tail] &&
    (
      (
        (
          mode == "config" ||
          mode == "weight-plan" ||
          mode == "audio-config" ||
          mode == "video-config"
        ) &&
        tail.is_empty()
      ) ||
      ((mode == "export-audio" || mode == "audio-memory") && tail.length() == 5) ||
      (mode == "export-video" && tail.length() == 9) ||
      (mode == "export-request" && tail.length() == 10) ||
      (mode == "encode-text" && tail.length() == 7) ||
      (mode == "encode-text-layers" && tail.length() == 8) ||
      (mode == "text-ids" && tail.length() == 4) ||
      (mode == "decode-audio" && tail.length() == 9) ||
      (mode == "decode-video" && tail.length() == 12) ||
      (mode == "run-joint" && tail.length() == 18) ||
      (
        (mode == "inspect-component" || mode == "prepare-component") &&
        tail.length() == 4
      )
    ) else {
    fail(
      "text-ids VARIANT_ROOT TOKENIZER_LABEL INPUT_UTF8_FILE NEW_OUTPUT_DIRECTORY MAX_TOKENS | run-joint VARIANT_ROOT HEIGHT WIDTH FRAMES STEPS TEXT_ROWS ENCODED_BF16 ROTARY_F32 AOT_DIRECTORY DENOISER_INVENTORY AUDIO_INVENTORY VIDEO_INVENTORY TILE_SIZE OVERLAP MAX_TILES WORKSPACE_LIMIT DEVICE_LIMIT SEED OUTPUT_DIRECTORY | export-request VARIANT_ROOT HEIGHT WIDTH FRAMES STEPS TEXT_ROWS TILE_SIZE OVERLAP MAX_TILES WORKSPACE_LIMIT OUTPUT_DIRECTORY | encode-text VARIANT_ROOT INVENTORY TEXT_ROWS INPUT_IDS_I32 AOT_DIRECTORY OUTPUT_BF16 COMPONENT_LIMIT DEVICE_LIMIT | encode-text-layers VARIANT_ROOT INVENTORY TEXT_ROWS INPUT_IDS_I32 AOT_DIRECTORY NEW_OUTPUT_DIRECTORY COMPONENT_LIMIT DEVICE_LIMIT HOST_CAPTURE_LIMIT | " +
      "config|weight-plan|audio-config|video-config VARIANT_ROOT | export-audio|audio-memory VARIANT_ROOT HEIGHT WIDTH FRAMES STEPS OUTPUT_DIRECTORY_OR_WORKSPACE_LIMIT | export-video VARIANT_ROOT HEIGHT WIDTH FRAMES STEPS TILE_SIZE OVERLAP MAX_TILES WORKSPACE_LIMIT OUTPUT_DIRECTORY | decode-audio VARIANT_ROOT INVENTORY HEIGHT WIDTH FRAMES STEPS AOT_DIRECTORY INPUT_F32 OUTPUT_F32 DEVICE_BYTE_LIMIT | decode-video VARIANT_ROOT INVENTORY HEIGHT WIDTH FRAMES STEPS TILE_SIZE OVERLAP MAX_TILES AOT_DIRECTORY INPUT_F32 OUTPUT_F32 DEVICE_BYTE_LIMIT | inspect-component|prepare-component VARIANT_ROOT COMPONENT SHA256_INVENTORY_RELATIVE_TO_VARIANT COMPONENT_BYTE_LIMIT DEVICE_BYTE_LIMIT",
    )
  }
  let (root_name, supplied_label) = checkpoint_location(root_input)
  let root = @approved_fs.ApprovedRoot::open_absolute(root_name)
  defer root.close()
  if mode == "text-ids" {
    prepare_text_ids(root_name, root, tail)
    return
  }
  let pipeline = @config.parse_pipeline_index(
    @utf8.decode(read(root, "model_index.json", 1048576L)),
    max_bytes=1048576,
    max_depth=16,
  )
  let document = read(root, "transformer/config.json", 1048576L)
  let metadata = @config.parse_transformer(
    @utf8.decode(document),
    variant=pipeline.variant(),
    content_digest=checkpoint_label(root, supplied_label, mode, tail),
    max_bytes=1048576,
    max_depth=16,
  )
  let plan = @minimax_h3.build(metadata)
  println(
    "MiniMax-H3 layers=\{plan.denoiser().layer_count()} hidden=\{plan.denoiser().hidden_size()}",
  )
  if mode == "config" {
    return
  }
  if mode == "export-request" {
    export_request(root, metadata, plan, tail)
    return
  }
  if mode == "encode-text" || mode == "encode-text-layers" {
    run_text(
      root_name,
      root,
      metadata,
      plan,
      tail,
      capture_layers=mode == "encode-text-layers",
    )
    return
  }
  if mode == "run-joint" {
    run_joint(root_name, root, metadata, plan, tail)
    return
  }
  if mode == "video-config" || mode == "export-video" {
    let video = read_video_config(root)
    println(
      "VideoVAE latent_channels=\{video.latent_channels()} layers=\{video.decoder_layer_count()} hidden=\{video.decoder_hidden_size()} compression=\{video.spatial_compression()}x\{video.temporal_compression()}",
    )
    if mode == "export-video" {
      let shape = plan
        .shapes()
        .resolve_request(
          height=@string.parse_int(tail[0]),
          width=@string.parse_int(tail[1]),
          requested_frames=@string.parse_int(tail[2]),
          inference_steps=@string.parse_int(tail[3]),
        )
      let prepared = @loop.PreparedPlan::new(
        plan,
        shape,
        max_latent_elements=2147483647L,
      )
      let sources = @export.build_video(
        metadata,
        prepared,
        tile_size=@string.parse_int(tail[4]),
        minimum_overlap=@string.parse_int(tail[5]),
        max_tiles=@string.parse_int(tail[6]),
        max_workspace_bytes=@string.parse_int64(tail[7]),
      )
      sources.write(tail[8])
      for (name, bytes) in sources.units[0].geometry {
        println("\{name}=\{bytes}")
      }
    }
    return
  }
  if mode == "decode-audio" {
    run_audio(root_name, root, metadata, plan, tail)
    return
  }
  if mode == "decode-video" {
    run_video(root_name, root, metadata, plan, tail)
    return
  }
  if mode == "audio-config" || mode == "export-audio" || mode == "audio-memory" {
    let audio = read_audio_config(root)
    println(
      "AudioVAE channels=\{audio.output_channels()} sample_rate=\{audio.sampling_rate()} decoder_stages=\{audio.decoder_rates().length()}",
    )
    if mode == "audio-config" {
      return
    }
    let shape = plan
      .shapes()
      .resolve_request(
        height=@string.parse_int(tail[0]),
        width=@string.parse_int(tail[1]),
        requested_frames=@string.parse_int(tail[2]),
        inference_steps=@string.parse_int(tail[3]),
      )
    let prepared = @loop.PreparedPlan::new(
      plan,
      shape,
      max_latent_elements=2147483647L,
    )
    if mode == "audio-memory" {
      let sources = @source.build_audio_decoder(
        @vae.bind_audio_decoder_upsample_stages(metadata, audio, shape)[:],
      )
      let decoder = @audio.CheckpointAudioDecoder::new(
        sources,
        maximum_workspace_bytes=@string.parse_int64(tail[4]),
      )
      defer decoder.close()
      let m = decoder.memory()
      println(
        "input_bytes=\{m.input_bytes} output_bytes=\{m.output_bytes} frontend_bytes=\{m.frontend_bytes} tail_bytes=\{m.tail_bytes} chain_bytes=\{m.chain_bytes} bridge_bytes=\{m.bridge_bytes} cache_bytes=\{m.cache_bytes} total_workspace_bytes=\{m.total_bytes}",
      )
      for i, bytes in m.stage_bytes {
        println("stage=\{i} workspace_bytes=\{bytes}")
      }
      return
    }
    let source = @export.build_audio(metadata, prepared, audio)
    source.write(tail[4])
    println(
      "audio_aot_modules=\{source.units.length()} latent_frames=\{shape.audio_latent_frames()}",
    )
    return
  }
  if mode == "weight-plan" {
    for name in ["text_encoder", "transformer", "video_vae", "audio_vae"] {
      let (kind, _) = component(name, pipeline.variant())
      let packings = @checkpoint.component_packings(
        kind,
        plan,
        maximum_device_bytes=103079215104L,
      )
      let required = @startup.component_bytes(kind, packings[:], 103079215104L)
      println("component=\{name} packed_device_weight_bytes=\{required}")
    }
    println(
      "weight-only plan: excludes request workspace, output buffers and transfer staging; not a physical admission",
    )
    return
  }
  let (kind, directory) = component(tail[0], pipeline.variant())
  let component_root = @approved_fs.ApprovedRoot::open_absolute(
    root_name + "/" + directory,
  )
  defer component_root.close()
  // Keep inventories outside the read-only model component if necessary.
  let inventory = @inventory.CheckpointInventory::parse(
    @utf8.decode(read(root, tail[1], 1048576L)),
    maximum_bytes=1048576,
    maximum_shards=64,
  )
  let checkpoint = @checkpoint.ComponentCheckpoint::inspect(
    component_root,
    metadata,
    plan,
    kind,
    inventory,
    maximum_component_bytes=@string.parse_int64(tail[2]),
    maximum_device_bytes=@string.parse_int64(tail[3]),
  )
  println(
    "component=\{tail[0]} tensors=\{checkpoint.manifest().tensor_count()} checkpoint_bytes=\{checkpoint.manifest().total_tensor_bytes()} device_weight_bytes=\{checkpoint.required_device_bytes()}",
  )
  if mode == "inspect-component" {
    return
  }
  let context = @device.open_context(0)
  defer context.close()
  let owner = @startup.WeightStartup::new(
    max_device_bytes=@string.parse_int64(tail[3]),
  )
  defer owner.close()
  checkpoint.prepare(owner, context, component_root, chunk_bytes=1048576)
  println(
    "prepared_components=\{owner.count()} reserved_weight_bytes=\{owner.reserved_bytes()}",
  )
  owner.close()
  println(
    "component GPU weights released; no full-request or kernel correctness claim",
  )
}
