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
fn main raise {
  guard @env.args() is [_, mode, root_name, .. tail] &&
    (
      ((mode == "config" || mode == "weight-plan") && tail.is_empty()) ||
      (
        (mode == "inspect-component" || mode == "prepare-component") &&
        tail.length() == 4
      )
    ) else {
    fail(
      "config|weight-plan VARIANT_ROOT | inspect-component|prepare-component VARIANT_ROOT COMPONENT SHA256_INVENTORY_RELATIVE_TO_VARIANT COMPONENT_BYTE_LIMIT DEVICE_BYTE_LIMIT",
    )
  }
  let root = @approved_fs.ApprovedRoot::open_absolute(root_name)
  defer root.close()
  let pipeline = @config.parse_pipeline_index(
    @utf8.decode(read(root, "model_index.json", 1048576L)),
    max_bytes=1048576,
    max_depth=16,
  )
  let document = read(root, "transformer/config.json", 1048576L)
  let metadata = @config.parse_transformer(
    @utf8.decode(document),
    variant=pipeline.variant(),
    content_digest=@spec.ContentDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(document)),
    ),
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
