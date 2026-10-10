///|
priv struct Startup {
  spec : @spec.DeepSeekV4ModelSpec
  checkpoint : @materialize.PackedCheckpoint
  limits : @streaming.StreamingSafetensorsLimits
  placement : @placement.DecoderPipelinePlan
  rows : Int
  history : Int
  budgets : Array[Int64]
  reserve : Int64
}

///|
fn read_bounded(
  root : @approved_fs.ApprovedRoot,
  name : String,
  maximum_bytes : Int64,
) -> Bytes raise {
  let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(name))
  defer file.close()
  file.read_immutable_snapshot(maximum_bytes~)
}

///|
// A deployment may keep its read-only checkpoint and inventory in separate
// mounts. Absolute inventories are opened through a separate bounded root;
// shard locators still resolve exclusively inside the checkpoint root.
fn read_inventory(
  model_root : @approved_fs.ApprovedRoot,
  name : String,
) -> Bytes raise {
  if !name.has_prefix("/") {
    return read_bounded(model_root, name, 1048576L)
  }
  guard name.rev_find("/") is Some(separator) && separator > 0 else {
    fail("inventory requires an absolute parent directory and filename")
  }
  let inventory_root = @approved_fs.ApprovedRoot::open_absolute(
    name[:separator],
  )
  defer inventory_root.close()
  read_bounded(inventory_root, name[separator + 1:].to_owned(), 1048576L)
}

///|
fn stage_plan(
  startup : Startup,
  ordinal : Int,
) -> @deepseek.DeepSeekDecoderStage raise {
  guard startup.placement.stages().get(ordinal) is Some(range) else {
    fail("stage ordinal is outside the planned placement")
  }
  let ceiling = startup.budgets[ordinal] - startup.reserve
  @deepseek.DeepSeekDecoderStage::new(
    startup.spec,
    range,
    maximum_rows=startup.rows,
    maximum_context=startup.history,
    maximum_layer_state_bytes=ceiling,
    maximum_layer_workspace_bytes=ceiling,
    maximum_layer_expert_bytes=ceiling,
    maximum_text_workspace_bytes=ceiling,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
  )
}

///|
fn inspect_config(
  root : @approved_fs.ApprovedRoot,
) -> @spec.DeepSeekV4ModelSpec raise {
  let config = read_bounded(root, "config.json", 1048576L)
  let digest = @model_spec.ContentDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(config)),
  )
  @config.parse_bytes(config, digest, max_bytes=1048576, max_depth=16).spec()
}

///|
fn inspect_startup(
  root : @approved_fs.ApprovedRoot,
  inventory_name : String,
  rows : Int,
  history : Int,
  budgets : Array[Int64],
  reserve : Int64,
) -> Startup raise {
  guard budgets.length() == 2 &&
    rows > 0 &&
    history >= rows &&
    reserve > 0L &&
    budgets.all(b => b > reserve) else {
    fail("invalid geometry or per-host memory budget")
  }
  let spec = inspect_config(root)
  guard history <= spec.max_position_embeddings() else {
    fail("history exceeds the model context limit")
  }
  let inventory = @inventory.CheckpointInventory::parse(
    @utf8.decode(read_inventory(root, inventory_name)),
    maximum_bytes=1048576,
    maximum_shards=512,
  )
  let limits = checkpoint_limits(budgets[0].max(budgets[1]))
  let dtypes = [
    @streaming.StreamingSafetensorsDTypeSpec::bfloat16(),
    @streaming.StreamingSafetensorsDTypeSpec::float32(),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_fixed_width(
      "F16",
      bytes_per_element=2L,
    ),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("U8"),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("I8"),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_fixed_width(
      "I32",
      bytes_per_element=4L,
    ),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_fixed_width(
      "I64",
      bytes_per_element=8L,
    ),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3FN"),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E8M0"),
    @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E8M0FNU"),
  ]
  println(
    "checkpoint_startup: authenticating shards=\{inventory.shards().length()}",
  )
  let inspection = @streaming.inspect_shards(
    root,
    inventory.shards(),
    dtype_specs=dtypes,
    limits~,
  )
  println("checkpoint_startup: tensor binding and memory planning")
  let checkpoint = @materialize.PackedCheckpoint::new(inspection)
  let ceiling = budgets[0].max(budgets[1]) - reserve
  let decoder = @deepseek.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=spec.layer_count(),
    maximum_rows=rows,
    maximum_context=history,
    maximum_layer_state_bytes=ceiling,
    maximum_layer_workspace_bytes=ceiling,
    maximum_layer_expert_bytes=ceiling,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
  )
  defer decoder.close()
  let ingress = @text.DeepSeekTextIngress::new(spec, maximum_rows=rows)
  defer ingress.close()
  let egress = @text.DeepSeekTextIo::new(
    spec,
    maximum_rows=rows,
    maximum_output_rows=1,
    maximum_workspace_bytes=ceiling,
  )
  defer egress.close()
  let placement = decoder.partition(
    checkpoint,
    budgets,
    ingress_bytes=ingress.required_device_bytes(checkpoint),
    egress_bytes=egress.required_device_bytes(checkpoint),
    additional_stage_bytes=reserve,
  )
  guard placement.stages().length() == 2 else {
    fail("checkpoint does not require exactly two stages under these budgets")
  }
  let startup = {
    spec,
    checkpoint,
    limits,
    placement,
    rows,
    history,
    budgets,
    reserve,
  }
  // Recheck the executable owners, not only the planner's layer estimates.
  for ordinal in 0..<2 {
    let stage = stage_plan(startup, ordinal)
    defer stage.close()
    guard stage.required_device_bytes(checkpoint) <= budgets[ordinal] - reserve else {
      fail("executable stage exceeds its planned memory budget")
    }
  }
  startup
}

///|
fn worker_limits(startup : Startup) -> @protocol.WorkerProtocolLimits raise {
  @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=2,
    max_plan_tokens=startup.rows,
    max_plan_pages=1,
    max_capabilities=1,
    max_completion_slots=2,
    max_sequence_tokens=startup.history,
    max_token_id=startup.spec.vocabulary_size() - 1,
  )
}

///|
fn inference_limits(startup : Startup) -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=1048576,
    max_input_tokens=startup.history,
    max_new_tokens=startup.history,
    max_context_tokens=startup.history,
    max_token_id=startup.spec.vocabulary_size() - 1,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=300000UL,
    max_top_k=1,
    max_temperature=1.0,
  )
}

///|
fn new_frame(startup : Startup) -> @frame.SerialDecoderFrame raise {
  request_frame(startup.rows, startup.history, startup.spec.vocabulary_size())
}

///|
fn request_frame(
  rows : Int,
  history : Int,
  vocabulary : Int,
) -> @frame.SerialDecoderFrame raise {
  @frame.SerialDecoderFrame::new(
    maximum_rows=rows,
    maximum_context=history,
    vocabulary_size=vocabulary,
    model_generation=1UL,
    with_positions=true,
  )
}
