///|
// Offline matched-layer replay. The ordinary service graph is unchanged.
// Only one complete production decoder layer and its state reside on device.
fn matched_layer_shape(layer : Int, rows : Int, input_bytes : Int) -> Int raise {
  guard layer >= 0 &&
    layer < 43 &&
    rows > 0 &&
    rows <= 144 &&
    input_bytes > 0 &&
    input_bytes % 32768 == 0 else {
    fail("matched replay requires a base layer and complete residual streams")
  }
  let count = input_bytes / 32768
  guard count <= 256 else { fail("matched replay exceeds retained context") }
  count
}

///|
test "matched layer replay retains every stream and partial final submission" {
  assert_eq(matched_layer_shape(40, 6, 144 * 32768), 144)
  assert_eq(matched_layer_shape(42, 7, 144 * 32768), 144)
  assert_eq(matched_layer_shape(0, 1, 144 * 32768), 144)
  for
    (layer, rows, bytes) in [
      (-1, 6, 32768),
      (43, 6, 32768),
      (40, 0, 32768),
      (40, 6, 32767),
    ] {
    try matched_layer_shape(layer, rows, bytes) catch {
      _ => ()
    } noraise {
      _ => fail("invalid diagnostic shape accepted")
    }
  }
}

///|
async fn matched_upload(pool : @device.TransferPool, bytes : Bytes) -> Unit {
  let payload = FixedArray::makei(bytes.length(), i => bytes[i])
  pool.write_host(
    payload,
    host_offset=0L,
    source_offset=0,
    bytes=bytes.length(),
  )
  pool.enqueue(
    lane=0,
    host_offset=0L,
    device_offset=0L,
    bytes=bytes.length().to_int64(),
    to_device=true,
  )
  pool.record(0)
  while !pool.poll(0) {
    @async.pause()
  }
  pool.reset(0)
}

///|
fn matched_ints(values : ArrayView[Int]) -> Bytes {
  Bytes::from_array(
    [
      for i in 0..<(values.length() * 4) => {
        ((values[i / 4].reinterpret_as_uint() >> (i % 4 * 8)) & 255U).to_byte()
      }
    ],
  )
}

///|
async fn replay_matched_layer(
  arguments : ArrayView[String],
  attention_accumulation? : @precision.AttentionAccumulationLaw = OrderedFloat32,
  rotary_product? : @precision.RotaryProductLaw = SeparateFloat32Products,
  output_accumulation? : @precision.DotReductionLaw = OrderedFloat32,
) -> Unit {
  let model_root = arguments[0]
  let layer = @string.parse_int(arguments[2])
  let rows = @string.parse_int(arguments[3])
  let mode = arguments[4]
  let input = @fs.read_file(arguments[7]).binary()
  let count = matched_layer_shape(layer, rows, input.length())
  guard mode is ("export" | "run") else { fail("export|run required") }
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
  defer root.close()
  let spec = inspect_config(root)
  let ids = if arguments.length() == 10 {
    let bytes = @fs.read_file(arguments[9]).binary()
    guard bytes.length() == count * 4 else {
      fail("literal token IDs must match the complete predecessor rows")
    }
    Some(bytes)
  } else {
    guard layer >= spec.hash_layer_count() else {
      fail("token-hash replay requires literal original I32 token IDs")
    }
    None
  }
  let ceiling = 8589934592L
  let decoder = @deepseek.DeepSeekDecoderSlice::new(
    spec,
    first_layer=layer,
    end_layer=layer + 1,
    maximum_rows=rows,
    maximum_context=256,
    maximum_layer_state_bytes=ceiling,
    maximum_layer_workspace_bytes=ceiling,
    maximum_layer_expert_bytes=ceiling,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    attention_accumulation~,
    rotary_product~,
    output_accumulation~,
  )
  defer decoder.close()
  @fs.mkdir(arguments[8])
  if mode == "export" {
    @fs.write_file(
      arguments[8] + "/layer.cu",
      decoder.source_bytes(),
      create_mode=CreateNew,
    )
    println(
      "matched_layer_export layer=\{layer} physical_rows=\{rows} input_rows=\{count} accumulation=\{Repr(attention_accumulation)} rotary_product=\{Repr(rotary_product)} output_accumulation=\{Repr(output_accumulation)}",
    )
    return
  }
  let inventory = @inventory.CheckpointInventory::parse(
    @utf8.decode(read_inventory(root, arguments[1])),
    maximum_bytes=1048576,
    maximum_shards=512,
  )
  let limits = checkpoint_limits(ceiling)
  let checkpoint = @materialize.PackedCheckpoint::new(
    @streaming.inspect_shards(
      root,
      inventory.shards(),
      dtype_specs=[
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
      ],
      limits~,
    ),
  )
  guard decoder.required_device_bytes(checkpoint) + rows.to_int64() * 65536L <
    ceiling else {
    fail("matched single layer exceeds device budget")
  }
  let context = @device.open_context(0)
  defer close_rank_context(context, "matched-layer")
  let stream = context.create_stream()
  defer stream.close()
  let module = load_module(context, arguments[5], arguments[6], 1073741824L)
  defer module.close()
  let tables = @expert.bind_token_hash_tables(
    checkpoint,
    spec,
    first_layer=layer,
    end_layer=layer + 1,
  )
  let table_owner = @upload.CheckpointDeviceIndexTables::new()
  defer table_owner.close()
  if !tables.is_empty() {
    table_owner.upload(
      context,
      checkpoint,
      root,
      tables,
      minimum_value=0,
      exclusive_maximum_value=spec.routed_expert_count(),
      maximum_device_bytes=tables.fold(init=0L, (total, table) => {
        total + table.i32_bytes()
      }),
      maximum_host_bytes=16777216L,
      chunk_bytes=1048576,
      limits~,
    )
  }
  let token_ids = context.allocate(rows.to_int64() * 4L)
  defer token_ids.close()
  let token_pool = context.prepare_transfer_pool(
    token_ids,
    host_bytes=rows.to_int64() * 4L,
    lanes=1,
    max_regions_per_lane=1,
  )
  defer {
    token_pool.drain(0)
    token_pool.close()
  }
  let counts = context.allocate(20L)
  defer counts.close()
  let positions = context.allocate(rows.to_int64() * 4L)
  defer positions.close()
  let reset = context.allocate(4L)
  defer reset.close()
  let port_bytes = rows * 32768
  let source = context.allocate(port_bytes.to_int64())
  defer source.close()
  let destination = context.allocate(port_bytes.to_int64())
  defer destination.close()
  let counts_pool = context.prepare_transfer_pool(
    counts,
    host_bytes=20L,
    lanes=1,
    max_regions_per_lane=1,
  )
  defer {
    counts_pool.drain(0)
    counts_pool.close()
  }
  let positions_pool = context.prepare_transfer_pool(
    positions,
    host_bytes=rows.to_int64() * 4L,
    lanes=1,
    max_regions_per_lane=1,
  )
  defer {
    positions_pool.drain(0)
    positions_pool.close()
  }
  let reset_pool = context.prepare_transfer_pool(
    reset,
    host_bytes=4L,
    lanes=1,
    max_regions_per_lane=1,
  )
  defer {
    reset_pool.drain(0)
    reset_pool.close()
  }
  let input_pool = context.prepare_transfer_pool(
    source,
    host_bytes=port_bytes.to_int64(),
    lanes=1,
    max_regions_per_lane=1,
  )
  defer {
    input_pool.drain(0)
    input_pool.close()
  }
  let output_pool = context.prepare_transfer_pool(
    destination,
    host_bytes=port_bytes.to_int64(),
    lanes=1,
    max_regions_per_lane=1,
  )
  defer {
    output_pool.drain(0)
    output_pool.close()
  }
  // Retire the slice before caller ports/module/context. It owns no oracle math.
  defer decoder.close()
  decoder.prepare(
    context,
    module,
    stream,
    checkpoint,
    root,
    counts~,
    positions~,
    reset_flag=reset,
    input=source,
    output=destination,
    token_ids?=if tables.is_empty() { None } else { Some(token_ids) },
    token_tables=if tables.is_empty() { [] } else { table_owner.allocations() },
    maximum_device_bytes=ceiling,
    chunk_bytes=1048576,
    limits~,
  )
  let captured = Array::new(capacity=input.length())
  let boundaries : Map[String, Array[Byte]] = Map([])
  let readback = FixedArray::make(port_bytes, b'\x00')
  for start = 0; start < count; start = start + rows {
    let live = rows.min(count - start)
    let padded = input[start * 32768:(start + live) * 32768].to_owned() +
      Bytes::make((rows - live) * 32768, 0)
    matched_upload(input_pool, padded)
    if ids is Some(bytes) {
      matched_upload(
        token_pool,
        bytes[start * 4:(start + live) * 4].to_owned() +
        Bytes::make((rows - live) * 4, 0),
      )
    }
    matched_upload(counts_pool, matched_ints([0, 0, 1, live, 0]))
    matched_upload(
      positions_pool,
      matched_ints(Array::makei(rows, i => start + i)),
    )
    matched_upload(reset_pool, matched_ints([if start == 0 { 1 } else { 0 }]))
    decoder.submit(live_rows=live)
    while !decoder.poll() {
      @async.pause()
    }
    decoder.visit_retired_boundaries((name, allocation, width) => {
      let values = allocation.copy_to_host(
        0L,
        live.to_int64() * width.to_int64() * 2L,
      )
      let target = boundaries.get_or_init(name, () => [])
      for byte in values {
        target.push(byte)
      }
    })
    output_pool.enqueue(
      lane=0,
      host_offset=0L,
      device_offset=0L,
      bytes=port_bytes.to_int64(),
      to_device=false,
    )
    output_pool.record(0)
    while !output_pool.poll(0) {
      @async.pause()
    }
    output_pool.read_host(
      readback,
      host_offset=0L,
      destination_offset=0,
      bytes=port_bytes,
    )
    output_pool.reset(0)
    captured.append(readback[:live * 32768])
  }
  @fs.write_file(
    arguments[8] + "/hidden.bf16",
    Bytes::from_array(captured),
    create_mode=CreateNew,
  )
  for name, bytes in boundaries {
    @fs.write_file(
      arguments[8] + "/" + name + ".bf16",
      Bytes::from_array(bytes),
      create_mode=CreateNew,
    )
  }
  println(
    "matched_layer_complete layer=\{layer} input_rows=\{count} physical_rows=\{rows} shared_decoder_slice=true accumulation=\{Repr(attention_accumulation)} rotary_product=\{Repr(rotary_product)} output_accumulation=\{Repr(output_accumulation)} production_default_policy=\{attention_accumulation == @precision.AttentionAccumulationLaw::OrderedFloat32 && rotary_product == @precision.RotaryProductLaw::SeparateFloat32Products && output_accumulation == @precision.DotReductionLaw::OrderedFloat32} full_model=false",
  )
}
