///|
/// Checkpoint-backed Qwen3-VL hidden_states[50], not an LLM language-head result.
/// The complete component stays bounded and streamed; no host weight arena.
async fn run_text(
  root_name : String,
  root : @approved_fs.ApprovedRoot,
  metadata : @h3.MiniMaxH3ModelMetadata,
  model : @joint.JointDiffusionPlan,
  args : ArrayView[String],
  capture_layers? : Bool = false,
  capture_operators? : Bool = false,
  matched_layers? : String,
) -> Unit {
  let config = @text.parse_config(
    read(root, "text_encoder/config.json", 1048576L),
    max_bytes=1048576,
  )
  let rows = @string.parse_int(args[1])
  guard rows > 0 && rows <= config.max_positions else {
    fail("text rows exceed encoder capacity")
  }
  let ids_bytes = read(root, args[2], rows.to_int64() * 4L)
  let ids = text_input_ids(ids_bytes, rows)
  let plan = config.plan(ids[:])
  let source = @text_execution.build_source(plan, prefix="h3_text")
  if capture_layers || capture_operators {
    if capture_operators {
      validate_capture_bytes(
        source.first_layer_observation_bytes(),
        @string.parse_int64(args[7]),
      )
    } else {
      validate_text_capture_budget(
        plan.output_bytes(),
        @string.parse_int64(args[7]),
        matched_inputs=matched_layers is Some(_),
      )
    }
    @fs.mkdir(args[4])
  }
  let original_inputs : Array[Bytes] = []
  if matched_layers is Some(directory) {
    for index in 0..<50 {
      let bytes = @fs.read_file(directory + "/hidden-\{index}.bf16").binary()
      guard bytes.length().to_int64() == plan.output_bytes() else {
        fail("matched original hidden-state extent differs")
      }
      original_inputs.push(bytes)
    }
  }
  let device_limit = @string.parse_int64(args[6])
  let component_root = @approved_fs.ApprovedRoot::open_absolute(
    root_name + "/text_encoder",
  )
  defer component_root.close()
  let inventory = @inventory.CheckpointInventory::parse(
    @utf8.decode(read(root, args[0], 1048576L)),
    maximum_bytes=1048576,
    maximum_shards=64,
  )
  let checkpoint = @checkpoint.ComponentCheckpoint::inspect(
    component_root,
    metadata,
    model,
    TextConditioner,
    inventory,
    maximum_component_bytes=@string.parse_int64(args[5]),
    maximum_device_bytes=device_limit,
  )
  let weights_bytes = checkpoint.required_device_bytes()
  let module_reserve = 67108864L
  let regions = plan.workspace_regions()
  let end = regions[regions.length() - 1]
  let workspace_bytes = end.0 + end.1
  let table_bytes = rows.to_int64() * 128L * 2L * 2L
  let request_bytes = workspace_bytes + 16L + rows.to_int64() * 4L + table_bytes
  guard device_limit > weights_bytes &&
    device_limit - weights_bytes > module_reserve &&
    request_bytes <= device_limit - weights_bytes - module_reserve else {
    fail("text encoder aggregate exceeds device budget")
  }
  println(
    "text_weights=\{weights_bytes} text_request_bytes=\{request_bytes} text_rows=\{rows} device_limit=\{device_limit}",
  )
  let aot = @approved_fs.ApprovedRoot::open_absolute(args[3])
  defer aot.close()
  let module_bytes = read(aot, "text.cubin", module_reserve)
  let context = @device.open_context(0)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let aot_module = context.load_module(module_bytes)
  defer aot_module.close()
  let owner = @startup.WeightStartup::new(max_device_bytes=weights_bytes)
  defer owner.close()
  checkpoint.prepare(owner, context, component_root, chunk_bytes=1048576)
  let lease = owner.acquire()
  defer lease.release()
  let request = @text_execution.EncoderRequest::new(source, config)
  defer request.close()
  lease.with_encoder(0, weights => {
    if capture_operators {
      request.prepare_operator_observed(
        context,
        stream,
        aot_module,
        weights,
        ids[:],
        max_device_bytes=request_bytes,
        max_table_bytes=table_bytes,
      )
    } else if capture_layers {
      request.prepare_observed(
        context,
        stream,
        aot_module,
        weights,
        ids[:],
        max_device_bytes=request_bytes,
        max_table_bytes=table_bytes,
      )
    } else {
      request.prepare(
        context,
        stream,
        aot_module,
        weights,
        ids[:],
        max_device_bytes=request_bytes,
        max_table_bytes=table_bytes,
      )
    }
  })
  if capture_operators {
    let tensors : Array[(String, Bytes)] = []
    request.execute_operator_observed(on_tensor=(name, bytes) => {
      tensors.push((name, bytes))
    })
    for (name, bytes) in tensors {
      let (nonfinite, maximum) = text_hidden_statistics(bytes)
      @fs.write_file(
        args[4] + "/" + name + ".bf16",
        bytes,
        create_mode=CreateNew,
      )
      println(
        "text_operator_complete name=\{name} bytes=\{bytes.length()} nonfinite=\{nonfinite} maximum_absolute=\{maximum}",
      )
      guard nonfinite == 0 else {
        fail("nonfinite operator output; data preserved")
      }
    }
    println(
      "text_operator_prefix_complete layers=1 tensors=\{tensors.length()} rows=\{rows}",
    )
    return
  }
  let captures : Array[(Int, Bytes)] = []
  let hidden = if capture_layers {
    request.execute_observed(
      on_hidden_state=(index, bytes) => captures.push((index, bytes)),
      original_hidden_state?=matched_layers.map(_ => {
        index => original_inputs[index]
      }),
    )
    for (index, bytes) in captures {
      @fs.write_file(
        args[4] + "/hidden-\{index}.bf16",
        bytes,
        create_mode=CreateNew,
      )
    }
    captures[captures.length() - 1].1
  } else {
    request.execute()
    request.with_output((output, offset) => {
      output.copy_to_host(offset, plan.output_bytes())
    })
  }
  guard hidden.length().to_int64() == plan.output_bytes() else {
    fail("incomplete encoder hidden-state output")
  }
  if !capture_layers {
    @fs.write_file(args[4], hidden, create_mode=CreateNew, sync=Full)
  }
  let (nonfinite, maximum) = text_hidden_statistics(hidden)
  println(
    "text_encoder_complete hidden_state=50 rows=\{rows} width=5120 bytes=\{hidden.length()} nonfinite=\{nonfinite} maximum_absolute=\{maximum} matched_layer_inputs=\{matched_layers is Some(_)}",
  )
  guard nonfinite == 0 else {
    fail("nonfinite hidden-state output; data preserved")
  }
}

///|
fn validate_text_capture_budget(
  output_bytes : Int64,
  maximum : Int64,
  matched_inputs? : Bool = false,
) -> Unit raise {
  let states = if matched_inputs { 101L } else { 51L }
  guard output_bytes > 0L && output_bytes <= 2147483647L / states else {
    fail("invalid encoder capture extent")
  }
  validate_capture_bytes(output_bytes * states, maximum)
}

///|
fn validate_capture_bytes(required : Int64, maximum : Int64) -> Unit raise {
  guard required > 0L &&
    maximum > 0L &&
    maximum <= 2147483647L &&
    required <= maximum else {
    fail("tensor snapshots exceed host capture budget")
  }
}

///|
fn text_input_ids(bytes : Bytes, rows : Int) -> Array[Int] raise {
  guard rows > 0 && rows.to_int64() * 4L == bytes.length().to_int64() else {
    fail("wrong text token-ID byte count")
  }
  Array::makei(rows, i => {
    let offset = i * 4
    bytes[offset].to_int() |
    (bytes[offset + 1].to_int() << 8) |
    (bytes[offset + 2].to_int() << 16) |
    (bytes[offset + 3].to_int() << 24)
  })
}

///|
fn text_hidden_statistics(bytes : Bytes) -> (Int, Float) raise {
  guard !bytes.is_empty() && bytes.length() % 2 == 0 else {
    fail("empty or unaligned BF16 hidden output")
  }
  for i in 0..<(bytes.length() / 2); nonfinite = 0, maximum = 0.0F {
    let bits = bytes[i * 2].to_uint() | (bytes[i * 2 + 1].to_uint() << 8)
    if (bits & 0x7f80U) == 0x7f80U {
      continue nonfinite + 1, maximum
    }
    continue nonfinite,
      Float::max(maximum, Float::reinterpret_from_uint(bits << 16).abs())
  } nobreak {
    (nonfinite, maximum)
  }
}
