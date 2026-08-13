///|
fn wb_stream_limits(
  transfer_chunk_bytes? : Int = 17,
) -> DeviceWeightFileLimits raise {
  let safetensors = @safetensors.HeaderLimits::new(
    max_header_bytes=8192,
    max_tensor_count=32,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=4096L,
    max_materialized_tensor_bytes=4096L,
  )
  DeviceWeightFileLimits::new(
    max_file_bytes=16384L,
    transfer_chunk_bytes~,
    safetensors~,
  )
}

///|
fn wb_stream_entry(
  name : String,
  shape : String,
  start : Int,
  end : Int,
) -> String {
  "\"\{name}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{start},\{end}]}"
}

///|
fn wb_stream_file() -> Bytes {
  let entries : Array[String] = [
    wb_stream_entry("model.embed_tokens.weight", "[8,4]", 0, 64),
    wb_stream_entry("model.layers.0.input_layernorm.weight", "[4]", 64, 72),
    wb_stream_entry("model.layers.0.self_attn.q_proj.weight", "[4,4]", 72, 104),
    wb_stream_entry("model.layers.0.self_attn.k_proj.weight", "[2,4]", 104, 120),
    wb_stream_entry("model.layers.0.self_attn.v_proj.weight", "[2,4]", 120, 136),
    wb_stream_entry("model.layers.0.self_attn.o_proj.weight", "[4,4]", 136, 168),
    wb_stream_entry(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 168, 176,
    ),
    wb_stream_entry("model.layers.0.mlp.gate_proj.weight", "[6,4]", 176, 224),
    wb_stream_entry("model.layers.0.mlp.up_proj.weight", "[6,4]", 224, 272),
    wb_stream_entry("model.layers.0.mlp.down_proj.weight", "[4,6]", 272, 320),
    wb_stream_entry("model.norm.weight", "[4]", 320, 328),
    wb_stream_entry("lm_head.weight", "[8,4]", 328, 392),
  ]
  let header = @utf8.encode("{" + entries.join(",") + "}")
  let output : Array[Byte] = []
  for shift in 0..<8 {
    output.push((header.length().to_uint64() >> (shift * 8)).to_byte())
  }
  for byte in header {
    output.push(byte)
  }
  for index in 0..<392 {
    output.push((index % 251).to_byte())
  }
  Bytes::from_array(output)
}

///|
fn wb_stream_model(file : BytesView) -> @spec.LlamaModelMetadata raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=8,
    hidden_size=4,
    intermediate_size=6,
    layer_count=1,
    attention_head_count=2,
    kv_head_count=1,
    context_length=16,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let content = @spec.ContentDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(file)),
  )
  @spec.LlamaModelMetadata::from_verified_content(spec, content)
}

///|
fn wb_slice(file : Bytes, position : Int64, count : Int) -> Bytes {
  let start = position.to_int()
  file[start:start + count].to_owned()
}

///|
fn wb_stamp(file : BytesView, modified? : Int = 1) -> SourceStamp {
  {
    file_bytes: file.length().to_int64(),
    modified_seconds: modified.to_int64(),
    modified_nanoseconds: 0,
  }
}

///|
async test "streaming validation opens no destination for malformed or unapproved bytes" {
  let opens = Ref(0)
  try
    load_streaming_with_resource(
      wb_test_model(),
      { file_bytes: 5L, modified_seconds: 1L, modified_nanoseconds: 0 },
      wb_stream_limits(),
      64L,
      4096L,
      (_position, _count) => b"short",
      () => { file_bytes: 5L, modified_seconds: 1L, modified_nanoseconds: 0 },
      _arena_bytes => {
        opens.val = opens.val + 1
        1
      },
      (_resource, _bytes, _destination, _count) => (),
      _resource => (),
    )
  catch {
    FileTruncated => ()
    _ => fail("unexpected malformed stream error")
  } noraise {
    _ => fail("expected malformed stream failure")
  }
  assert_eq(opens.val, 0)
  let file = wb_stream_file()
  try
    load_streaming_with_resource(
      wb_test_model(),
      wb_stamp(file),
      wb_stream_limits(),
      64L,
      4096L,
      (position, count) => wb_slice(file, position, count),
      () => wb_stamp(file),
      _arena_bytes => {
        opens.val = opens.val + 1
        1
      },
      (_resource, _bytes, _destination, _count) => (),
      _resource => (),
    )
  catch {
    ContentDigestMismatch => ()
    _ => fail("unexpected digest error")
  } noraise {
    _ => fail("expected digest mismatch")
  }
  assert_eq(opens.val, 0)
}

///|
async test "streaming copies are chunk bounded and target final layout regions" {
  let file = wb_stream_file()
  let opens = Ref(0)
  let closes = Ref(0)
  let copies = Ref(0)
  let maximum_copy = Ref(0L)
  let opened_arena_bytes = Ref(0L)
  let invalid_copy = Ref(false)
  let arena = FixedArray::make(768, b'\x00')
  let outcome = load_streaming_with_resource(
    wb_stream_model(file),
    wb_stamp(file),
    wb_stream_limits(transfer_chunk_bytes=17),
    64L,
    4096L,
    (position, count) => wb_slice(file, position, count),
    () => wb_stamp(file),
    arena_bytes => {
      opens.val = opens.val + 1
      opened_arena_bytes.val = arena_bytes
      9
    },
    (_resource, bytes, destination, count) => {
      copies.val = copies.val + 1
      if bytes.length().to_int64() != count || destination < 0L {
        invalid_copy.val = true
      }
      if count > maximum_copy.val {
        maximum_copy.val = count
      }
      arena.blit_from_bytes(destination.to_int(), bytes, 0, count.to_int())
    },
    _resource => closes.val = closes.val + 1,
  )
  guard outcome is ResourceReady(loaded) else {
    fail("successful stream retained cleanup authority")
  }
  assert_eq(loaded.resource, 9)
  assert_eq(loaded.layout.arena_bytes(), 768L)
  assert_eq(opens.val, 1)
  assert_eq(opened_arena_bytes.val, 768L)
  assert_eq(closes.val, 0)
  assert_false(invalid_copy.val)
  assert_true(copies.val > loaded.layout.length())
  assert_eq(maximum_copy.val, 17L)
  let source_starts = [0, 64, 72, 104, 120, 136, 168, 176, 224, 272, 320, 328]
  let header_bytes = @safetensors_reader.declared_header_bytes(
    file[:8],
    limits=wb_stream_limits().safetensors(),
  )
  for reference in 0..<loaded.layout.length() {
    let tensor_reference = try! @plan.TensorRef::new(reference)
    guard loaded.layout.region(tensor_reference) is Some(region) else {
      fail("expected loaded region")
    }
    let source_start = 8 + header_bytes + source_starts[reference]
    let byte_count = region.byte_length().to_int()
    assert_eq(
      Bytes::from_array(
        arena[region.offset_bytes().to_int():region.offset_bytes().to_int() +
        byte_count],
      ),
      file[source_start:source_start + byte_count].to_owned(),
    )
  }
}

///|
async test "same-size mutation of copied bytes rejects and closes exactly once" {
  let file = wb_stream_file()
  let changed_array = file.to_array()
  changed_array[changed_array.length() - 1] = (changed_array[changed_array.length() -
  1].to_int() ^
  1).to_byte()
  let changed = Bytes::from_array(changed_array)
  let opens = Ref(0)
  let closes = Ref(0)
  let copies = Ref(0)
  try
    load_streaming_with_resource(
      wb_stream_model(file),
      wb_stamp(file),
      wb_stream_limits(transfer_chunk_bytes=19),
      64L,
      4096L,
      (position, count) => {
        let source = if opens.val == 0 { file } else { changed }
        wb_slice(source, position, count)
      },
      () => wb_stamp(file),
      _arena_bytes => {
        opens.val = opens.val + 1
        3
      },
      (_resource, _bytes, _destination, _count) => copies.val = copies.val + 1,
      _resource => closes.val = closes.val + 1,
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected mutation error")
  } noraise {
    _ => fail("expected mutation rejection")
  }
  assert_eq(opens.val, 1)
  assert_true(copies.val > 0)
  assert_eq(closes.val, 1)
}

///|
async test "source stamp changes reject before open and clean up after open" {
  let file = wb_stream_file()
  let opens = Ref(0)
  let stamp_reads = Ref(0)
  try
    load_streaming_with_resource(
      wb_stream_model(file),
      wb_stamp(file),
      wb_stream_limits(),
      64L,
      4096L,
      (position, count) => wb_slice(file, position, count),
      () => {
        stamp_reads.val = stamp_reads.val + 1
        wb_stamp(file, modified=2)
      },
      _arena_bytes => {
        opens.val = opens.val + 1
        1
      },
      (_resource, _bytes, _destination, _count) => (),
      _resource => (),
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected pre-open stamp error")
  } noraise {
    _ => fail("expected pre-open stamp change")
  }
  assert_eq(stamp_reads.val, 1)
  assert_eq(opens.val, 0)

  let closes = Ref(0)
  stamp_reads.val = 0
  try
    load_streaming_with_resource(
      wb_stream_model(file),
      wb_stamp(file),
      wb_stream_limits(),
      64L,
      4096L,
      (position, count) => wb_slice(file, position, count),
      () => {
        stamp_reads.val = stamp_reads.val + 1
        if stamp_reads.val == 1 {
          wb_stamp(file)
        } else {
          wb_stamp(file, modified=2)
        }
      },
      _arena_bytes => {
        opens.val = opens.val + 1
        2
      },
      (_resource, _bytes, _destination, _count) => (),
      _resource => closes.val = closes.val + 1,
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected post-open stamp error")
  } noraise {
    _ => fail("expected post-open stamp change")
  }
  assert_eq(stamp_reads.val, 2)
  assert_eq(opens.val, 1)
  assert_eq(closes.val, 1)
}

///|
async test "streaming transfer failure closes the destination exactly once" {
  let file = wb_stream_file()
  let copies = Ref(0)
  let closes = Ref(0)
  try
    load_streaming_with_resource(
      wb_stream_model(file),
      wb_stamp(file),
      wb_stream_limits(),
      64L,
      4096L,
      (position, count) => wb_slice(file, position, count),
      () => wb_stamp(file),
      _arena_bytes => 5,
      (_resource, _bytes, _destination, _count) => {
        copies.val = copies.val + 1
        if copies.val == 4 {
          raise DriverFailure(CopyToDevice)
        }
      },
      _resource => closes.val = closes.val + 1,
    )
  catch {
    DeviceFailed(DriverFailure(CopyToDevice)) => ()
    _ => fail("unexpected transfer failure")
  } noraise {
    _ => fail("expected transfer failure")
  }
  assert_eq(copies.val, 4)
  assert_eq(closes.val, 1)
}

///|
async test "streaming double failure retains retryable cleanup authority" {
  let file = wb_stream_file()
  let copies = Ref(0)
  let closes = Ref(0)
  let outcome = load_streaming_with_resource(
    wb_stream_model(file),
    wb_stamp(file),
    wb_stream_limits(),
    64L,
    4096L,
    (position, count) => wb_slice(file, position, count),
    () => wb_stamp(file),
    _arena_bytes => 5,
    (_resource, _bytes, _destination, _count) => {
      copies.val += 1
      if copies.val == 4 {
        raise DriverFailure(CopyToDevice)
      }
    },
    _resource => {
      closes.val += 1
      if closes.val < 3 {
        raise DriverFailure(AllocationClose)
      }
    },
  )
  guard outcome is ResourceCleanupRequired(failed) else {
    fail("double failure discarded cleanup authority")
  }
  assert_eq(failed.cause, DeviceFailed(DriverFailure(CopyToDevice)))
  assert_eq(failed.cleanup_failure, DriverFailure(AllocationClose))
  let open = Ref(true)
  let cleanup_failure = Ref(failed.cleanup_failure)
  let retry_close = (close : (Int) -> Unit raise @device.DeviceError) => {
    if !open.val {
      return
    }
    close(failed.resource) catch {
      error => {
        cleanup_failure.val = error
        raise CleanupFailed(stage=Transfer, error)
      }
    }
    open.val = false
  }
  try
    retry_close(_resource => {
      closes.val += 1
      if closes.val < 3 {
        raise DriverFailure(AllocationClose)
      }
    })
  catch {
    CleanupFailed(stage=Transfer, DriverFailure(AllocationClose)) => ()
    _ => fail("unexpected retry cleanup failure")
  } noraise {
    _ => fail("first retry unexpectedly closed")
  }
  assert_true(open.val)
  retry_close(_resource => closes.val += 1)
  assert_false(open.val)
  retry_close(_resource => closes.val += 1)
  assert_eq(closes.val, 3)
}
