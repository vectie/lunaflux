///|
async test "GLM full-sized recurrent checkpoint binds all fifteen physical weight planes" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-glm-recurrent"))
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let shapes = [
      ("q_proj.weight", "BF16", "8192,4096", 67108864L),
      ("k_proj.weight", "BF16", "8192,4096", 67108864L),
      ("v_proj.weight", "BF16", "8192,4096", 67108864L),
      ("f_a_proj.weight", "BF16", "128,4096", 1048576L),
      ("f_b_proj.weight", "BF16", "8192,128", 2097152L),
      ("b_proj.weight", "BF16", "64,4096", 524288L),
      ("dt_bias", "F32", "8192", 32768L),
      ("A_log", "F32", "64", 256L),
      ("g_a_proj.weight", "BF16", "128,4096", 1048576L),
      ("g_b_proj.weight", "BF16", "8192,128", 2097152L),
      ("o_norm.weight", "BF16", "128", 256L),
      ("o_proj.weight", "BF16", "4096,8192", 67108864L),
      ("q_conv1d.weight", "BF16", "8192,1,4", 65536L),
      ("k_conv1d.weight", "BF16", "8192,1,4", 65536L),
      ("v_conv1d.weight", "BF16", "8192,1,4", 65536L),
    ]
    let entries : Array[String] = []
    let mut offset = 0L
    for (name, dtype, shape, count) in shapes {
      entries.push(
        "\"model.language_model.layers.0.self_attn.\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset + count}]}",
      )
      offset += count
    }
    assert_eq(offset, 275481088L)
    let header = @utf8.encode("{" + entries.join(",") + "}")
    let prefix : Array[Byte] = []
    for i in 0..<8 {
      prefix.push((header.length().to_uint64() >> (8 * i)).to_byte())
    }
    for byte in header {
      prefix.push(byte)
    }
    let prefix = Bytes::from_array(prefix)
    // Sparse fixture: bounded host storage, full official tensor sizes.
    let file = @fs.open(
      directory + "/weights",
      mode=WriteOnly,
      create_mode=CreateNew,
    )
    file.write_at(prefix, position=0L)
    file.write_at(b"\x00", position=prefix.length().to_int64() + offset - 1L)
    file.close()
    let hasher = @crypto.SHA256::new()
    hasher.update(prefix)
    let zeros = Bytes::from_array(FixedArray::make(65536, b'\x00'))
    for remaining = offset; remaining > 0L; {
      let count = remaining.min(65536L).to_int()
      hasher.update(zeros[:count])
      continue remaining - count.to_int64()
    }
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=300000000L,
      maximum_header_bytes=8192,
      maximum_tensors_per_shard=15,
      maximum_total_tensors=15,
      maximum_rank=3,
      maximum_name_bytes=256,
      maximum_dtype_specs=2,
      authentication_chunk_bytes=65536,
      maximum_copy_requests=15,
      maximum_copy_bytes=275481088L,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              @crypto.bytes_to_hex_string(hasher.finalize()),
            ),
          ),
        ],
        dtype_specs=[
          @streaming.StreamingSafetensorsDTypeSpec::bfloat16(),
          @streaming.StreamingSafetensorsDTypeSpec::float32(),
        ],
        limits~,
      ),
    )
    let weights = @glm53_packed_expert_upload.bind_recurrent_weights(
      checkpoint,
      @glm53_spec.Glm53FlashSpec::official(),
      layer_index=0,
    )
    assert_eq(weights.length(), shapes.length())
    for i, weight in weights {
      let (_, _, _, bytes) = shapes[i]
      assert_eq(weight.buffer_bytes(), bytes)
    }
    let layer = @glm53_packed_expert_upload.GlmRecurrentLayer::new(
      @glm53_spec.Glm53FlashSpec::official(),
      layer_index=0,
      maximum_rows=32,
      maximum_sequences=16,
      cache_slots=32,
      maximum_state_bytes=134217728L,
    )
    assert_eq(layer.workspace_bytes(), 145252352L)
    assert_true(
      @utf8.decode(layer.source_bytes()).contains("glm_recurrent_0_15("),
    )
    let baseline = live_resources()
    let context = @device.open_context(0)
    defer (context.close() catch { _ => () })
    let stream = context.create_stream()
    defer (stream.close() catch { _ => () })
    let aot = context.load_module(b"fake\x00")
    defer (aot.close() catch { _ => () })
    let operand = context.allocate(262144L)
    defer (operand.close() catch { _ => () })
    defer (layer.close() catch { _ => () })
    layer.prepare(
      context,
      aot,
      stream,
      checkpoint,
      root,
      maximum_device_bytes=420733440L,
      chunk_bytes=65533,
      limits~,
      counts=operand,
      sequence_offsets=operand,
      sequence_slots=operand,
      reset_flags=operand,
      hidden=operand,
      output=operand,
    )
    assert_eq(layer.device_bytes(), 420733440L)
    let before = enqueues()
    let waits = blocking_syncs()
    begin_allocations()
    for _ in 0..<32 {
      layer.submit(live_rows=32)
      while !layer.poll() {
        ()
      }
    }
    assert_eq(end_allocations(), 0UL)
    assert_eq(enqueues() - before, 512UL)
    assert_eq(blocking_syncs(), waits)
    // Active cancellation closes the executor before its borrowed weights.
    layer.submit(live_rows=1)
    layer.close()
    layer.close()
    operand.close()
    aot.close()
    stream.close()
    context.close()
    assert_eq(live_resources(), baseline)
  }
}
