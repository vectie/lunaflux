///|
async fn exercise_glm_checkpoint_owner(
  with_dense_block~ : Bool,
  with_moe_block? : Bool = false,
) -> Unit {
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
      ("hc_attn_fn", "BF16", "24,16384", 786432L),
      ("hc_attn_base", "F32", "24", 96L),
      ("hc_attn_scale", "F32", "3", 12L),
      ("input_layernorm.weight", "BF16", "4096", 8192L),
    ]
    if with_dense_block {
      shapes.append([
        ("mlp.gate_proj.weight", "BF16", "12288,4096", 100663296L),
        ("mlp.up_proj.weight", "BF16", "12288,4096", 100663296L),
        ("mlp.down_proj.weight", "BF16", "4096,12288", 100663296L),
        ("hc_ffn_fn", "BF16", "24,16384", 786432L),
        ("hc_ffn_base", "F32", "24", 96L),
        ("hc_ffn_scale", "F32", "3", 12L),
        ("post_attention_layernorm.weight", "BF16", "4096", 8192L),
      ])
    }
    if with_moe_block {
      shapes.append([
        ("mlp.gate.weight", "BF16", "288,4096", 2359296L),
        ("mlp.gate.e_score_correction_bias", "F32", "288", 1152L),
        ("hc_ffn_fn", "BF16", "24,16384", 786432L),
        ("hc_ffn_base", "F32", "24", 96L),
        ("hc_ffn_scale", "F32", "3", 12L),
        ("post_attention_layernorm.weight", "BF16", "4096", 8192L),
      ])
      for
        (projection, shape) in [
          ("gate_proj", "2048,4096"),
          ("up_proj", "2048,4096"),
          ("down_proj", "4096,2048"),
        ] {
        shapes.push(
          (
            "mlp.shared_experts." + projection + ".weight",
            "BF16",
            shape,
            16777216L,
          ),
        )
      }
      for expert in 0..<288 {
        for
          (projection, payload_shape, scale_shape) in [
            ("gate_proj", "2048,2048", "2048,256"),
            ("up_proj", "2048,2048", "2048,256"),
            ("down_proj", "4096,1024", "4096,128"),
          ] {
          let name = "mlp.experts.\{expert}.\{projection}.weight"
          shapes.append([
            (name, "U8", payload_shape, 4194304L),
            (name + "_scale", "F8_E4M3", scale_shape, 524288L),
            (name + "_scale_2", "F32", "", 4L),
          ])
        }
      }
    }
    let layer_index = if with_moe_block { 4 } else { 0 }
    let weight_bytes = if with_moe_block {
      4406629592L
    } else if with_dense_block {
      579060440L
    } else {
      276275820L
    }
    let entries : Array[String] = []
    let mut offset = 0L
    for index, (name, dtype, shape, count) in shapes {
      let parent = if index < 15 { "self_attn." } else { "" }
      entries.push(
        "\"model.language_model.layers.\{layer_index}.\{parent}\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset + count}]}",
      )
      offset += count
    }
    assert_eq(offset, weight_bytes)
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
      maximum_file_bytes=5000000000L,
      maximum_header_bytes=1048576,
      maximum_tensors_per_shard=shapes.length(),
      maximum_total_tensors=shapes.length(),
      maximum_rank=3,
      maximum_name_bytes=256,
      maximum_dtype_specs=4,
      authentication_chunk_bytes=65536,
      maximum_copy_requests=shapes.length(),
      maximum_copy_bytes=weight_bytes,
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
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("U8"),
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
        ],
        limits~,
      ),
    )
    let weights = @glm53_packed_expert_upload.bind_recurrent_weights(
      checkpoint,
      @glm53_spec.Glm53FlashSpec::official(),
      layer_index~,
    )
    assert_eq(weights.length(), 15)
    for i, weight in weights {
      let (_, _, _, bytes) = shapes[i]
      assert_eq(weight.buffer_bytes(), bytes)
    }
    let layer = if with_moe_block {
      @glm53_packed_expert_upload.GlmRecurrentLayer::with_moe_block(
        @glm53_spec.Glm53FlashSpec::official(),
        layer_index~,
        maximum_rows=32,
        maximum_sequences=16,
        cache_slots=32,
        maximum_state_bytes=134217728L,
        maximum_expert_bytes=5000000000L,
        accuracy=@precision.Accuracy::new(absolute=0.125, relative=0.125),
      )
    } else if with_dense_block {
      @glm53_packed_expert_upload.GlmRecurrentLayer::with_dense_block(
        @glm53_spec.Glm53FlashSpec::official(),
        layer_index=0,
        maximum_rows=32,
        maximum_sequences=16,
        cache_slots=32,
        maximum_state_bytes=134217728L,
      )
    } else {
      @glm53_packed_expert_upload.GlmRecurrentLayer::with_hyper_connection(
        @glm53_spec.Glm53FlashSpec::official(),
        layer_index=0,
        maximum_rows=32,
        maximum_sequences=16,
        cache_slots=32,
        maximum_state_bytes=134217728L,
      )
    }
    let workspace_bytes = layer.workspace_bytes()
    if with_moe_block {
      assert_eq(workspace_bytes, 152584580L)
    } else {
      assert_eq(
        workspace_bytes,
        if with_dense_block {
          148669440L
        } else {
          146043392L
        },
      )
    }
    // Packed banks include their own sixteen-byte plane alignment.
    let owned_weights = if with_moe_block { 4406639960L } else { weight_bytes }
    let total_bytes = owned_weights + workspace_bytes
    assert_true(
      @utf8.decode(layer.source_bytes()).contains(
        "glm_recurrent_\{layer_index}_15(",
      ),
    )
    let baseline = live_resources()
    let context = @device.open_context(0)
    defer (context.close() catch { _ => () })
    let stream = context.create_stream()
    defer (stream.close() catch { _ => () })
    let aot = context.load_module(b"fake\x00")
    defer (aot.close() catch { _ => () })
    let operand = context.allocate(1048576L)
    defer (operand.close() catch { _ => () })
    defer (layer.close() catch { _ => () })
    layer.prepare(
      context,
      aot,
      stream,
      checkpoint,
      root,
      maximum_device_bytes=total_bytes,
      chunk_bytes=if with_moe_block { 1048573 } else { 65533 },
      limits~,
      counts=operand,
      sequence_offsets=operand,
      sequence_slots=operand,
      reset_flags=operand,
      hidden=operand,
      output=operand,
    )
    assert_eq(layer.device_bytes(), total_bytes)
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
    assert_eq(
      enqueues() - before,
      if with_moe_block {
        1088UL
      } else if with_dense_block {
        832UL
      } else {
        640UL
      },
    )
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

///|
async test "GLM full attention sublayer binds nineteen weights and one twenty-launch queue" {
  exercise_glm_checkpoint_owner(with_dense_block=false)
}

///|
async test "GLM full dense decoder binds twenty-six weights and one twenty-six-launch queue" {
  exercise_glm_checkpoint_owner(with_dense_block=true)
}

///|
async test "GLM full recurrent MoE decoder streams all 288 packed experts into one queue" {
  exercise_glm_checkpoint_owner(with_dense_block=false, with_moe_block=true)
}
