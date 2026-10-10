///|
async fn exercise_glm_checkpoint_owner(
  with_dense_block~ : Bool,
  with_moe_block? : Bool = false,
  with_sparse_block? : Bool = false,
  with_decoder_slice? : Bool = false,
  with_decoder_stage? : Bool = false,
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
    if with_sparse_block {
      shapes.clear()
      shapes.append([
        ("q_a_proj.weight", "BF16", "1536,4096", 12582912L),
        ("q_a_layernorm.weight", "BF16", "1536", 3072L),
        ("q_b_proj.weight", "BF16", "16384,1536", 50331648L),
        ("kv_a_proj_with_mqa.weight", "BF16", "512,4096", 4194304L),
        ("kv_a_layernorm.weight", "BF16", "512", 1024L),
        ("kv_b_proj.weight", "BF16", "32768,512", 33554432L),
        ("indexer.wq_b.weight", "BF16", "4096,1536", 12582912L),
        ("indexer.wk.weight", "BF16", "128,4096", 1048576L),
        ("indexer.k_norm.weight", "BF16", "128", 256L),
        ("indexer.k_norm.bias", "BF16", "128", 256L),
        ("indexer.weights_proj.weight", "BF16", "32,4096", 262144L),
        ("indexer.index_kpool_compress_gate", "BF16", "128,4096", 1048576L),
        ("o_proj.weight", "BF16", "4096,16384", 134217728L),
        ("indexer.index_kpool_compress_ape", "BF16", "4,128", 1024L),
        ("hc_attn_fn", "BF16", "24,16384", 786432L),
        ("hc_attn_base", "F32", "24", 96L),
        ("hc_attn_scale", "F32", "3", 12L),
        ("input_layernorm.weight", "BF16", "4096", 8192L),
      ])
    }
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
    let layer_index = if with_sparse_block {
      3
    } else if with_moe_block {
      4
    } else {
      0
    }
    let weight_bytes = if with_sparse_block {
      shapes.fold(init=0L, (sum, entry) => sum + entry.3)
    } else if with_moe_block {
      4406629592L
    } else if with_dense_block {
      579060440L
    } else {
      276275820L
    }
    let layer_count = if with_decoder_slice && !with_sparse_block {
      3
    } else {
      1
    }
    let global_bytes = if with_decoder_stage {
      @glm53_spec.Glm53FlashSpec::official().vocabulary_size().to_int64() *
      4096L *
      2L
    } else {
      0L
    }
    let weight_bytes = weight_bytes * layer_count.to_int64() + global_bytes
    let entries : Array[String] = []
    let mut offset = 0L
    for layer in layer_index..<(layer_index + layer_count) {
      for index, (name, dtype, shape, count) in shapes {
        let parent = if index < (if with_sparse_block { 14 } else { 15 }) {
          "self_attn."
        } else {
          ""
        }
        entries.push(
          "\"model.language_model.layers.\{layer}.\{parent}\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset + count}]}",
        )
        offset += count
      }
    }
    device_fixture_lane.acquire()
    defer device_fixture_lane.release()
    if with_decoder_stage {
      let vocabulary = @glm53_spec.Glm53FlashSpec::official().vocabulary_size()
      entries.push(
        "\"model.language_model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[\{vocabulary},4096],\"data_offsets\":[\{offset},\{offset + global_bytes}]}",
      )
      offset += global_bytes
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
      maximum_tensors_per_shard=shapes.length() * layer_count +
        (if with_decoder_stage { 1 } else { 0 }),
      maximum_total_tensors=shapes.length() * layer_count +
        (if with_decoder_stage { 1 } else { 0 }),
      maximum_rank=3,
      maximum_name_bytes=256,
      maximum_dtype_specs=4,
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
    if !with_sparse_block {
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
    }
    if with_decoder_stage {
      exercise_glm_stage(checkpoint, root, limits, weight_bytes)
      return
    }
    let layer : CheckpointTestLayer = if with_decoder_slice {
      Slice(
        @glm53_packed_expert_upload.GlmDecoderSlice::new(
          @glm53_spec.Glm53FlashSpec::official(),
          first_layer=layer_index,
          end_layer=layer_index + layer_count,
          maximum_rows=32,
          maximum_sequences=if with_sparse_block { 1 } else { 16 },
          cache_slots=if with_sparse_block { 1 } else { 32 },
          maximum_history=64,
          index_layer_norm_epsilon=1.0e-5,
          maximum_layer_state_bytes=160000000L,
          maximum_layer_workspace_bytes=200000000L,
          maximum_layer_expert_bytes=5000000000L,
          accuracy=@precision.Accuracy::new(absolute=0.125, relative=0.125),
        ),
      )
    } else if with_sparse_block {
      Sparse(
        @glm53_packed_expert_upload.GlmSparseLayer::new(
          @glm53_spec.Glm53FlashSpec::official(),
          layer_index~,
          maximum_queries=32,
          maximum_history=64,
          index_layer_norm_epsilon=1.0e-5,
          maximum_state_bytes=10000000L,
          maximum_workspace_bytes=20000000L,
          maximum_expert_bytes=5000000000L,
          accuracy=@precision.Accuracy::new(absolute=0.125, relative=0.125),
        ),
      )
    } else if with_moe_block {
      Recurrent(
        @glm53_packed_expert_upload.GlmRecurrentLayer::with_moe_block(
          @glm53_spec.Glm53FlashSpec::official(),
          layer_index~,
          maximum_rows=32,
          maximum_sequences=16,
          cache_slots=32,
          maximum_state_bytes=134217728L,
          maximum_expert_bytes=5000000000L,
          accuracy=@precision.Accuracy::new(absolute=0.125, relative=0.125),
        ),
      )
    } else if with_dense_block {
      Recurrent(
        @glm53_packed_expert_upload.GlmRecurrentLayer::with_dense_block(
          @glm53_spec.Glm53FlashSpec::official(),
          layer_index=0,
          maximum_rows=32,
          maximum_sequences=16,
          cache_slots=32,
          maximum_state_bytes=134217728L,
        ),
      )
    } else {
      Recurrent(
        @glm53_packed_expert_upload.GlmRecurrentLayer::with_hyper_connection(
          @glm53_spec.Glm53FlashSpec::official(),
          layer_index=0,
          maximum_rows=32,
          maximum_sequences=16,
          cache_slots=32,
          maximum_state_bytes=134217728L,
        ),
      )
    }
    let workspace_bytes = layer.workspace_bytes()
    if with_decoder_slice && !with_sparse_block {
      assert_eq(workspace_bytes, 3L * 148669440L + 2L * 1048576L)
    } else if with_sparse_block {
      assert_true(workspace_bytes > 0L)
    } else if with_moe_block {
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
    let owned_weights = if with_sparse_block {
      weight_bytes + 10368L
    } else if with_moe_block {
      4406639960L
    } else {
      weight_bytes
    }
    let total_bytes = owned_weights + workspace_bytes
    assert_true(
      @utf8.decode(layer.source_bytes()).contains(
        if with_sparse_block {
          "glm_sparse_\{layer_index}_output("
        } else {
          "glm_recurrent_\{layer_index}_15("
        },
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
    let output = context.allocate(1048576L)
    defer (output.close() catch { _ => () })
    defer (layer.close() catch { _ => () })
    layer.prepare(
      context,
      aot,
      stream,
      checkpoint,
      root,
      operand,
      output,
      total_bytes,
      if with_moe_block {
        1048573
      } else {
        65533
      },
      limits,
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
      if with_decoder_slice && !with_sparse_block {
        2496UL
      } else if with_sparse_block {
        1184UL
      } else if with_moe_block {
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
    output.close()
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

///|
async test "GLM sparse decoder streams exact projection output APE and FFN operands into one queue" {
  exercise_glm_checkpoint_owner(
    with_dense_block=false,
    with_moe_block=true,
    with_sparse_block=true,
  )
}

///|
async test "GLM three-layer decoder slice owns one queue and alternating residual frames" {
  exercise_glm_checkpoint_owner(with_dense_block=true, with_decoder_slice=true)
}

///|
async test "GLM sparse decoder frame composes its checkpoint FFN in a slice queue" {
  exercise_glm_checkpoint_owner(
    with_dense_block=false,
    with_moe_block=true,
    with_sparse_block=true,
    with_decoder_slice=true,
  )
}
