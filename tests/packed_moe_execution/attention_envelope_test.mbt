///|
async test "DeepSeek complete attention envelopes retain all compression profiles in one parent queue" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(
      @fs.tmpdir(prefix="lunaflux-attention-envelope"),
    )
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @deepseek_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    for ratio in [0, 4, 128] {
      let layer = for layer in 0..<spec.layer_count() {
        if spec.compression_ratio(layer) == Some(ratio) {
          break layer
        }
      } nobreak {
        fail("missing compression profile")
      }
      let parts : Array[String] = []
      let entries = compressed_sublayer_entries(spec, layer).map(entry => {
        let (name, dtype, shape, size) = entry
        ("attn." + name, dtype, shape, size)
      })
      entries.append([
        ("hc_attn_fn", "F32", "24,16384", 1572864L),
        ("hc_attn_base", "F32", "24", 96L),
        ("hc_attn_scale", "F32", "3", 12L),
        ("attn_norm.weight", "BF16", "4096", 8192L),
        ("hc_ffn_fn", "F32", "24,16384", 1572864L),
        ("hc_ffn_base", "F32", "24", 96L),
        ("hc_ffn_scale", "F32", "3", 12L),
        ("ffn_norm.weight", "BF16", "4096", 8192L),
      ])
      let size = for (name, dtype, shape, bytes) in entries; offset = 0L {
        parts.push(
          "\"layers.\{layer}.\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset + bytes}]}",
        )
        continue offset + bytes
      } nobreak {
        offset
      }
      let header = @utf8.encode("{" + parts.join(",") + "}")
      let file = FixedArray::make(8 + header.length() + size.to_int(), b'\x38')
      for index in 0..<8 {
        file[index] = (header.length().to_uint64() >> (8 * index)).to_byte()
      }
      for index, byte in header {
        file[8 + index] = byte
      }
      let bytes = Bytes::from_array(file)
      let locator = "layer-\{layer}"
      @fs.write_file(directory + "/" + locator, bytes, create_mode=CreateNew)
      let root = @approved_fs.ApprovedRoot::open_absolute(directory)
      defer root.close()
      let limits = @streaming.StreamingSafetensorsLimits::new(
        maximum_shards=1,
        maximum_file_bytes=size + 8192L,
        maximum_header_bytes=8192,
        maximum_tensors_per_shard=32,
        maximum_total_tensors=32,
        maximum_rank=2,
        maximum_name_bytes=128,
        maximum_dtype_specs=4,
        authentication_chunk_bytes=65536,
        maximum_copy_requests=32,
        maximum_copy_bytes=size,
      )
      let checkpoint = @materialize.PackedCheckpoint::new(
        @streaming.inspect_shards(
          root,
          [
            @streaming.StreamingSafetensorsShardInput::new(
              locator=@approved_fs.ApprovedRelativeLocator::new(locator),
              expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
                @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
              ),
            ),
          ],
          dtype_specs=[
            @streaming.StreamingSafetensorsDTypeSpec::bfloat16(),
            @streaming.StreamingSafetensorsDTypeSpec::float32(),
            @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
            @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E8M0"),
          ],
          limits~,
        ),
      )
      device_fixture_lane.acquire()
      defer device_fixture_lane.release()
      let baseline = live_resources()
      let context = @device.open_context(0)
      defer (context.close() catch { _ => () })
      let stream = context.create_stream()
      defer (stream.close() catch { _ => () })
      let aot = context.load_module(b"fake\x00")
      defer (aot.close() catch { _ => () })
      let operand = context.allocate(3L * 4L * 4096L * 2L)
      defer (operand.close() catch { _ => () })
      let index_offsets = if ratio == 4 { Some(operand) } else { None }
      let owner = @deepseek.DeepSeekAttentionEnvelope::new(
        spec,
        layer~,
        maximum_rows=3,
        maximum_context=256,
        maximum_state_bytes=16777216L,
        maximum_workspace_bytes=16777216L,
      )
      defer (owner.close() catch { _ => () })
      let required = owner.required_device_bytes(checkpoint)
      let before = live_resources()
      try
        owner.prepare(
          context,
          aot,
          checkpoint,
          root,
          counts=operand,
          positions=operand,
          reset_flag=operand,
          input=operand,
          output=operand,
          maximum_device_bytes=required - 1L,
          index_offsets?,
          chunk_bytes=65536,
          limits~,
        )
      catch {
        @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
        e => raise e
      } noraise {
        _ => fail("ignored aggregate budget")
      }
      assert_eq(live_resources(), before)
      owner.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts=operand,
        positions=operand,
        reset_flag=operand,
        input=operand,
        output=operand,
        maximum_device_bytes=required,
        index_offsets?,
        chunk_bytes=65536,
        limits~,
      )
      let effects = match ratio {
        0 => 19
        4 => 48
        _ => 29
      }
      assert_eq(owner.launches().length(), effects)
      assert_eq(
        owner.append_descriptors().length(),
        match ratio {
          0 => 1
          4 => 5
          _ => 3
        },
      )
      let queue = context.create_ordered_kernel_executor(
        stream,
        owner.launches(),
      )
      defer (queue.abort() catch { _ => () })
      defer (queue.close() catch { _ => () })
      let before_submit = enqueues()
      let waits = blocking_syncs()
      begin_allocations()
      for _ in 0..<32 {
        for stage in 0..<effects {
          queue.enqueue(stage)
        }
        queue.record_completion()
        while !queue.poll_completion() {
          ()
        }
        queue.reset()
      }
      assert_eq(end_allocations(), 0UL)
      assert_eq(enqueues() - before_submit, 32UL * effects.to_uint64())
      assert_eq(blocking_syncs(), waits)
      queue.close()
      owner.close()
      owner.close()
      assert_eq(live_resources(), before)
      let block = @deepseek_decoder.DeepSeekDecoderBlock::new(
        spec,
        layer~,
        maximum_rows=3,
        maximum_context=256,
        maximum_state_bytes=16777216L,
        maximum_workspace_bytes=16777216L,
        maximum_expert_bytes=8589934592L,
        accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
      )
      defer (block.close() catch { _ => () })
      let (routed_bytes, shared_bytes) = block.bank_bytes()
      assert_true(routed_bytes + shared_bytes < 4294967296L)
      let routed_bank = context.allocate(routed_bytes)
      defer (routed_bank.close() catch { _ => () })
      let shared_bank = context.allocate(shared_bytes)
      defer (shared_bank.close() catch { _ => () })
      let router_weight = context.allocate(
        spec.routed_expert_count().to_int64() *
        spec.hidden_size().to_int64() *
        2L,
      )
      defer (router_weight.close() catch { _ => () })
      let table = context.allocate(
        spec.vocabulary_size().to_int64() *
        spec.experts_per_token().to_int64() *
        4L,
      )
      defer (table.close() catch { _ => () })
      let correction_bias = if layer < spec.hash_layer_count() {
        None
      } else {
        Some(operand)
      }
      let token_ids = if layer < spec.hash_layer_count() {
        Some(operand)
      } else {
        None
      }
      let token_table = if layer < spec.hash_layer_count() {
        Some(table)
      } else {
        None
      }
      let required = block.required_device_bytes(checkpoint)
      let before_block = live_resources()
      try
        block.prepare(
          context,
          aot,
          checkpoint,
          root,
          counts=operand,
          positions=operand,
          reset_flag=operand,
          input=operand,
          output=operand,
          index_offsets?,
          routed_bank~,
          shared_bank~,
          router_weight~,
          correction_bias?,
          token_ids?,
          token_table?,
          maximum_device_bytes=required - 1L,
          chunk_bytes=65536,
          limits~,
        )
      catch {
        @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
        e => raise e
      } noraise {
        _ => fail("ignored complete block budget")
      }
      assert_eq(live_resources(), before_block)
      block.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts=operand,
        positions=operand,
        reset_flag=operand,
        input=operand,
        output=operand,
        index_offsets?,
        routed_bank~,
        shared_bank~,
        router_weight~,
        correction_bias?,
        token_ids?,
        token_table?,
        maximum_device_bytes=required,
        chunk_bytes=65536,
        limits~,
      )
      let total_effects = block.launches().length()
      // Two dynamic-FP8 expert branches (5 each), combine (1), router (3/4), mHC (4).
      assert_eq(
        total_effects - effects,
        if layer < spec.hash_layer_count() {
          19
        } else {
          18
        },
      )
      let queue = context.create_ordered_kernel_executor(
        stream,
        block.launches(),
      )
      defer (queue.abort() catch { _ => () })
      defer (queue.close() catch { _ => () })
      let before_submit = enqueues()
      let waits = blocking_syncs()
      begin_allocations()
      for _ in 0..<32 {
        for stage in 0..<total_effects {
          queue.enqueue(stage)
        }
        queue.record_completion()
        while !queue.poll_completion() {
          ()
        }
        queue.reset()
      }
      assert_eq(end_allocations(), 0UL)
      assert_eq(enqueues() - before_submit, 32UL * total_effects.to_uint64())
      assert_eq(blocking_syncs(), waits)
      queue.close()
      block.close()
      block.close()
      assert_eq(live_resources(), before_block)
      table.close()
      router_weight.close()
      shared_bank.close()
      routed_bank.close()
      operand.close()
      aot.close()
      stream.close()
      context.close()
      assert_eq(live_resources(), baseline)
    }
  }
}
