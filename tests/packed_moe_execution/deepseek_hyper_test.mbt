///|
async test "checkpoint F32 hyper envelopes stream exact controls and borrow four allocation-free effects" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-deepseek-hyper"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let entries : Array[String] = []
    let payload_bytes = for
      (prefix, name) in [
        ("layers.3.", "attn"),
        ("layers.3.", "ffn"),
        ("mtp.2.", "attn"),
        ("mtp.2.", "ffn"),
      ]
      offset = 0L {
      let next = for
        (suffix, dtype, shape, size) in [
          ("hc_" + name + "_fn", "F32", "24,16384", 1572864L),
          ("hc_" + name + "_base", "F32", "24", 96L),
          ("hc_" + name + "_scale", "F32", "3", 12L),
          (name + "_norm.weight", "BF16", "4096", 8192L),
        ]
        current = offset {
        entries.push(
          "\"\{prefix}\{suffix}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{current},\{current + size}]}",
        )
        continue current + size
      } nobreak {
        current
      }
      continue next
    } nobreak {
      offset
    }
    let header = @utf8.encode("{" + entries.join(",") + "}")
    let file = FixedArray::make(
      8 + header.length() + payload_bytes.to_int(),
      b'\x38',
    )
    for i in 0..<8 {
      file[i] = (header.length().to_uint64() >> (i * 8)).to_byte()
    }
    for i, byte in header {
      file[8 + i] = byte
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/hyper", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=8388608L,
      maximum_header_bytes=8192,
      maximum_tensors_per_shard=16,
      maximum_total_tensors=16,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=2,
      authentication_chunk_bytes=65536,
      maximum_copy_requests=16,
      maximum_copy_bytes=payload_bytes,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("hyper"),
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
            ),
          ),
        ],
        dtype_specs=[
          @streaming.StreamingSafetensorsDTypeSpec::float32(),
          @streaming.StreamingSafetensorsDTypeSpec::bfloat16(),
        ],
        limits~,
      ),
    )
    let spec = @deepseek_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
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
    let counts = context.allocate(20L)
    let input = context.allocate(3L * 4L * 4096L * 2L)
    let output = context.allocate(input.byte_count())
    defer (counts.close() catch { _ => () })
    defer (input.close() catch { _ => () })
    defer (output.close() catch { _ => () })
    for
      (block, branch) in [
        (
          @deepseek_spec.DeepSeekV4BlockAddress::BaseLayer(3),
          @deepseek_hyper.DeepSeekHyperBranch::Attention,
        ),
        (BaseLayer(3), FeedForward),
        (PredictionStage(2), Attention),
        (PredictionStage(2), FeedForward),
      ] {
      let owner = @deepseek_hyper.DeepSeekHyperConnection::for_block(
        spec,
        block~,
        branch~,
        maximum_rows=3,
      )
      defer (owner.close() catch { _ => () })
      let required = owner.required_device_bytes(checkpoint)
      let before_prepare = live_resources()
      try
        owner.prepare(
          context,
          aot,
          checkpoint,
          root,
          counts~,
          input~,
          output~,
          maximum_device_bytes=required - 1L,
          chunk_bytes=4093,
          limits~,
        )
      catch {
        @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
        e => raise e
      } noraise {
        _ => fail("aggregate budget was ignored")
      }
      assert_eq(live_resources(), before_prepare)
      owner.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts~,
        input~,
        output~,
        maximum_device_bytes=required,
        chunk_bytes=4093,
        limits~,
      )
      assert_eq(owner.prefix().length(), 3)
      assert_eq(owner.suffix().length(), 1)
      assert_eq(owner.branch_input().byte_count(), 3L * 4096L * 2L)
      let launches = owner.prefix().to_owned()
      launches.append(owner.suffix())
      let queue = context.create_ordered_kernel_executor(stream, launches)
      defer (queue.abort() catch { _ => () })
      defer (queue.close() catch { _ => () })
      let before = enqueues()
      let waits = blocking_syncs()
      begin_allocations()
      for _ in 0..<32 {
        for stage in 0..<4 {
          queue.enqueue(stage)
        }
        queue.record_completion()
        while !queue.poll_completion() {
          ()
        }
        queue.reset()
      }
      assert_eq(end_allocations(), 0UL)
      assert_eq(enqueues() - before, 128UL)
      assert_eq(blocking_syncs(), waits)
      queue.close()
      owner.close()
      owner.close()
      assert_eq(live_resources(), before_prepare)
    }
    output.close()
    input.close()
    counts.close()
    aot.close()
    stream.close()
    context.close()
    assert_eq(live_resources(), baseline)
  }
}

///|
test "F32 hyper projection rejects half-width checkpoint banks before scratch allocation" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let envelope = @hyper_frame.HyperConnectionFrame::new(
    @hyper_source.hyper_connection_source(
      @precision.HyperConnectionPrecision::new(
        stream_count=4,
        hidden_width=6,
        maximum_rows=3,
        sinkhorn_iterations=20,
        control_epsilon=0.000001,
        normalization_epsilon=0.00001,
        control_law=F32ProjectionThenNormalize,
        mix_law=TransposedF32ResidualThenBranch,
      ),
      symbol_prefix="short_control",
    ),
  )
  defer (envelope.close() catch { _ => () })
  let short = context.allocate(24L * 24L * 2L)
  defer (short.close() catch { _ => () })
  let before = live_resources()
  try
    envelope.prepare(
      context,
      aot,
      counts=short,
      input=short,
      output=short,
      function_weight=short,
      base=short,
      scale=short,
      norm_weight=short,
      maximum_workspace_bytes=envelope.workspace_bytes(),
    )
  catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("BF16-sized function bank accepted as F32")
  }
  assert_eq(live_resources(), before)
}
