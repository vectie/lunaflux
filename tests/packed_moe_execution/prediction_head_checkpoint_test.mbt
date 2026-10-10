///|
async test "prediction head binds final mtp controls and borrows the global vocabulary" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-prediction-head"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @deepseek_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let v = spec.vocabulary_size()
    let h = spec.hidden_size()
    let k = spec.dspark_markov_rank()
    let fields : Array[(String, String, String, Int64)] = [
      ("norm.weight", "BF16", "\{h}", h.to_int64() * 2L),
      ("hc_head_fn", "F32", "4,\{h*4}", h.to_int64() * 64L),
      ("hc_head_base", "F32", "4", 16L),
      ("hc_head_scale", "F32", "1", 4L),
      (
        "markov_head.markov_w1.weight",
        "BF16",
        "\{v},\{k}",
        v.to_int64() * k.to_int64() * 2L,
      ),
      (
        "markov_head.markov_w2.weight",
        "BF16",
        "\{v},\{k}",
        v.to_int64() * k.to_int64() * 2L,
      ),
      (
        "confidence_head.proj.weight",
        "BF16",
        "1,\{h+k}",
        (h + k).to_int64() * 2L,
      ),
    ]
    let parts : Array[String] = []
    let mut offset = 0L
    for (name, dtype, shape, size) in fields {
      parts.push(
        "\"mtp.2.\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset+size}]}",
      )
      offset += size
    }
    let header = @utf8.encode("{" + parts.join(",") + "}")
    let file = FixedArray::make(8 + header.length() + offset.to_int(), b'\x38')
    for i in 0..<8 {
      file[i] = (header.length().to_uint64() >> (8 * i)).to_byte()
    }
    for i, byte in header {
      file[8 + i] = byte
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/weights", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=offset + 8192L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=7,
      maximum_total_tensors=7,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=2,
      authentication_chunk_bytes=65536,
      maximum_copy_requests=7,
      maximum_copy_bytes=offset,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
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
    let bound = @prediction_main.bind_prediction_head_weights(checkpoint, spec)
    assert_eq(bound.length(), 7)
    assert_eq(bound.fold(init=0L, (sum, w) => sum + w.buffer_bytes()), offset)
    device_fixture_lane.acquire()
    defer device_fixture_lane.release()
    let baseline = live_resources()
    let context = @device.open_context(0)
    defer (context.close() catch { _ => () })
    let aot = context.load_module(b"fake\x00")
    defer (aot.close() catch { _ => () })
    let stream = context.create_stream()
    defer (stream.close() catch { _ => () })
    let input = context.allocate(163840L)
    defer (input.close() catch { _ => () })
    let vocabulary = context.allocate(v.to_int64() * h.to_int64() * 2L)
    defer (vocabulary.close() catch { _ => () })
    let tokens = context.allocate(24L)
    defer (tokens.close() catch { _ => () })
    let confidence = context.allocate(20L)
    defer (confidence.close() catch { _ => () })
    let owner = @prediction_main.DeepSeekPredictionHead::new(
      spec,
      temperature=0.7,
      maximum_workspace_bytes=4194304L,
    )
    defer (owner.close() catch { _ => () })
    let total = owner.required_device_bytes(checkpoint)
    assert_true(total < offset + 4194304L)
    let before = live_resources()
    try
      owner.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts=input,
        residual=input,
        vocabulary_weight=vocabulary,
        tokens~,
        confidence~,
        rng=input,
        maximum_device_bytes=total - 1L,
        chunk_bytes=65536,
        limits~,
      )
    catch {
      @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
      e => raise e
    } noraise {
      _ => fail("head budget ignored")
    }
    assert_eq(live_resources(), before)
    owner.prepare(
      context,
      aot,
      checkpoint,
      root,
      counts=input,
      residual=input,
      vocabulary_weight=vocabulary,
      tokens~,
      confidence~,
      rng=input,
      maximum_device_bytes=total,
      chunk_bytes=65536,
      limits~,
    )
    assert_eq(owner.launches().length(), 23)
    assert_eq(owner.logits().byte_count(), v.to_int64() * 5L * 4L)
    let queue = context.create_ordered_kernel_executor(stream, owner.launches())
    defer (queue.abort() catch { _ => () })
    defer (queue.close() catch { _ => () })
    let before_launch = enqueues()
    begin_allocations()
    for _ in 0..<32 {
      for stage in 0..<23 {
        queue.enqueue(stage)
      }
      queue.record_completion()
      while !queue.poll_completion() {
        ()
      }
      queue.reset()
    }
    assert_eq(end_allocations(), 0UL)
    assert_eq(enqueues() - before_launch, 736UL)
    queue.close()
    owner.close()
    owner.close()
    // Borrowed vocabulary and result ports remain live after closing the owner.
    vocabulary.close()
    tokens.close()
    confidence.close()
    input.close()
    aot.close()
    stream.close()
    context.close()
    assert_eq(live_resources(), baseline)
  }
}
