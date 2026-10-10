///|
async test "checkpoint-backed learned indexer shares nineteen allocation-free effects" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-full-indexer"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @deepseek_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let query = @deepseek.index_queries_plan(
      spec,
      layer=2,
      maximum_rows=3,
      maximum_workspace_bytes=1048576L,
    )
    let pool = @deepseek.compressor_pool_plan(
      spec,
      layer=2,
      for_indexer=true,
      maximum_rows=3,
      maximum_context=16,
      maximum_state_bytes=1048576L,
      maximum_workspace_bytes=1048576L,
    )
    let (_, width, ratio, _, _, _) = pool.geometry()
    let (qr, qc) = query.query_weight().shape()
    let entries = [
      ("wq_b.weight", "F8_E4M3", "\{qr},\{qc}", query.weight_bytes()[0]),
      (
        "wq_b.scale",
        "F8_E8M0",
        "\{qr / 128},\{qc / 128}",
        query.weight_bytes()[1],
      ),
      (
        "weights_proj.weight",
        "BF16",
        "\{spec.index_head_count()},\{spec.hidden_size()}",
        query.weight_bytes()[2],
      ),
      (
        "compressor.wkv.weight",
        "BF16",
        "\{2 * width},\{spec.hidden_size()}",
        pool.parameter_bytes()[0],
      ),
      (
        "compressor.wgate.weight",
        "BF16",
        "\{2 * width},\{spec.hidden_size()}",
        pool.parameter_bytes()[1],
      ),
      (
        "compressor.ape",
        "F32",
        "\{ratio},\{2 * width}",
        pool.parameter_bytes()[2],
      ),
      ("compressor.norm.weight", "BF16", "\{width}", pool.parameter_bytes()[3]),
    ]
    let parts : Array[String] = []
    let mut size = 0L
    for (name, dtype, shape, bytes) in entries {
      parts.push(
        "\"layers.2.attn.indexer.\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{size},\{size + bytes}]}",
      )
      size += bytes
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
    @fs.write_file(directory + "/indexer", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=size + 4096L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=7,
      maximum_total_tensors=7,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=4,
      maximum_copy_requests=7,
      maximum_copy_bytes=size,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("indexer"),
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
    let aot = context.load_module(b"fake\x00")
    defer (aot.close() catch { _ => () })
    let stream = context.create_stream()
    defer (stream.close() catch { _ => () })
    let operand = context.allocate(65536L)
    defer (operand.close() catch { _ => () })
    let indexer = @deepseek.DeepSeekLearnedIndexer::new(
      spec,
      layer=2,
      maximum_rows=3,
      maximum_context=16,
      maximum_state_bytes=1048576L,
      maximum_workspace_bytes=1048576L,
    )
    defer (indexer.close() catch { _ => () })
    let required = indexer.required_device_bytes(checkpoint)
    assert_true(required > size)
    let before_prepare = live_resources()
    try
      indexer.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts=operand,
        positions=operand,
        reset_flag=operand,
        hidden=operand,
        low_rank_queries=operand,
        maximum_device_bytes=required - 1L,
        chunk_bytes=4093,
        limits~,
      )
    catch {
      @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
      error => raise error
    } noraise {
      _ => fail("aggregate budget was granted separately to each owner")
    }
    assert_eq(live_resources(), before_prepare)
    indexer.prepare(
      context,
      aot,
      checkpoint,
      root,
      counts=operand,
      positions=operand,
      reset_flag=operand,
      hidden=operand,
      low_rank_queries=operand,
      maximum_device_bytes=required,
      chunk_bytes=4093,
      limits~,
    )
    assert_eq(indexer.launches().length(), 19)
    assert_eq(indexer.append_descriptors().length(), 2)
    assert_eq(indexer.outputs().row_ids.byte_count(), 48L)
    assert_eq(indexer.outputs().row_counts.byte_count(), 12L)
    let queue = context.create_ordered_kernel_executor(
      stream,
      indexer.launches().to_owned(),
    )
    defer (queue.abort() catch { _ => () })
    defer (queue.close() catch { _ => () })
    let before = enqueues()
    let waits = blocking_syncs()
    begin_allocations()
    for _ in 0..<32 {
      for stage in 0..<19 {
        queue.enqueue(stage)
      }
      queue.record_completion()
      while !queue.poll_completion() {
        ()
      }
      queue.reset()
    }
    assert_eq(end_allocations(), 0UL)
    assert_eq(enqueues() - before, 608UL)
    assert_eq(blocking_syncs(), waits)
    queue.close()
    indexer.close()
    indexer.close()
    operand.close()
    stream.close()
    aot.close()
    context.close()
    assert_eq(live_resources(), baseline)
  }
}
