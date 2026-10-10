///|
async test "DSpark main prefix uploads exact compact mtp weights and rejects partial captures" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-prediction-main"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @deepseek_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let plan = @prediction_main.prediction_main_plan(
      spec,
      maximum_rows=1,
      maximum_workspace_bytes=16384L,
    )
    let matrix = plan.weight_representation()
    let matrix_bytes = matrix.payload_bytes()
    let scale_bytes = matrix.scale_bytes()
    let norm_bytes = 8192L
    let weight_bytes = matrix_bytes + scale_bytes + norm_bytes
    let total = weight_bytes + plan.workspace_bytes()
    let header = @utf8.encode(
      "{\"mtp.0.main_proj.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[4096,12288],\"data_offsets\":[0,\{matrix_bytes}]}," +
      "\"mtp.0.main_proj.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[32,96],\"data_offsets\":[\{matrix_bytes},\{matrix_bytes + scale_bytes}]}," +
      "\"mtp.0.main_norm.weight\":{\"dtype\":\"BF16\",\"shape\":[4096],\"data_offsets\":[\{matrix_bytes + scale_bytes},\{weight_bytes}]}}",
    )
    let file = FixedArray::make(
      8 + header.length() + weight_bytes.to_int(),
      b'\x38',
    )
    for i in 0..<8 {
      file[i] = (header.length().to_uint64() >> (i * 8)).to_byte()
    }
    for i, b in header {
      file[8 + i] = b
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/weights", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=weight_bytes + 8192L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=3,
      maximum_total_tensors=3,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=3,
      maximum_copy_requests=3,
      maximum_copy_bytes=weight_bytes,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              "a".repeat(64),
            ),
          ),
        ],
        dtype_specs=[
          @streaming.StreamingSafetensorsDTypeSpec::bfloat16(),
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E8M0"),
        ],
        limits~,
      ),
    )
    let bound = @prediction_main.bind_prediction_main_weights(checkpoint, spec)
    assert_eq(bound.map(w => w.buffer_bytes()), [
      matrix_bytes + scale_bytes,
      norm_bytes,
    ])
    // Serialize only the process-global native test double, not production.
    device_fixture_lane.acquire()
    defer device_fixture_lane.release()
    let context = @device.open_context(0)
    defer context.close()
    let stream = context.create_stream()
    defer stream.close()
    let aot = context.load_module(b"fake\x00")
    defer aot.close()
    let input = context.allocate(24576L)
    defer input.close()
    let counts = context.allocate(20L)
    defer counts.close()
    let owner = @prediction_main.DeepSeekPredictionMain::new(
      spec,
      maximum_rows=1,
      maximum_workspace_bytes=16384L,
    )
    defer owner.close()
    assert_eq(owner.required_device_bytes(checkpoint), total)
    let baseline = live_resources()
    for layers in [[40, 42], [42, 41, 40], [40, 41, 42, 42]] {
      try
        owner.prepare(
          context,
          aot,
          checkpoint,
          root,
          counts~,
          captured_hidden=input,
          target_layers=layers,
          maximum_device_bytes=total,
          chunk_bytes=4093,
          limits~,
        )
      catch {
        @precision.PrecisionPlanError::IllegalPlacement => ()
        e => raise e
      } noraise {
        _ => fail("partial/unordered prediction capture accepted")
      }
      assert_eq(live_resources(), baseline)
    }
    try
      owner.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts~,
        captured_hidden=input,
        target_layers=[40, 41, 42],
        maximum_device_bytes=total - 1L,
        chunk_bytes=4093,
        limits~,
      )
    catch {
      @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
      e => raise e
    } noraise {
      _ => fail("device budget ignored")
    }
    assert_eq(live_resources(), baseline)
    owner.prepare(
      context,
      aot,
      checkpoint,
      root,
      counts~,
      captured_hidden=input,
      target_layers=[40, 41, 42],
      maximum_device_bytes=total,
      chunk_bytes=4093,
      limits~,
    )
    assert_eq(owner.output().byte_count(), 8192L)
    let queue = context.create_ordered_kernel_executor(stream, owner.launches())
    defer (queue.close() catch { _ => () })
    defer (queue.abort() catch { _ => () })
    begin_allocations()
    for _ in 0..<16 {
      queue.enqueue(0)
      queue.enqueue(1)
      queue.record_completion()
      while !queue.poll_completion() {
        ()
      }
      queue.reset()
    }
    assert_eq(end_allocations(), 0UL)
    queue.close()
    owner.close()
    owner.close()
    assert_eq(live_resources(), baseline)
  }
}
