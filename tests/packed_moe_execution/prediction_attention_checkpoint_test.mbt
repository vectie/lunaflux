///|
async test "DSpark attention shares compact main/draft KV weights in both prepared phases" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(
      @fs.tmpdir(prefix="lunaflux-prediction-attention"),
    )
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @deepseek_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let q = @precision.QuantizedQueryKvPrecision::new(
      hidden_width=spec.hidden_size(),
      query_rank=spec.q_lora_rank(),
      head_count=spec.attention_head_count(),
      head_width=spec.head_dimension(),
      maximum_rows=5,
      normalization_epsilon=spec.rms_norm_epsilon(),
      activation_amax_floor=0.0001,
      maximum_workspace_bytes=4194304L,
    )
    let matrices = q.weight_representations()[:].to_array()
    matrices.append(
      @prediction_main.prediction_attention_output_plan(
        spec,
        maximum_positions=256,
        maximum_workspace_bytes=4194304L,
      ).weight_representations(),
    )
    let fields : Array[(String, String, String, Int64)] = []
    for i, name in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
      let matrix = matrices[i]
      let (rows, columns) = matrix.shape()
      fields.append([
        (
          name + ".weight",
          "F8_E4M3",
          "\{rows},\{columns}",
          matrix.payload_bytes(),
        ),
        (
          name + ".scale",
          "F8_E8M0",
          "\{rows / 128},\{columns / 128}",
          matrix.scale_bytes(),
        ),
      ])
    }
    fields.append([
      (
        "q_norm.weight",
        "BF16",
        "\{spec.q_lora_rank()}",
        spec.q_lora_rank().to_int64() * 2L,
      ),
      (
        "kv_norm.weight",
        "BF16",
        "\{spec.head_dimension()}",
        spec.head_dimension().to_int64() * 2L,
      ),
      (
        "attn_sink",
        "F32",
        "\{spec.attention_head_count()}",
        spec.attention_head_count().to_int64() * 4L,
      ),
    ])
    let parts : Array[String] = []
    let mut offset = 0L
    for (name, dtype, shape, size) in fields {
      parts.push(
        "\"mtp.0.attn.\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset + size}]}",
      )
      offset += size
    }
    let header = @utf8.encode("{" + parts.join(",") + "}")
    let file = FixedArray::make(8 + header.length() + offset.to_int(), b'\x38')
    for i in 0..<8 {
      file[i] = (header.length().to_uint64() >> (8 * i)).to_byte()
    }
    for i, value in header {
      file[8 + i] = value
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/weights", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=offset + 8192L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=13,
      maximum_total_tensors=13,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=4,
      maximum_copy_requests=13,
      maximum_copy_bytes=offset,
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
          @streaming.StreamingSafetensorsDTypeSpec::float32(),
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E8M0"),
        ],
        limits~,
      ),
    )
    let bound = @prediction_main.bind_prediction_attention_weights(
      checkpoint,
      spec,
      stage=0,
    )
    assert_eq(bound.length(), 8)
    assert_eq(bound.fold(init=0L, (sum, w) => sum + w.buffer_bytes()), offset)
    try
      @prediction_main.bind_prediction_attention_weights(
        checkpoint,
        spec,
        stage=1,
      )
    catch {
      @materialize.PackedCheckpointError::MissingTensor(
        "mtp.1.attn.wq_a.weight"
      ) => ()
      e => raise e
    } noraise {
      _ =>
        fail("missing mtp stage substituted with base or first-stage weights")
    }
    device_fixture_lane.acquire()
    defer device_fixture_lane.release()
    let baseline = live_resources()
    let context = @device.open_context(0)
    defer (context.close() catch { _ => () })
    let stream = context.create_stream()
    defer (stream.close() catch { _ => () })
    let aot = context.load_module(b"fake\x00")
    defer (aot.close() catch { _ => () })
    let main = context.allocate(65536L)
    defer (main.close() catch { _ => () })
    let draft = context.allocate(65536L)
    defer (draft.close() catch { _ => () })
    let owner = @prediction_main.DeepSeekPredictionAttention::new(
      spec,
      stage=0,
      maximum_main_rows=8,
      maximum_positions=256,
      maximum_workspace_bytes=4194304L,
    )
    defer (owner.close() catch { _ => () })
    let total = offset + owner.workspace_bytes()
    assert_eq(owner.required_device_bytes(checkpoint), total)
    let before = live_resources()
    let block = @prediction_main.DeepSeekPredictionBlock::new(
      spec,
      stage=0,
      maximum_main_rows=8,
      maximum_positions=256,
      maximum_workspace_bytes=16777216L,
      maximum_expert_bytes=8589934592L,
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    )
    defer (block.close() catch { _ => () })
    try block.required_device_bytes(checkpoint) catch {
      @materialize.PackedCheckpointError::MissingTensor(
        "mtp.0.ffn.experts.0.w1.weight"
      ) => ()
      e => raise e
    } noraise {
      _ =>
        fail("attention-only checkpoint accepted as complete predictor block")
    }
    block.close()
    assert_eq(live_resources(), before)
    try
      owner.prepare(
        context,
        aot,
        checkpoint,
        root,
        main_counts=main,
        draft_counts=draft,
        main_positions=main,
        draft_positions=draft,
        main_hidden=main,
        draft_hidden=draft,
        reset_flag=main,
        start_position=draft,
        maximum_device_bytes=total - 1L,
        chunk_bytes=4093,
        limits~,
      )
    catch {
      @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
      e => raise e
    } noraise {
      _ => fail("budget not checked before weight upload")
    }
    assert_eq(live_resources(), before)
    owner.prepare(
      context,
      aot,
      checkpoint,
      root,
      main_counts=main,
      draft_counts=draft,
      main_positions=main,
      draft_positions=draft,
      main_hidden=main,
      draft_hidden=draft,
      reset_flag=main,
      start_position=draft,
      maximum_device_bytes=total,
      chunk_bytes=4093,
      limits~,
    )
    assert_eq(owner.prime_launches().length(), 7)
    assert_eq(owner.prediction_launches().length(), 19)
    assert_eq(owner.persistent_state_bytes(), 131080L)
    let persistent = owner.persistent_state()
    assert_eq(persistent.map(b => b.byte_count()), [131072L, 8L])
    for buffer in persistent {
      assert_false(buffer.same_storage(owner.output()))
      assert_false(buffer.same_storage(main))
      assert_false(buffer.same_storage(draft))
    }
    assert_eq(
      owner.output().byte_count(),
      5L * spec.hidden_size().to_int64() * 2L,
    )
    let prime = context.create_ordered_kernel_executor(
      stream,
      owner.prime_launches(),
    )
    defer (prime.close() catch { _ => () })
    defer (prime.abort() catch { _ => () })
    let predict = context.create_ordered_kernel_executor(
      stream,
      owner.prediction_launches(),
    )
    defer (predict.close() catch { _ => () })
    defer (predict.abort() catch { _ => () })
    begin_allocations()
    for _ in 0..<8 {
      for i in 0..<7 {
        prime.enqueue(i)
      }
      prime.record_completion()
      while !prime.poll_completion() {
        ()
      }
      prime.reset()
      for i in 0..<19 {
        predict.enqueue(i)
      }
      predict.record_completion()
      while !predict.poll_completion() {
        ()
      }
      predict.reset()
    }
    assert_eq(end_allocations(), 0UL)
    predict.enqueue(0)
    predict.abort()
    predict.close()
    prime.close()
    owner.close()
    let result_bytes = 5L * spec.hidden_size().to_int64() * 2L
    let result = context.allocate(result_bytes)
    defer (result.close() catch { _ => () })
    let borrowed = @prediction_main.DeepSeekPredictionAttention::new(
      spec,
      stage=0,
      maximum_main_rows=8,
      maximum_positions=256,
      maximum_workspace_bytes=4194304L,
    )
    defer (borrowed.close() catch { _ => () })
    assert_eq(
      total - borrowed.required_device_bytes(checkpoint, output_borrowed=true),
      result_bytes,
    )
    borrowed.prepare(
      context,
      aot,
      checkpoint,
      root,
      main_counts=main,
      draft_counts=draft,
      main_positions=main,
      draft_positions=draft,
      main_hidden=main,
      draft_hidden=draft,
      reset_flag=main,
      start_position=draft,
      output=result,
      maximum_device_bytes=total - result_bytes,
      chunk_bytes=4093,
      limits~,
    )
    assert_true(borrowed.output().same_storage(result))
    assert_eq(borrowed.prime_launches().length(), 7)
    assert_eq(borrowed.prediction_launches().length(), 19)
    borrowed.close()
    // Closing the branch must not release the enclosing block's result.
    context.validate_allocation_region(
      result,
      offset=0L,
      byte_count=result_bytes,
      alignment=2L,
    )
    result.close()
    draft.close()
    main.close()
    aot.close()
    stream.close()
    context.close()
    assert_eq(live_resources(), baseline)
  }
}
