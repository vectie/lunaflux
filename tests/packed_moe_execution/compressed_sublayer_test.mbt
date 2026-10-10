///|
// Fixture construction is outside the device lane. All model planes retain
// their actual shapes/storage; fake execution checks integration, not numerics.
fn compressed_sublayer_entries(
  spec : @deepseek_spec.DeepSeekV4ModelSpec,
  layer : Int,
) -> Array[(String, String, String, Int64)] raise {
  let entries : Array[(String, String, String, Int64)] = []
  let matrices = @deepseek.query_kv_plan(
    spec,
    maximum_rows=1,
    maximum_workspace_bytes=1048576L,
  ).weight_representations()
  for index, matrix in matrices {
    let name = ["wq_a", "wq_b", "wkv"][index]
    let (rows, columns) = matrix.shape()
    entries.append([
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
  entries.append([
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
  let output = @deepseek.attention_output_plan(
    spec,
    layer~,
    maximum_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  for index, matrix in output.weight_representations() {
    let name = if index == 0 { "wo_a" } else { "wo_b" }
    let (rows, columns) = matrix.shape()
    entries.append([
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
  let learned = spec.compression_ratio(layer) == Some(4)
  for indexer in (if learned { [false, true] } else { [false] }) {
    let pool = @deepseek.compressor_pool_plan(
      spec,
      layer~,
      for_indexer=indexer,
      maximum_rows=3,
      maximum_context=256,
      maximum_state_bytes=16777216L,
      maximum_workspace_bytes=16777216L,
    )
    let (hidden, width, ratio, overlap, _, _) = pool.geometry()
    let channels = width * (if overlap { 2 } else { 1 })
    let prefix = if indexer { "indexer.compressor." } else { "compressor." }
    let sizes = pool.parameter_bytes()
    entries.append([
      (prefix + "wkv.weight", "BF16", "\{channels},\{hidden}", sizes[0]),
      (prefix + "wgate.weight", "BF16", "\{channels},\{hidden}", sizes[1]),
      (prefix + "ape", "F32", "\{ratio},\{channels}", sizes[2]),
      (prefix + "norm.weight", "BF16", "\{width}", sizes[3]),
    ])
  }
  if learned {
    let query = @deepseek.index_queries_plan(
      spec,
      layer~,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    let (rows, columns) = query.query_weight().shape()
    entries.append([
      (
        "indexer.wq_b.weight",
        "F8_E4M3",
        "\{rows},\{columns}",
        query.weight_bytes()[0],
      ),
      (
        "indexer.wq_b.scale",
        "F8_E8M0",
        "\{rows / 128},\{columns / 128}",
        query.weight_bytes()[1],
      ),
      (
        "indexer.weights_proj.weight",
        "BF16",
        "\{spec.index_head_count()},\{spec.hidden_size()}",
        query.weight_bytes()[2],
      ),
    ])
  }
  entries
}

///|
async test "actual-shape compressed sublayers prepare 44 or 25 shared effects with aggregate budgets" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(
      @fs.tmpdir(prefix="lunaflux-compressed-sublayer"),
    )
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @deepseek_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let layers : Array[Int] = []
    for ratio in [4, 128] {
      for layer in 0..<spec.layer_count() {
        if spec.compression_ratio(layer) == Some(ratio) {
          layers.push(layer)
          break
        }
      }
    }
    assert_eq(layers.length(), 2)
    for layer in layers {
      let entries = compressed_sublayer_entries(spec, layer)
      let parts : Array[String] = []
      let mut size = 0L
      for (name, dtype, shape, bytes) in entries {
        parts.push(
          "\"layers.\{layer}.attn.\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{size},\{size + bytes}]}",
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
      let stream = context.create_stream()
      let aot = context.load_module(b"fake\x00")
      let operand = context.allocate(65536L)
      let new_owner = () => {
        @deepseek.DeepSeekCompressedSublayer::new(
          spec,
          layer~,
          maximum_rows=3,
          maximum_context=256,
          maximum_state_bytes=16777216L,
          maximum_workspace_bytes=16777216L,
        )
      }
      let learned = spec.compression_ratio(layer) == Some(4)
      let index_offsets = if learned { Some(operand) } else { None }
      let owner = new_owner()
      let required = owner.required_device_bytes(checkpoint)
      let refused = new_owner()
      let before_refusal = live_resources()
      try
        refused.prepare(
          context,
          aot,
          checkpoint,
          root,
          counts=operand,
          positions=operand,
          reset_flag=operand,
          hidden=operand,
          index_offsets?,
          maximum_device_bytes=required - 1L,
          chunk_bytes=65536,
          limits~,
        )
      catch {
        @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
        error => raise error
      } noraise {
        _ => fail("one-byte aggregate deficit accepted")
      }
      assert_eq(live_resources(), before_refusal)
      refused.close()
      owner.prepare(
        context,
        aot,
        checkpoint,
        root,
        counts=operand,
        positions=operand,
        reset_flag=operand,
        hidden=operand,
        index_offsets?,
        maximum_device_bytes=required,
        chunk_bytes=65536,
        limits~,
      )
      let effects = if learned { 44 } else { 25 }
      assert_eq(owner.launches().length(), effects)
      assert_eq(
        owner.append_descriptors().length(),
        if learned {
          5
        } else {
          3
        },
      )
      assert_eq(
        owner.output().byte_count(),
        3L * spec.hidden_size().to_int64() * 2L,
      )
      let queue = context.create_ordered_kernel_executor(
        stream,
        owner.launches(),
      )
      let submitted = enqueues()
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
      assert_eq(enqueues() - submitted, 32UL * effects.to_uint64())
      assert_eq(blocking_syncs(), waits)
      queue.close()
      owner.close()
      owner.close()
      operand.close()
      aot.close()
      stream.close()
      context.close()
      assert_eq(live_resources(), baseline)
    }
  }
}
