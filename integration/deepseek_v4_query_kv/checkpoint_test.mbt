///|
async test "official query and KV planes bind and stream without expansion" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-query-kv"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let plan = @deepseek_v4_query_kv.query_kv_plan(
      spec,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    let matrices = plan.weight_representations()
    let output_plan = @deepseek_v4_query_kv.attention_output_plan(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    let output_matrices = output_plan.weight_representations()
    let shapes : Array[String] = []
    for matrix in matrices {
      let (rows, columns) = matrix.shape()
      shapes.append(["\{rows},\{columns}", "\{rows / 128},\{columns / 128}"])
    }
    let fields = [
      ("wq_a.weight", "F8_E4M3", shapes[0]),
      ("wq_a.scale", "F8_E8M0", shapes[1]),
      ("q_norm.weight", "BF16", "\{spec.q_lora_rank()}"),
      ("wq_b.weight", "F8_E4M3", shapes[2]),
      ("wq_b.scale", "F8_E8M0", shapes[3]),
      ("wkv.weight", "F8_E4M3", shapes[4]),
      ("wkv.scale", "U8", shapes[5]),
      ("kv_norm.weight", "BF16", "\{spec.head_dimension()}"),
      ("attn_sink", "F32", "\{spec.attention_head_count()}"),
    ]
    let sizes = plan.weight_bytes()[:].to_owned()
    let prefix_weight_bytes = sizes.fold(init=0L, (total, bytes) => {
      total + bytes
    })
    let sink_bytes = spec.attention_head_count().to_int64() * 4L
    sizes.push(sink_bytes)
    for index, matrix in output_matrices {
      let (rows, columns) = matrix.shape()
      let name = if index == 0 { "wo_a" } else { "wo_b" }
      fields.append([
        (name + ".weight", "F8_E4M3", "\{rows},\{columns}"),
        (name + ".scale", "F8_E8M0", "\{rows / 128},\{columns / 128}"),
      ])
      sizes.append([matrix.payload_bytes(), matrix.scale_bytes()])
    }
    let parts : Array[String] = []
    let mut offset = 0L
    for i, (name, dtype, shape) in fields {
      parts.push(
        "\"layers.0.attn.\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset + sizes[i]}]}",
      )
      offset += sizes[i]
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
    @fs.write_file(directory + "/queries", bytes, create_mode=CreateNew)
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
      maximum_dtype_specs=5,
      authentication_chunk_bytes=65536,
      maximum_copy_requests=8,
      maximum_copy_bytes=offset,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("queries"),
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
          @streaming.StreamingSafetensorsDTypeSpec::opaque_one_byte("U8"),
        ],
        limits~,
      ),
    )
    let bound = @deepseek_v4_query_kv.bind_query_kv_weights(checkpoint, spec, 0)
    assert_eq(bound.map(weight => weight.buffer_bytes()), [
      sizes[0] + sizes[1],
      sizes[2],
      sizes[3] + sizes[4],
      sizes[5] + sizes[6],
      sizes[7],
    ])
    let owner = @deepseek_v4_query_kv.DeepSeekQueryKv::new(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    assert_eq(
      owner.required_device_bytes(checkpoint),
      prefix_weight_bytes + plan.workspace_bytes(),
    )
    owner.close()
    let inputs = @deepseek_v4_query_kv.DeepSeekAttentionInputs::new(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    let rotary = @deepseek_v4_query_kv.rotary_kv_plan(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    assert_eq(
      inputs.required_device_bytes(checkpoint),
      prefix_weight_bytes + plan.workspace_bytes() + rotary.workspace_bytes(),
    )
    inputs.close()
    inputs.close()
    assert_eq(
      @deepseek_v4_query_kv.bind_window_attention_sink(
        checkpoint,
        spec,
        layer=0,
      ).buffer_bytes(),
      sink_bytes,
    )
    let attention = @deepseek_v4_query_kv.DeepSeekWindowAttention::new(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    let window = @deepseek_v4_query_kv.window_attention_plan(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_state_bytes=1048576L,
    )
    assert_eq(
      attention.required_device_bytes(checkpoint),
      prefix_weight_bytes +
      sink_bytes +
      plan.workspace_bytes() +
      rotary.workspace_bytes() +
      window.state_bytes(),
    )
    attention.close()
    let output_weights = @deepseek_v4_query_kv.bind_attention_output_weights(
      checkpoint,
      spec,
      layer=0,
    )
    let output_weight_bytes = output_weights.fold(init=0L, (total, weight) => {
      total + weight.buffer_bytes()
    })
    assert_eq(output_weight_bytes, sizes[9] + sizes[10] + sizes[11] + sizes[12])
    let output = @deepseek_v4_query_kv.DeepSeekAttentionOutput::new(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    assert_eq(
      output.required_device_bytes(checkpoint),
      output_weight_bytes + output_plan.workspace_bytes(),
    )
    output.close()
    let sublayer = @deepseek_v4_query_kv.DeepSeekWindowSublayer::new(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    assert_eq(
      sublayer.required_device_bytes(checkpoint),
      offset +
      plan.workspace_bytes() +
      rotary.workspace_bytes() +
      window.state_bytes() +
      output_plan.workspace_bytes(),
    )
    sublayer.close()
    let copied = [0L, 0L, 0L, 0L, 0L]
    checkpoint.transfer_weights(
      root,
      bound,
      scratch=FixedArray::make(4093, b'?'),
      limits~,
      consume=(destination, position, chunk, count) => {
        if position != copied[destination] ||
          chunk[0] != b'\x38' ||
          chunk[count - 1] != b'\x38' {
          return false
        }
        copied[destination] += count.to_int64()
        true
      },
    )
    assert_eq(copied, bound.map(weight => weight.buffer_bytes()))
    try @deepseek_v4_query_kv.bind_query_kv_weights(checkpoint, spec, 1) catch {
      @materialize.PackedCheckpointError::MissingTensor(
        "layers.1.attn.wq_a.weight"
      ) => ()
      error => raise error
    } noraise {
      _ => fail("missing layer replaced with fixture parameters")
    }
    try
      @deepseek_v4_query_kv.bind_query_kv_weights(
        checkpoint,
        spec,
        spec.layer_count(),
      )
    catch {
      @precision.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("out-of-range layer accepted")
    }
  }
}
