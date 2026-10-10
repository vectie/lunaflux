///|
async test "attention and indexer learned pools bind their real BF16 F32 checkpoint planes" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(
      @fs.tmpdir(prefix="lunaflux-compressor-binding"),
    )
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let parts : Array[String] = []
    let sizes : Array[Int64] = []
    let mut offset = 0L
    for indexer in [false, true] {
      let plan = @deepseek_v4_query_kv.compressor_pool_plan(
        spec,
        layer=2,
        for_indexer=indexer,
        maximum_rows=5,
        maximum_state_bytes=1048576L,
        maximum_workspace_bytes=1048576L,
      )
      let (_, width, ratio, _, _, _) = plan.geometry()
      let prefix = "layers.2.attn." +
        (if indexer { "indexer.compressor." } else { "compressor." })
      let fields = [
        ("wkv.weight", "BF16", "\{2 * width},\{spec.hidden_size()}"),
        ("wgate.weight", "BF16", "\{2 * width},\{spec.hidden_size()}"),
        ("ape", "F32", "\{ratio},\{2 * width}"),
        ("norm.weight", "BF16", "\{width}"),
      ]
      for index, (name, dtype, shape) in fields {
        let bytes = plan.parameter_bytes()[index]
        parts.push(
          "\"\{prefix}\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset + bytes}]}",
        )
        offset += bytes
        sizes.push(bytes)
      }
    }
    let header = @utf8.encode("{" + parts.join(",") + "}")
    let file = FixedArray::make(8 + header.length() + offset.to_int(), b'\x38')
    for index in 0..<8 {
      file[index] = (header.length().to_uint64() >> (8 * index)).to_byte()
    }
    for index, byte in header {
      file[8 + index] = byte
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/compressors", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=offset + 4096L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=8,
      maximum_total_tensors=8,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=2,
      maximum_copy_requests=8,
      maximum_copy_bytes=offset,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("compressors"),
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
    for ordinal, indexer in [false, true] {
      let bound = @deepseek_v4_query_kv.bind_compressor_pool_weights(
        checkpoint,
        spec,
        layer=2,
        for_indexer=indexer,
      )
      assert_eq(
        bound.map(weight => weight.buffer_bytes()),
        sizes[ordinal * 4:ordinal * 4 + 4].to_owned(),
      )
      let plan = @deepseek_v4_query_kv.compressor_pool_plan(
        spec,
        layer=2,
        for_indexer=indexer,
        maximum_rows=5,
        maximum_state_bytes=1048576L,
        maximum_workspace_bytes=1048576L,
      )
      let owner = @deepseek_v4_query_kv.DeepSeekCompressorPool::new(
        spec,
        layer=2,
        for_indexer=indexer,
        maximum_rows=5,
        maximum_state_bytes=1048576L,
        maximum_workspace_bytes=1048576L,
      )
      assert_eq(
        owner.required_device_bytes(checkpoint),
        plan.state_bytes().fold(init=0L, (total, bytes) => total + bytes) +
        plan.frame_bytes().fold(init=0L, (total, bytes) => total + bytes) +
        bound.fold(init=0L, (total, weight) => total + weight.buffer_bytes()),
      )
      owner.close()
      owner.close()
      let copied = [0L, 0L, 0L, 0L]
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
    }
    try
      @deepseek_v4_query_kv.bind_compressor_pool_weights(
        checkpoint,
        spec,
        layer=3,
      )
    catch {
      @materialize.PackedCheckpointError::MissingTensor(_) => ()
      error => raise error
    } noraise {
      _ => fail("missing compressor layer replaced with fixture data")
    }
  }
}
