///|
async test "learned index queries bind and stream their compact checkpoint banks" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-index-queries"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let plan = @deepseek_v4_query_kv.index_queries_plan(
      spec,
      layer=2,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    let sizes = plan.weight_bytes()
    let (rows, columns) = plan.query_weight().shape()
    let fields = [
      ("wq_b.weight", "F8_E4M3", "\{rows},\{columns}"),
      ("wq_b.scale", "F8_E8M0", "\{rows / 128},\{columns / 128}"),
      (
        "weights_proj.weight",
        "BF16",
        "\{spec.index_head_count()},\{spec.hidden_size()}",
      ),
    ]
    let parts : Array[String] = []
    let mut offset = 0L
    for index, (name, dtype, shape) in fields {
      parts.push(
        "\"layers.2.attn.indexer.\{name}\":{\"dtype\":\"\{dtype}\",\"shape\":[\{shape}],\"data_offsets\":[\{offset},\{offset + sizes[index]}]}",
      )
      offset += sizes[index]
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
    @fs.write_file(directory + "/index-queries", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=offset + 4096L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=3,
      maximum_total_tensors=3,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=3,
      maximum_copy_requests=3,
      maximum_copy_bytes=offset,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("index-queries"),
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
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
    let bound = @deepseek_v4_query_kv.bind_index_queries_weights(
      checkpoint,
      spec,
      layer=2,
    )
    assert_eq(bound.map(weight => weight.buffer_bytes()), [
      sizes[0] + sizes[1],
      sizes[2],
    ])
    let owner = @deepseek_v4_query_kv.DeepSeekIndexQueries::new(
      spec,
      layer=2,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    assert_eq(
      owner.required_device_bytes(checkpoint),
      offset + plan.workspace_bytes(),
    )
    owner.close()
    owner.close()
    let copied = [0L, 0L]
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
    try
      @deepseek_v4_query_kv.bind_index_queries_weights(
        checkpoint,
        spec,
        layer=4,
      )
    catch {
      @materialize.PackedCheckpointError::MissingTensor(_) => ()
      error => raise error
    } noraise {
      _ => fail("missing learned query bank was fabricated")
    }
  }
}
