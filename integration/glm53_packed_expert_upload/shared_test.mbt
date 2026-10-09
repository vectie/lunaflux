///|
async test "GLM BF16 shared checkpoint binds and streams without invented scales" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-glm-shared")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let weights = @glm53_weights.shared_expert_weights(layer_index=3)
    let entries : Array[String] = []
    let mut cursor = 0L
    for weight in weights {
      let rep = weight.representation()
      let (rows, columns) = rep.shape()
      let bytes = rep.payload_bytes()
      entries.push(
        "\"\{weight.payload_name()}\":{\"dtype\":\"BF16\",\"shape\":[\{rows},\{columns}],\"data_offsets\":[\{cursor},\{cursor + bytes}]}",
      )
      cursor += bytes
    }
    let header = @utf8.encode("{" + entries.join(",") + "}")
    let file = FixedArray::make(8 + header.length() + cursor.to_int(), b'\x38')
    for i in 0..<8 {
      file[i] = (header.length().to_uint64() >> (8 * i)).to_byte()
    }
    for i, byte in header {
      file[8 + i] = byte
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(canonical + "/shared", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    defer root.close()
    let limits = @streaming_safetensors.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=60000000L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=3,
      maximum_total_tensors=3,
      maximum_rank=2,
      maximum_name_bytes=256,
      maximum_dtype_specs=1,
      authentication_chunk_bytes=65536,
      maximum_copy_requests=3,
      maximum_copy_bytes=50331648L,
    )
    let checkpoint = @precision_materialize.PackedCheckpoint::new(
      @streaming_safetensors.inspect_shards(
        root,
        [
          @streaming_safetensors.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("shared"),
            expected_digest=@streaming_safetensors.StreamingSafetensorsFileDigest::from_sha256(
              @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
            ),
          ),
        ],
        dtype_specs=[
          @streaming_safetensors.StreamingSafetensorsDTypeSpec::bfloat16(),
        ],
        limits~,
      ),
    )
    let bound = @glm53_packed_expert_upload.bind_shared_expert(
      checkpoint,
      layer_index=3,
    )
    assert_eq(bound.length(), 3)
    for weight in bound {
      assert_eq(weight.buffer_bytes(), 16777216L)
      assert_eq(weight.representation().scale_bytes(), 0L)
    }
    let copied = [0L, 0L, 0L]
    checkpoint.transfer_weights(
      root,
      bound,
      scratch=FixedArray::make(4093, b'?'),
      limits~,
      consume=(destination, offset, buffer, count) => {
        if offset != copied[destination] ||
          buffer[0] != b'\x38' ||
          buffer[count - 1] != b'\x38' {
          return false
        }
        copied[destination] += count.to_int64()
        true
      },
    )
    assert_eq(copied, [16777216L, 16777216L, 16777216L])
  }
}
