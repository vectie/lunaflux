///|
async test "DeepSeek router checkpoint has no bias in hash layers" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-ds-router")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let header_text =
      #|{"layers.0.ffn.gate.weight":{"dtype":"BF16","shape":[256,4096],"data_offsets":[0,2097152]},"layers.3.ffn.gate.weight":{"dtype":"BF16","shape":[256,4096],"data_offsets":[2097152,4194304]},"layers.3.ffn.gate.bias":{"dtype":"F32","shape":[256],"data_offsets":[4194304,4195328]}}
    let header = @utf8.encode(header_text)
    let file = FixedArray::make(8 + header.length() + 4195328, b'\x38')
    for i in 0..<8 {
      file[i] = (header.length().to_uint64() >> (8 * i)).to_byte()
    }
    for i, byte in header {
      file[8 + i] = byte
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(canonical + "/router", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=5000000L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=3,
      maximum_total_tensors=3,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=2,
      authentication_chunk_bytes=65536,
      maximum_copy_requests=3,
      maximum_copy_bytes=4195328L,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("router"),
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
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let early = @deepseek_v4_packed_expert.bind_router(
      checkpoint,
      spec,
      layer_index=0,
    )
    let later = @deepseek_v4_packed_expert.bind_router(
      checkpoint,
      spec,
      layer_index=3,
    )
    assert_eq(early.length(), 1)
    assert_eq(later.length(), 2)
    assert_eq(early[0].buffer_bytes(), 2097152L)
    assert_eq(later[1].buffer_bytes(), 1024L)
    let bound = early + later
    let copied = [0L, 0L, 0L]
    checkpoint.transfer_weights(
      root,
      bound,
      scratch=FixedArray::make(4093, b'?'),
      limits~,
      consume=(destination, offset, chunk, count) => {
        if offset != copied[destination] ||
          chunk[0] != b'\x38' ||
          chunk[count - 1] != b'\x38' {
          return false
        }
        copied[destination] += count.to_int64()
        true
      },
    )
    assert_eq(copied, [2097152L, 2097152L, 1024L])
  }
}
