///|
async test "learned head binds and streams exact F32 checkpoint controls" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-learned-head"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let header = @utf8.encode(
      (
        #|{"hc_head_fn":{"dtype":"F32","shape":[4,16384],"data_offsets":[0,262144]},"hc_head_base":{"dtype":"F32","shape":[4],"data_offsets":[262144,262160]},"hc_head_scale":{"dtype":"F32","shape":[1],"data_offsets":[262160,262164]}}
      ),
    )
    let file = FixedArray::make(8 + header.length() + 262164, b'\x38')
    for i in 0..<8 {
      file[i] = (header.length().to_uint64() >> (8 * i)).to_byte()
    }
    for i, byte in header {
      file[8 + i] = byte
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/controls", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=300000L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=3,
      maximum_total_tensors=3,
      maximum_rank=2,
      maximum_name_bytes=128,
      maximum_dtype_specs=1,
      maximum_copy_requests=3,
      maximum_copy_bytes=262164L,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("controls"),
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
            ),
          ),
        ],
        dtype_specs=[@streaming.StreamingSafetensorsDTypeSpec::float32()],
        limits~,
      ),
    )
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    let bound = @deepseek_v4_text_io.bind_head_control_weights(checkpoint, spec)
    assert_eq(bound.length(), 3)
    assert_eq(bound.map(weight => weight.buffer_bytes()), [262144L, 16L, 4L])
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
    assert_eq(copied, [262144L, 16L, 4L])
    try @deepseek_v4_text_io.bind_text_ingress_weight(checkpoint, spec) catch {
      @materialize.PackedCheckpointError::MissingTensor("embed.weight") => ()
      e => raise e
    } noraise {
      _ => fail("missing official embedding accepted")
    }
    // A missing final norm/head cannot silently become a mean/fixture head.
    try @deepseek_v4_text_io.bind_text_io_weights(checkpoint, spec) catch {
      @materialize.PackedCheckpointError::MissingTensor("norm.weight") => ()
      e => raise e
    } noraise {
      _ => fail("incomplete output checkpoint accepted")
    }
  }
}
