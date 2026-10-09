///|
async test "quantized checkpoint scalar scales contain one element, not zero" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-scalar-scale")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let shard = streaming_test_file(
      (
        #|{"weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}
      ),
      b"\x00\x00\x80\x3f",
    )
    @fs.write_file(canonical + "/scales", shard, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = streaming_test_limits()
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [streaming_test_input("scales", shard)],
      dtype_specs=streaming_test_specs(),
      limits~,
    )
    guard inspection.shard(0) is Some(metadata) &&
      metadata.tensor(0) is Some(scale) else {
      fail("missing scalar")
    }
    assert_eq(scale.shape().length(), 0)
    assert_eq(scale.byte_count(), 4L)
    let request = @streaming_safetensors.StreamingSafetensorsTransfer::new(
      shard_ordinal=0,
      tensor_ordinal=0,
      tensor_byte_offset=0L,
      destination_ordinal=0,
      destination_offset=0L,
      byte_count=4L,
    )
    let payload : Array[Byte] = []
    @streaming_safetensors.transfer_ranges(
      root,
      inspection,
      requests=[request],
      destination_bytes=[4L],
      scratch=FixedArray::make(3, b'!'),
      limits~,
      consume=(_, _, buffer, count) => {
        for i in 0..<count {
          payload.push(buffer[i])
        }
        true
      },
    )
    assert_eq(Bytes::from_array(payload), b"\x00\x00\x80\x3f")
    root.close()
  }
}
