///|
async test "startup transfer session opens a referenced shard once and closes explicitly" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-transfer-session"))
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let shard = streaming_test_shard0()
    @fs.write_file(directory + "/weights", shard, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = streaming_test_limits()
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [streaming_test_input("weights", shard)],
      dtype_specs=streaming_test_specs(),
      limits~,
    )
    let session = @streaming_safetensors.StreamingSafetensorsTransferSession::new(
      root, inspection, limits,
    )
    defer session.close()
    let request = @streaming_safetensors.StreamingSafetensorsTransfer::new(
      shard_ordinal=0,
      tensor_ordinal=0,
      tensor_byte_offset=0L,
      destination_ordinal=0,
      destination_offset=0L,
      byte_count=4L,
    )
    let scratch = FixedArray::make(2, b'?')
    let bytes : Array[Byte] = []
    for _ in 0..<8 {
      session.transfer_ranges(
        root,
        requests=[request],
        destination_bytes=[4L],
        scratch~,
        limits~,
        consume=(_, _, buffer, count) => {
          for i in 0..<count {
            bytes.push(buffer[i])
          }
          true
        },
      )
    }
    assert_eq(session.opened_shards(), 1)
    assert_eq(Bytes::from_array(bytes), b"abcdabcdabcdabcdabcdabcdabcdabcd")
    // Another approved root object is a cache miss, not an identity shortcut.
    let other = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer other.close()
    session.transfer_ranges(
      other,
      requests=[request],
      destination_bytes=[4L],
      scratch~,
      limits~,
      consume=(_, _, _, _) => true,
    )
    assert_eq(session.opened_shards(), 1)
    session.close()
    session.close()
    try
      session.transfer_ranges(
        root,
        requests=[request],
        destination_bytes=[4L],
        scratch~,
        limits~,
        consume=(_, _, _, _) => true,
      )
    catch {
      @streaming_safetensors.StreamingSafetensorsError::InvalidLimit => ()
      e => raise e
    } noraise {
      _ => fail("closed startup transfer session reused")
    }
  }
}

///|
async test "startup session retires after a rejected upload chunk" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(
      @fs.tmpdir(prefix="lunaflux-transfer-session-fail"),
    )
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let shard = streaming_test_shard0()
    @fs.write_file(directory + "/weights", shard, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = streaming_test_limits()
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [streaming_test_input("weights", shard)],
      dtype_specs=streaming_test_specs(),
      limits~,
    )
    let session = @streaming_safetensors.StreamingSafetensorsTransferSession::new(
      root, inspection, limits,
    )
    defer session.close()
    let request = @streaming_safetensors.StreamingSafetensorsTransfer::new(
      shard_ordinal=0,
      tensor_ordinal=0,
      tensor_byte_offset=0L,
      destination_ordinal=0,
      destination_offset=0L,
      byte_count=4L,
    )
    let calls = Ref(0)
    try
      session.transfer_ranges(
        root,
        requests=[request],
        destination_bytes=[4L],
        scratch=FixedArray::make(1, b'?'),
        limits~,
        consume=(_, _, _, _) => {
          calls.val += 1
          false
        },
      )
    catch {
      @streaming_safetensors.StreamingSafetensorsError::VisitorRejected(0) => ()
      e => raise e
    } noraise {
      _ => fail("rejected upload chunk continued")
    }
    assert_eq(calls.val, 1)
    session.close()
    session.close()
  }
}
