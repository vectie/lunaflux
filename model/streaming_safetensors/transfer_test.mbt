///|
async test "streamed slices preserve packed bytes and 64-bit device offsets" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-transfer")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let shard = streaming_test_shard0()
    @fs.write_file(canonical + "/weights", shard, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = streaming_test_limits()
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [streaming_test_input("weights", shard)],
      dtype_specs=streaming_test_specs(),
      limits~,
    )
    let base = 4294967296L
    let requests = [
      @streaming_safetensors.StreamingSafetensorsTransfer::new(
        shard_ordinal=0,
        tensor_ordinal=0,
        tensor_byte_offset=1L,
        destination_ordinal=0,
        destination_offset=base,
        byte_count=3L,
      ),
      @streaming_safetensors.StreamingSafetensorsTransfer::new(
        shard_ordinal=0,
        tensor_ordinal=1,
        tensor_byte_offset=0L,
        destination_ordinal=1,
        destination_offset=0L,
        byte_count=3L,
      ),
    ]
    let offsets : Array[Int64] = []
    let payload : Array[Byte] = []
    let destinations : Array[Int] = []
    @streaming_safetensors.transfer_ranges(
      root,
      inspection,
      requests~,
      destination_bytes=[base + 3L, 3L],
      scratch=FixedArray::make(2, b'!'),
      limits~,
      consume=(destination, offset, bytes, count) => {
        offsets.push(offset)
        destinations.push(destination)
        for i in 0..<count {
          payload.push(bytes[i])
        }
        true
      },
    )
    assert_eq(Bytes::from_array(payload), b"bcdXYZ")
    assert_eq(offsets, [base, base + 2L, 0L, 2L])
    assert_eq(destinations, [0, 0, 1, 1])
    root.close()
  }
}

///|
async test "streaming stops on a rejected chunk without reading subsequent slices" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-transfer-failure")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let shard = streaming_test_shard0()
    @fs.write_file(canonical + "/weights", shard, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = streaming_test_limits()
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [streaming_test_input("weights", shard)],
      dtype_specs=streaming_test_specs(),
      limits~,
    )
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
      @streaming_safetensors.transfer_ranges(
        root,
        inspection,
        requests=[request],
        destination_bytes=[4L],
        scratch=FixedArray::make(1, b'!'),
        limits~,
        consume=(_, _, _, _) => {
          calls.val += 1
          false
        },
      )
    catch {
      VisitorRejected(0) => ()
      _ => fail("wrong streaming failure")
    } noraise {
      _ => fail("rejected chunk continued")
    }
    assert_eq(calls.val, 1)
    try
      @streaming_safetensors.transfer_ranges(
        root,
        inspection,
        requests=[request],
        destination_bytes=[3L],
        scratch=FixedArray::make(1, b'!'),
        limits~,
        consume=(_, _, _, _) => {
          calls.val += 1
          true
        },
      )
    catch {
      CopyRangeOutsideDestination(0) => ()
      _ => fail("wrong destination failure")
    } noraise {
      _ => fail("short destination accepted")
    }
    assert_eq(calls.val, 1)
    root.close()
  }
}
