///|
async test "segmented copy authenticates shards once and fills distinct arenas" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-streaming-segmented")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let shard0 = streaming_test_shard0()
    let shard1 = streaming_test_shard1()
    @fs.write_file(canonical + "/s0", shard0, create_mode=CreateNew)
    @fs.write_file(canonical + "/s1", shard1, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = streaming_test_limits()
    let inspection = inspect_shards(
      root,
      [streaming_test_input("s0", shard0), streaming_test_input("s1", shard1)],
      dtype_specs=streaming_test_specs(),
      limits~,
    )
    let arenas = [FixedArray::make(5, b'!'), FixedArray::make(6, b'?')]
    let requests = [
      try! StreamingSafetensorsSegmentedCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=0,
        tensor_byte_offset=0L,
        destination_arena_ordinal=0,
        destination_offset=1,
        byte_count=4,
      ),
      try! StreamingSafetensorsSegmentedCopyRequest::new(
        shard_ordinal=1,
        tensor_ordinal=1,
        tensor_byte_offset=2L,
        destination_arena_ordinal=1,
        destination_offset=1,
        byte_count=4,
      ),
    ]
    let visited : Array[StreamingSafetensorsSegmentedVisitedRange] = []
    copy_segmented_ranges(
      root,
      inspection,
      destination_arenas=arenas,
      requests~,
      limits~,
      visit=(_, range) => visited.push(range),
    )
    assert_eq(arenas[0], FixedArray::from_array([b'!', b'a', b'b', b'c', b'd']))
    assert_eq(
      arenas[1],
      FixedArray::from_array([b'?', b'o', b'p', b'q', b'r', b'?']),
    )
    assert_eq(visited.length(), 2)
    assert_eq(visited[1].destination_arena_ordinal(), 1)
    assert_eq(visited[1].destination_offset(), 1)
    root.close()
  }
}

///|
async test "segmented copy rejects arena bounds and only overlaps within an arena" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-streaming-segmented-hostile")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let bytes = streaming_test_shard0()
    @fs.write_file(canonical + "/s0", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = streaming_test_limits()
    let inspection = inspect_shards(
      root,
      [streaming_test_input("s0", bytes)],
      dtype_specs=streaming_test_specs(),
      limits~,
    )
    let arenas = [FixedArray::make(4, b'!'), FixedArray::make(4, b'?')]
    let distinct_arenas = [
      try! StreamingSafetensorsSegmentedCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=0,
        tensor_byte_offset=0L,
        destination_arena_ordinal=0,
        destination_offset=0,
        byte_count=4,
      ),
      try! StreamingSafetensorsSegmentedCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=1,
        tensor_byte_offset=0L,
        destination_arena_ordinal=1,
        destination_offset=0,
        byte_count=3,
      ),
    ]
    copy_segmented_ranges(
      root,
      inspection,
      destination_arenas=arenas,
      requests=distinct_arenas,
      limits~,
      visit=(_, _) => (),
    )
    let outside = [
      try! StreamingSafetensorsSegmentedCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=0,
        tensor_byte_offset=0L,
        destination_arena_ordinal=2,
        destination_offset=0,
        byte_count=4,
      ),
    ]
    try
      copy_segmented_ranges(
        root,
        inspection,
        destination_arenas=arenas,
        requests=outside,
        limits~,
        visit=(_, _) => (),
      )
    catch {
      InvalidDestinationArena(0) => ()
      _ => fail("unexpected invalid-arena rejection")
    } noraise {
      _ => fail("out-of-range destination arena was admitted")
    }
    let overlap = [
      distinct_arenas[0],
      try! StreamingSafetensorsSegmentedCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=1,
        tensor_byte_offset=0L,
        destination_arena_ordinal=0,
        destination_offset=1,
        byte_count=3,
      ),
    ]
    try
      copy_segmented_ranges(
        root,
        inspection,
        destination_arenas=arenas,
        requests=overlap,
        limits~,
        visit=(_, _) => (),
      )
    catch {
      OverlappingDestinationRanges(0, 1) => ()
      _ => fail("unexpected segmented overlap rejection")
    } noraise {
      _ => fail("same-arena overlap was admitted")
    }
    root.close()
  }
}
