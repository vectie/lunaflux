///|
fn streaming_test_append_u64(output : Array[Byte], value : UInt64) -> Unit {
  for shift in 0..<8 {
    output.push(((value >> (shift * 8)) & 0xffUL).to_byte())
  }
}

///|
fn streaming_test_file(header : String, payload : BytesView) -> Bytes {
  let encoded = @utf8.encode(header)
  let output : Array[Byte] = []
  streaming_test_append_u64(output, encoded.length().to_uint64())
  for byte in encoded {
    output.push(byte)
  }
  for byte in payload {
    output.push(byte)
  }
  Bytes::from_array(output)
}

///|
fn streaming_test_digest(bytes : Bytes) -> StreamingSafetensorsFileDigest {
  try! StreamingSafetensorsFileDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn streaming_test_input(
  name : StringView,
  bytes : Bytes,
) -> StreamingSafetensorsShardInput {
  StreamingSafetensorsShardInput::new(
    locator=try! @approved_fs.ApprovedRelativeLocator::new(name),
    expected_digest=streaming_test_digest(bytes),
  )
}

///|
fn streaming_test_specs() -> Array[StreamingSafetensorsDTypeSpec] {
  [
    StreamingSafetensorsDTypeSpec::bfloat16(),
    StreamingSafetensorsDTypeSpec::float32(),
    try! StreamingSafetensorsDTypeSpec::opaque_one_byte("F8_E4M3"),
    try! StreamingSafetensorsDTypeSpec::opaque_fixed_width(
      "CUSTOM3",
      bytes_per_element=3L,
    ),
  ]
}

///|
fn streaming_test_limits() -> StreamingSafetensorsLimits {
  try! StreamingSafetensorsLimits::new(
    maximum_shards=8,
    maximum_file_bytes=1048576L,
    maximum_header_bytes=65536,
    maximum_tensors_per_shard=16,
    maximum_total_tensors=64,
    maximum_rank=8,
    maximum_name_bytes=128,
    maximum_dtype_specs=16,
    maximum_copy_requests=16,
    maximum_copy_bytes=1024L,
  )
}

///|
fn streaming_test_shard0() -> Bytes {
  streaming_test_file(
    (
      #|{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[4,7]}}
    ),
    b"abcdXYZ",
  )
}

///|
fn streaming_test_shard1(name? : String = "c") -> Bytes {
  streaming_test_file(
    "{\"" +
    name +
    "\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"d\":{\"dtype\":\"CUSTOM3\",\"shape\":[2],\"data_offsets\":[4,10]}}",
    b"1234mnopqr",
  )
}

///|
async test "two shards inspect and copy Int64 ranges into final spans" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-streaming-safetensors")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let shard0 = streaming_test_shard0()
    let shard1 = streaming_test_shard1()
    @fs.write_file(canonical + "/s0", shard0, create_mode=CreateNew)
    @fs.write_file(canonical + "/s1", shard1, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let inputs = [
      streaming_test_input("s0", shard0),
      streaming_test_input("s1", shard1),
    ]
    let same_shard = root.open_file(inputs[0].locator())
    assert_eq(same_shard.stamp().byte_count(), shard0.length().to_int64())
    same_shard.close()
    let limits = streaming_test_limits()
    let inspection = inspect_shards(
      root,
      inputs,
      dtype_specs=streaming_test_specs(),
      limits~,
    )
    assert_eq(inspection.shard_count(), 2)
    assert_eq(inspection.total_tensor_count(), 4)
    guard inspection.shard(0) is Some(first) &&
      first.tensor(1) is Some(opaque_tensor) else {
      fail("inspected shard metadata missing")
    }
    assert_eq(first.file_byte_count(), shard0.length().to_int64())
    assert_eq(opaque_tensor.name(), "b")
    assert_eq(opaque_tensor.dtype().kind(), OpaqueFixedWidth)
    assert_eq(opaque_tensor.dtype().bytes_per_element(), 1L)
    let destination = FixedArray::make(9, b'!')
    let requests = [
      try! StreamingSafetensorsCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=0,
        tensor_byte_offset=0L,
        destination_offset=1,
        byte_count=4,
      ),
      try! StreamingSafetensorsCopyRequest::new(
        shard_ordinal=1,
        tensor_ordinal=1,
        tensor_byte_offset=3L,
        destination_offset=5,
        byte_count=3,
      ),
    ]
    let visited : Array[StreamingSafetensorsVisitedRange] = []
    copy_ranges(root, inspection, destination~, requests~, limits~, visit=(
      tensor,
      range,
    ) => {
      if tensor.name() != "a" && tensor.name() != "d" {
        raise VisitorRejected(99)
      }
      visited.push(range)
    })
    assert_eq(
      destination,
      FixedArray::from_array([
        b'!', b'a', b'b', b'c', b'd', b'p', b'q', b'r', b'!',
      ]),
    )
    assert_eq(visited.length(), 2)
    assert_eq(visited[1].tensor_byte_offset(), 3L)
    assert_eq(visited[1].destination_offset(), 5)
    root.close()
  }
}

///|
async test "cross-shard duplicates fail but inventory labels are not payload checks" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-streaming-hostile-shards")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let shard0 = streaming_test_shard0()
    let duplicate = streaming_test_shard1(name="a")
    @fs.write_file(canonical + "/s0", shard0, create_mode=CreateNew)
    @fs.write_file(canonical + "/dup", duplicate, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    try
      inspect_shards(
        root,
        [
          streaming_test_input("s0", shard0),
          streaming_test_input("dup", duplicate),
        ],
        dtype_specs=streaming_test_specs(),
        limits=streaming_test_limits(),
      )
    catch {
      DuplicateTensorName(
        first_shard=0,
        first_tensor=0,
        second_shard=1,
        second_tensor=0
      ) => ()
      _ => fail("unexpected cross-shard duplicate rejection")
    } noraise {
      _ => fail("duplicate tensor name crossed shard boundary")
    }
    let wrong_digest = StreamingSafetensorsShardInput::new(
      locator=try! @approved_fs.ApprovedRelativeLocator::new("s0"),
      expected_digest=try! StreamingSafetensorsFileDigest::from_sha256(
        String::make(64, '0'),
      ),
    )
    let inspection = inspect_shards(
      root,
      [wrong_digest],
      dtype_specs=streaming_test_specs(),
      limits=streaming_test_limits(),
    )
    assert_eq(inspection.shard_count(), 1)
    let destination = FixedArray::make(4, b'?')
    copy_ranges(
      root,
      inspection,
      destination~,
      requests=[
        StreamingSafetensorsCopyRequest::new(
          shard_ordinal=0,
          tensor_ordinal=0,
          tensor_byte_offset=0L,
          destination_offset=0,
          byte_count=4,
        ),
      ],
      limits=streaming_test_limits(),
      visit=(_, _) => (),
    )
    assert_eq(Bytes::from_array(destination), b"abcd")
    root.close()
  }
}

///|
async test "hostile dtype size overlap and header mutations fail closed" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-streaming-hostile-header")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let unknown = streaming_test_file(
      (
        #|{"x":{"dtype":"UNKNOWN","shape":[1],"data_offsets":[0,1]}}
      ),
      b"x",
    )
    let wrong_size = streaming_test_file(
      (
        #|{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}}
      ),
      b"abc",
    )
    let overlap = streaming_test_file(
      (
        #|{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}
      ),
      b"abcdef",
    )
    let fixtures = [
      ("unknown", unknown),
      ("wrong", wrong_size),
      ("overlap", overlap),
    ]
    for item in fixtures {
      let (name, bytes) = item
      @fs.write_file(canonical + "/" + name, bytes, create_mode=CreateNew)
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = streaming_test_limits()
    let specs = streaming_test_specs()
    try
      inspect_shards(
        root,
        [streaming_test_input("unknown", unknown)],
        dtype_specs=specs,
        limits~,
      )
    catch {
      UnknownDType(shard=0, tensor=0) => ()
      _ => fail("unexpected unknown dtype rejection")
    } noraise {
      _ => fail("unknown dtype was admitted")
    }
    try
      inspect_shards(
        root,
        [streaming_test_input("wrong", wrong_size)],
        dtype_specs=specs,
        limits~,
      )
    catch {
      RangeSizeMismatch(shard=0, tensor=0) => ()
      _ => fail("unexpected width mismatch rejection")
    } noraise {
      _ => fail("wrong F32 byte count was admitted")
    }
    try
      inspect_shards(
        root,
        [streaming_test_input("overlap", overlap)],
        dtype_specs=specs,
        limits~,
      )
    catch {
      OverlappingTensorRanges(shard=0, first=0, second=1) => ()
      _ => fail("unexpected overlap rejection")
    } noraise {
      _ => fail("overlapping payload ranges were admitted")
    }
    root.close()
  }
}

///|
async test "copy bounds destination overlap and visitor errors fail closed" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-streaming-hostile-copy")
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
    let destination = FixedArray::make(8, b'!')
    let overlap = [
      try! StreamingSafetensorsCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=0,
        tensor_byte_offset=0L,
        destination_offset=0,
        byte_count=4,
      ),
      try! StreamingSafetensorsCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=1,
        tensor_byte_offset=0L,
        destination_offset=3,
        byte_count=3,
      ),
    ]
    try
      copy_ranges(root, inspection, destination~, requests=overlap, limits~, visit=(
        _,
        _,
      ) => ())
    catch {
      OverlappingDestinationRanges(0, 1) => ()
      _ => fail("unexpected destination overlap rejection")
    } noraise {
      _ => fail("overlapping final spans were copied")
    }
    let outside = [
      try! StreamingSafetensorsCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=1,
        tensor_byte_offset=2L,
        destination_offset=0,
        byte_count=2,
      ),
    ]
    try
      copy_ranges(root, inspection, destination~, requests=outside, limits~, visit=(
        _,
        _,
      ) => ())
    catch {
      CopyRangeOutsideTensor(0) => ()
      _ => fail("unexpected tensor-range rejection")
    } noraise {
      _ => fail("copy escaped tensor payload range")
    }
    let exact = [
      try! StreamingSafetensorsCopyRequest::new(
        shard_ordinal=0,
        tensor_ordinal=0,
        tensor_byte_offset=0L,
        destination_offset=0,
        byte_count=4,
      ),
    ]
    try
      copy_ranges(root, inspection, destination~, requests=exact, limits~, visit=(
        _,
        _,
      ) => raise VisitorRejected(0))
    catch {
      VisitorRejected(0) => ()
      _ => fail("unexpected visitor rejection")
    } noraise {
      _ => fail("visitor failure was suppressed")
    }
    // A second operation succeeds, proving the failed visitor retained no
    // open-file authority or poisoned lifecycle state.
    copy_ranges(root, inspection, destination~, requests=exact, limits~, visit=(
      _,
      _,
    ) => ())
    root.close()
  }
}

///|
test "dtype digest and startup constructors reject hostile values" {
  for tag in ["", "BF16", "F32", "bad tag"] {
    try StreamingSafetensorsDTypeSpec::opaque_one_byte(tag) catch {
      InvalidDTypeSpec => ()
      _ => fail("unexpected opaque dtype rejection")
    } noraise {
      _ => fail("invalid or built-in opaque dtype was admitted")
    }
  }
  try StreamingSafetensorsFileDigest::from_sha256("ABC") catch {
    InvalidDigest => ()
    _ => fail("unexpected digest syntax rejection")
  } noraise {
    _ => fail("non-canonical digest was admitted")
  }
  try
    StreamingSafetensorsLimits::new(
      maximum_shards=0,
      maximum_file_bytes=9L,
      maximum_header_bytes=1,
      maximum_tensors_per_shard=1,
      maximum_total_tensors=1,
      maximum_rank=1,
      maximum_name_bytes=1,
      maximum_dtype_specs=1,
      maximum_copy_requests=1,
      maximum_copy_bytes=1L,
    )
  catch {
    InvalidLimit => ()
    _ => fail("unexpected zero-shard bound rejection")
  } noraise {
    _ => fail("zero shard bound was admitted")
  }
}
