///|
priv struct FakeSegmentedAllocation {
  bytes : FixedArray[Byte]
  mut validate_calls : Int
  mut copy_calls : Int
  fail_validate_at : Int
  fail_copy_at : Int
  mut close_failures : Int
  mut closed : Bool
}

///|
fn segmented_test_identity(seed : Char) -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(String::make(64, seed))
  let plan_seed = if seed == 'a' { 'b' } else { 'c' }
  let plan = try! @spec.PlanDigest::from_sha256(String::make(64, plan_seed))
  try! @spec.ModelIdentity::new(content, plan)
}

///|
fn segmented_test_source_digest(seed : Char) -> SegmentedHostSourceDigest {
  try! SegmentedHostSourceDigest::from_sha256(String::make(64, seed))
}

///|
fn segmented_test_limits(
  maximum_device_arena_bytes? : Int64 = 1024L,
) -> SegmentedDeviceMaterializationLimits {
  try! SegmentedDeviceMaterializationLimits::new(
    alignment_bytes=64L,
    maximum_source_arenas=4,
    maximum_regions=16,
    maximum_host_arena_bytes=128,
    maximum_total_source_bytes=256L,
    maximum_device_arena_bytes~,
  )
}

///|
fn segmented_test_layout() -> SegmentedDeviceLayout {
  try! plan_layout(
    source_identity=segmented_test_identity('a'),
    source_digest=segmented_test_source_digest('d'),
    source_arena_lengths=[8, 8],
    regions=[
      try! SegmentedDeviceTensorRegionInput::new(
        tensor_ordinal=0,
        source_arena_ordinal=0,
        source_offset=1,
        byte_count=3,
      ),
      try! SegmentedDeviceTensorRegionInput::new(
        tensor_ordinal=1,
        source_arena_ordinal=1,
        source_offset=0,
        byte_count=4,
      ),
      try! SegmentedDeviceTensorRegionInput::new(
        tensor_ordinal=2,
        source_arena_ordinal=0,
        source_offset=5,
        byte_count=2,
      ),
    ],
    limits=segmented_test_limits(),
  )
}

///|
fn fake_segmented_allocation(
  byte_count : Int64,
  fail_validate_at~ : Int,
  fail_copy_at~ : Int,
  close_failures~ : Int,
) -> FakeSegmentedAllocation {
  {
    bytes: FixedArray::make(byte_count.to_int(), b'!'),
    validate_calls: 0,
    copy_calls: 0,
    fail_validate_at,
    fail_copy_at,
    close_failures,
    closed: false,
  }
}

///|
fn fake_validate_segmented(
  resource : FakeSegmentedAllocation,
  offset : Int64,
  byte_count : Int64,
  alignment : Int64,
) -> Unit raise SegmentedDeviceMaterializationError {
  let call = resource.validate_calls
  resource.validate_calls += 1
  if call == resource.fail_validate_at {
    raise DeviceFailed(Unavailable(AllocationRegionValidate))
  }
  if resource.closed ||
    offset < 0L ||
    byte_count <= 0L ||
    offset % alignment != 0L ||
    offset > resource.bytes.length().to_int64() - byte_count {
    raise DeviceFailed(InvalidArgument(AllocationRegionValidate))
  }
}

///|
fn fake_copy_segmented(
  resource : FakeSegmentedAllocation,
  source : FixedArray[Byte],
  source_offset : Int64,
  destination_offset : Int64,
  byte_count : Int64,
) -> Unit raise SegmentedDeviceMaterializationError {
  let call = resource.copy_calls
  resource.copy_calls += 1
  if call == resource.fail_copy_at {
    raise DeviceFailed(Unavailable(CopyToDevice))
  }
  if resource.closed {
    raise DeviceFailed(Closed(CopyToDevice))
  }
  for index in 0L..<byte_count {
    resource.bytes[(destination_offset + index).to_int()] = source[(source_offset +
      index).to_int()]
  }
}

///|
fn fake_close_segmented(
  resource : FakeSegmentedAllocation,
) -> Unit raise @device.DeviceError {
  if resource.close_failures > 0 {
    resource.close_failures -= 1
    raise Busy(AllocationClose)
  }
  resource.closed = true
}

///|
fn fake_prepare(
  layout : SegmentedDeviceLayout,
  arenas : ArrayView[FixedArray[Byte]],
  fail_validate_at : Int,
  fail_copy_at : Int,
  close_failures : Int,
) -> SegmentedResourcePreparation[FakeSegmentedAllocation] raise SegmentedDeviceMaterializationError {
  prepare_with_resource(
    layout,
    arenas,
    layout.source_identity(),
    layout.source_digest(),
    layout.digest(),
    bytes => {
      fake_segmented_allocation(
        bytes,
        fail_validate_at~,
        fail_copy_at~,
        close_failures~,
      )
    },
    fake_validate_segmented,
    fake_copy_segmented,
    fake_close_segmented,
  )
}

///|
test "layout is deterministic aligned and copies segmented sources in order" {
  let layout = segmented_test_layout()
  assert_eq(layout.alignment_bytes(), 64L)
  assert_eq(layout.materialized_bytes(), 9L)
  assert_eq(layout.device_arena_bytes(), 130L)
  assert_eq(layout.region_count(), 3)
  assert_eq(layout, segmented_test_layout())
  let first = FixedArray::from_array([
    b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h',
  ])
  let second = FixedArray::from_array([
    b'W', b'X', b'Y', b'Z', b'1', b'2', b'3', b'4',
  ])
  match fake_prepare(layout, [first, second], -1, -1, 0) {
    SegmentedResourceReady(ready) => {
      assert_eq(ready.resource.validate_calls, 3)
      assert_eq(ready.resource.copy_calls, 3)
      assert_eq(Bytes::from_array(ready.resource.bytes[0:3]), b"bcd")
      assert_eq(Bytes::from_array(ready.resource.bytes[64:68]), b"WXYZ")
      assert_eq(Bytes::from_array(ready.resource.bytes[128:130]), b"fg")
      assert_eq(Bytes::from_array(first[:]), b"abcdefgh")
    }
    SegmentedResourceCleanupRequired(_) => fail("successful copy needs cleanup")
  }
}

///|
test "validation and transfer failures close or retain retryable authority" {
  let layout = segmented_test_layout()
  let arenas = [FixedArray::make(8, b'a'), FixedArray::make(8, b'b')]
  let allocated : Array[FakeSegmentedAllocation] = []
  try
    prepare_with_resource(
      layout,
      arenas,
      layout.source_identity(),
      layout.source_digest(),
      layout.digest(),
      bytes => {
        let resource = fake_segmented_allocation(
          bytes,
          fail_validate_at=1,
          fail_copy_at=-1,
          close_failures=0,
        )
        allocated.push(resource)
        resource
      },
      fake_validate_segmented,
      fake_copy_segmented,
      fake_close_segmented,
    )
  catch {
    DeviceFailed(Unavailable(AllocationRegionValidate)) => ()
    _ => fail("unexpected validation failure")
  } noraise {
    _ => fail("validation failure did not propagate after cleanup")
  }
  assert_eq(allocated.length(), 1)
  assert_true(allocated[0].closed)
  match fake_prepare(layout, arenas, -1, 1, 1) {
    SegmentedResourceCleanupRequired(failed) => {
      assert_eq(failed.stage, Transfer)
      assert_eq(failed.cause, DeviceFailed(Unavailable(CopyToDevice)))
      assert_eq(failed.cleanup_failure, Busy(AllocationClose))
      assert_false(failed.resource.closed)
      fake_close_segmented(failed.resource)
      assert_true(failed.resource.closed)
    }
    SegmentedResourceReady(_) => fail("failed transfer published readiness")
  }
}

///|
test "layout replay mismatches fail before allocation" {
  let layout = segmented_test_layout()
  let arenas = [FixedArray::make(8, b'a'), FixedArray::make(8, b'b')]
  let mut allocations = 0
  try
    prepare_with_resource(
      layout,
      arenas,
      layout.source_identity(),
      segmented_test_source_digest('e'),
      layout.digest(),
      bytes => {
        allocations += 1
        fake_segmented_allocation(
          bytes,
          fail_validate_at=-1,
          fail_copy_at=-1,
          close_failures=0,
        )
      },
      fake_validate_segmented,
      fake_copy_segmented,
      fake_close_segmented,
    )
  catch {
    SourceDigestMismatch => ()
    _ => fail("unexpected digest replay rejection")
  } noraise {
    _ => fail("foreign source digest was admitted")
  }
  assert_eq(allocations, 0)
  try
    validate_layout_replay(
      layout,
      [arenas[0]],
      layout.source_identity(),
      layout.source_digest(),
      layout.digest(),
    )
  catch {
    SourceArenaShapeMismatch(-1) => ()
    _ => fail("unexpected arena-count rejection")
  } noraise {
    _ => fail("wrong source arena count was admitted")
  }
}

///|
test "hostile region ordering overlap bounds and device limit fail closed" {
  let identity = segmented_test_identity('a')
  let digest = segmented_test_source_digest('d')
  let limits = segmented_test_limits()
  let first = try! SegmentedDeviceTensorRegionInput::new(
    tensor_ordinal=0,
    source_arena_ordinal=0,
    source_offset=0,
    byte_count=4,
  )
  let wrong_order = try! SegmentedDeviceTensorRegionInput::new(
    tensor_ordinal=2,
    source_arena_ordinal=0,
    source_offset=4,
    byte_count=2,
  )
  try
    plan_layout(
      source_identity=identity,
      source_digest=digest,
      source_arena_lengths=[8],
      regions=[first, wrong_order],
      limits~,
    )
  catch {
    InvalidTensorOrdinal(1) => ()
    _ => fail("unexpected tensor-order rejection")
  } noraise {
    _ => fail("out-of-order tensor ordinal was admitted")
  }
  let overlap = try! SegmentedDeviceTensorRegionInput::new(
    tensor_ordinal=1,
    source_arena_ordinal=0,
    source_offset=3,
    byte_count=2,
  )
  try
    plan_layout(
      source_identity=identity,
      source_digest=digest,
      source_arena_lengths=[8],
      regions=[first, overlap],
      limits~,
    )
  catch {
    OverlappingSourceRegions(0, 1) => ()
    _ => fail("unexpected overlap rejection")
  } noraise {
    _ => fail("overlapping host spans were admitted")
  }
  let outside = try! SegmentedDeviceTensorRegionInput::new(
    tensor_ordinal=0,
    source_arena_ordinal=0,
    source_offset=7,
    byte_count=2,
  )
  try
    plan_layout(
      source_identity=identity,
      source_digest=digest,
      source_arena_lengths=[8],
      regions=[outside],
      limits~,
    )
  catch {
    InvalidSourceRegion(0) => ()
    _ => fail("unexpected source-bound rejection")
  } noraise {
    _ => fail("source region escaped its arena")
  }
  try
    plan_layout(
      source_identity=identity,
      source_digest=digest,
      source_arena_lengths=[8],
      regions=[
        first,
        try! SegmentedDeviceTensorRegionInput::new(
          tensor_ordinal=1,
          source_arena_ordinal=0,
          source_offset=4,
          byte_count=4,
        ),
      ],
      limits=segmented_test_limits(maximum_device_arena_bytes=67L),
    )
  catch {
    DeviceArenaTooLarge => ()
    _ => fail("unexpected device bound rejection")
  } noraise {
    _ => fail("device arena exceeded its bound")
  }
}

///|
test "contiguous tensor segments retain their logical tensor ordinal" {
  let layout = try! plan_layout(
    source_identity=segmented_test_identity('a'),
    source_digest=segmented_test_source_digest('d'),
    source_arena_lengths=[8, 8],
    regions=[
      try! SegmentedDeviceTensorRegionInput::new(
        tensor_ordinal=0,
        source_arena_ordinal=0,
        source_offset=0,
        byte_count=4,
      ),
      try! SegmentedDeviceTensorRegionInput::new(
        tensor_ordinal=0,
        source_arena_ordinal=1,
        source_offset=0,
        byte_count=4,
      ),
      try! SegmentedDeviceTensorRegionInput::new(
        tensor_ordinal=1,
        source_arena_ordinal=0,
        source_offset=4,
        byte_count=4,
      ),
    ],
    limits=segmented_test_limits(),
  )
  assert_eq(layout.region_count(), 3)
  guard layout.region(0) is Some(first) else { fail("missing first segment") }
  guard layout.region(1) is Some(second) else { fail("missing second segment") }
  guard layout.region(2) is Some(third) else { fail("missing third segment") }
  assert_eq(first.tensor_ordinal(), 0)
  assert_eq(second.tensor_ordinal(), 0)
  assert_eq(third.tensor_ordinal(), 1)
}

///|
test "digest and limit constructors reject hostile values" {
  for digest in ["", String::make(64, 'A'), String::make(63, 'a')] {
    try SegmentedHostSourceDigest::from_sha256(digest) catch {
      InvalidDigest => ()
      _ => fail("unexpected digest syntax rejection")
    } noraise {
      _ => fail("non-canonical digest was admitted")
    }
  }
  try
    SegmentedDeviceMaterializationLimits::new(
      alignment_bytes=24L,
      maximum_source_arenas=1,
      maximum_regions=1,
      maximum_host_arena_bytes=1,
      maximum_total_source_bytes=1L,
      maximum_device_arena_bytes=1L,
    )
  catch {
    InvalidLimit => ()
    _ => fail("unexpected alignment-limit rejection")
  } noraise {
    _ => fail("non-power-of-two alignment was admitted")
  }
}
