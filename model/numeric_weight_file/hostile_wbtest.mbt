///|
fn hostile_layout(
  plan : @plan.ModelPlan,
  alignment? : Int64 = 16L,
) -> @device_materialize.DeviceWeightLayout {
  try! @device_materialize.plan_numeric_layout(
    plan,
    alignment_bytes=alignment,
    max_arena_bytes=4096L,
  )
}

///|
fn expect_authority_failure(
  actual_identity : @spec.ModelIdentity,
  actual_schema : @numeric_contract.ModelNumericSchemaDigest,
  actual_layout : @device_materialize.DeviceWeightLayout,
  actual_artifact : NumericWeightArtifactDigest,
  expected_identity : @spec.ModelIdentity,
  expected_schema : @numeric_contract.ModelNumericSchemaDigest,
  expected_layout : @device_materialize.DeviceWeightLayout,
  expected_artifact : NumericWeightArtifactDigest,
  expected : NumericWeightFileError,
) -> Unit raise {
  try
    validate_numeric_weight_authority(
      true, actual_identity, actual_schema, actual_layout, actual_artifact, expected_identity,
      expected_schema, expected_layout, expected_artifact,
    )
  catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("mismatched authority joined")
  }
}

///|
test "numeric authority checks liveness before four exact joins" {
  let plan = numeric_fixture_plan()
  let layout = hostile_layout(plan)
  let schema = plan.numeric_binding().digest()
  let artifact = try! NumericWeightArtifactDigest::from_sha256(
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  )
  validate_numeric_weight_authority(
    true,
    plan.identity(),
    schema,
    layout,
    artifact,
    plan.identity(),
    schema,
    layout,
    artifact,
  )
  try
    validate_numeric_weight_authority(
      false,
      plan.identity(),
      schema,
      layout,
      artifact,
      plan.identity(),
      schema,
      layout,
      artifact,
    )
  catch {
    Closed => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("closed owner minted authority")
  }
  let other_identity = try! @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
    plan.identity().plan(),
  )
  let other_schema = try! @numeric_contract.ModelNumericSchemaDigest::from_sha256(
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  )
  let other_layout = hostile_layout(plan, alignment=32L)
  let other_artifact = try! NumericWeightArtifactDigest::from_sha256(
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  )
  expect_authority_failure(
    plan.identity(),
    schema,
    layout,
    artifact,
    other_identity,
    schema,
    layout,
    artifact,
    AuthorityMismatch(Identity),
  )
  expect_authority_failure(
    plan.identity(),
    schema,
    layout,
    artifact,
    plan.identity(),
    other_schema,
    layout,
    artifact,
    AuthorityMismatch(NumericSchema),
  )
  expect_authority_failure(
    plan.identity(),
    schema,
    layout,
    artifact,
    plan.identity(),
    schema,
    other_layout,
    artifact,
    AuthorityMismatch(Layout),
  )
  expect_authority_failure(
    plan.identity(),
    schema,
    layout,
    artifact,
    plan.identity(),
    schema,
    layout,
    other_artifact,
    AuthorityMismatch(Artifact),
  )
}

///|
test "shared liveness guard accepts before close and rejects after close" {
  let open = Ref(true)
  require_numeric_weights_open(open.val)
  open.val = false
  try require_numeric_weights_open(open.val) catch {
    Closed => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("closed authority remained live")
  }
}

///|
test "loader scalar preflight admits 65536 and rejects overflow" {
  let limits = try! NumericWeightFileLimits::new(
    max_file_bytes=1024L,
    max_header_bytes=1024,
    transfer_chunk_bytes=16,
    max_tensor_count=65536,
  )
  preflight_numeric_weight_scalars(65536, limits)
  try preflight_numeric_weight_scalars(65537, limits) catch {
    InvalidLimit(TensorCount) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("tensor-count overflow admitted")
  }
}

///|
test "numeric chunk limit rejects 1 2 3 and exact F32 width admits payload" {
  for transfer_chunk_bytes in [1, 2, 3] {
    try
      NumericWeightFileLimits::new(
        max_file_bytes=1024L,
        max_header_bytes=1024,
        transfer_chunk_bytes~,
        max_tensor_count=16,
      )
    catch {
      InvalidLimit(TransferChunkBytes) => ()
      _ => fail("unexpected sub-F32 chunk error")
    } noraise {
      _ => fail("sub-F32 transfer chunk admitted")
    }
  }
  let limits = try! NumericWeightFileLimits::new(
    max_file_bytes=65536L,
    max_header_bytes=32768,
    transfer_chunk_bytes=4,
    max_tensor_count=16,
  )
  assert_eq(limits.transfer_chunk_bytes(), 4)
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan)
  let prepared = prepare_fixture(plan, file, transfer_chunk_bytes=4)
  assert_eq(prepared.regions.length(), 3)
}

///|
fn payload_read(
  bytes : Bytes,
  maximum : Ref[Int],
) -> (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError {
  (destination, destination_offset, position, byte_count) => {
    maximum.val = maximum.val.max(byte_count)
    for index in 0..<byte_count {
      destination[destination_offset + index] = bytes[position.to_int() + index]
    }
  }
}

///|
test "payload scanner caps reads and rejects -128 across chunk and row boundaries" {
  let chunk_boundary : Bytes = [0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 128, 63]
  let maximum = Ref(0)
  try
    validate_i8_scale_rows(
      {
        ordinal: 1,
        source_offset_bytes: 0L,
        destination_offset_bytes: 0L,
        byte_length: 8L,
      },
      {
        ordinal: 2,
        source_offset_bytes: 8L,
        destination_offset_bytes: 16L,
        byte_length: 4L,
      },
      1L,
      8L,
      FixedArray::make(64, b'\x00'),
      7,
      payload_read(chunk_boundary, maximum),
    )
  catch {
    ReservedI8Code(ordinal=1, row=0) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("reserved code at chunk boundary admitted")
  }
  assert_true(maximum.val <= 7)

  let row_boundary : Bytes = [
    0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 128, 63, 0, 0, 128, 63,
  ]
  try
    validate_i8_scale_rows(
      {
        ordinal: 3,
        source_offset_bytes: 0L,
        destination_offset_bytes: 0L,
        byte_length: 8L,
      },
      {
        ordinal: 4,
        source_offset_bytes: 8L,
        destination_offset_bytes: 16L,
        byte_length: 8L,
      },
      2L,
      4L,
      FixedArray::make(64, b'\x00'),
      7,
      payload_read(row_boundary, Ref(0)),
    )
  catch {
    ReservedI8Code(ordinal=3, row=1) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("reserved code at row boundary admitted")
  }
}

///|
fn f32_bytes(bits : UInt) -> Bytes {
  [
    (bits & 0xffU).to_byte(),
    ((bits >> 8) & 0xffU).to_byte(),
    ((bits >> 16) & 0xffU).to_byte(),
    ((bits >> 24) & 0xffU).to_byte(),
  ]
}

///|
test "scale scanner rejects signed zero negative and infinities" {
  for bits in [0U, 0x80000000U, 0xbf800000U, 0x7f800000U, 0xff800000U] {
    let bytes = f32_bytes(bits)
    try
      validate_scale_region(
        {
          ordinal: 2,
          source_offset_bytes: 0L,
          destination_offset_bytes: 0L,
          byte_length: 4L,
        },
        FixedArray::make(8, b'\x00'),
        4,
        payload_read(bytes, Ref(0)),
      )
    catch {
      InvalidScale(ordinal=2, row=0) => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid F32 scale admitted")
    }
  }
}

///|
test "zero-row exact-one rule proves little-endian scale interpretation" {
  let big_endian_one : Bytes = [0, 0, 0, 0, 63, 128, 0, 0]
  try
    validate_i8_scale_rows(
      {
        ordinal: 1,
        source_offset_bytes: 0L,
        destination_offset_bytes: 0L,
        byte_length: 4L,
      },
      {
        ordinal: 2,
        source_offset_bytes: 4L,
        destination_offset_bytes: 16L,
        byte_length: 4L,
      },
      1L,
      4L,
      FixedArray::make(8, b'\x00'),
      4,
      payload_read(big_endian_one, Ref(0)),
    )
  catch {
    ZeroRowScaleMismatch(ordinal=1, row=0) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("big-endian scale admitted as exact one")
  }
}

///|
priv struct HostileArena {
  mut close_attempts : Int
}

///|
test "copy failure retains cleanup failure and permits deterministic retry" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan)
  let prepared = prepare_fixture(plan, file)
  let loaded = transfer_prepared_numeric_file(
    prepared,
    payload_read(file, Ref(0)),
    () => 1,
    _ => { close_attempts: 0 },
    (_, _, _, _, _) => raise InvalidArgument(CopyToDevice),
    arena => {
      arena.close_attempts += 1
      if arena.close_attempts == 1 {
        raise DriverFailure(AllocationClose)
      }
    },
  )
  guard loaded
    is GenericNumericCleanupRequired(
      arena,
      DeviceFailed(InvalidArgument(CopyToDevice)),
      Transfer,
      DriverFailure(AllocationClose)
    ) else {
    fail("copy/cleanup compound failure lost authority")
  }
  let retried = close_failed_numeric_resource(arena, resource => {
    resource.close_attempts += 1
  })
  assert_true(retried is Ok(_))
  assert_eq(arena.close_attempts, 2)
}

///|
test "source-close compound semantics retain only allocation retry authority" {
  let arena = { close_attempts: 0 }
  let result = finish_ready_numeric_source_close(
    arena,
    () => raise FileCloseFailed,
    resource => {
      resource.close_attempts += 1
      raise DriverFailure(AllocationClose)
    },
  )
  guard result
    is NumericSourceCleanupRequired(retained, DriverFailure(AllocationClose)) else {
    fail("source-close compound failure lost allocation")
  }
  assert_eq(retained.close_attempts, 1)
  let closed = { close_attempts: 0 }
  try
    finish_ready_numeric_source_close(closed, () => raise FileCloseFailed, resource => {
      resource.close_attempts += 1
    })
  catch {
    FileCloseFailed => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("source close failure published readiness")
  }
  assert_eq(closed.close_attempts, 1)
}
