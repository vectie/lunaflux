///|
fn fp8_numeric_fixture_plan() -> @plan.ModelPlan {
  let base = numeric_fixture_plan()
  let numeric_schema = @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
    base.operations(),
    [try! @plan.OperationId::new(1)],
  )
  try! @plan.ModelPlan::new_stateless(
    content_digest=base.identity().content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=base.shape_constraints(),
    kv_cache_geometry=base.kv_cache_geometry(),
    workspace_bounds=base.workspace_bounds(),
    operations=base.operations(),
    final_output=base.final_output(),
    numeric_schema~,
  )
}

///|
test "authenticated numeric loader admits strict finite E4M3 payload" {
  let plan = fp8_numeric_fixture_plan()
  for code in [b'\x00', b'\x7e', b'\x80', b'\xfe'] {
    let file = numeric_fixture_file(plan, quantized_code=code)
    let prepared = prepare_fixture(plan, file)
    assert_eq(prepared.layout.identity(), plan.identity())
    assert_eq(
      prepared.layout.numeric_schema_digest(),
      Some(plan.numeric_binding().digest()),
    )
    assert_eq(prepared.regions.length(), 3)
  }
}

///|
test "authenticated second pass copies finite E4M3 into the final arena" {
  let plan = fp8_numeric_fixture_plan()
  let file = numeric_fixture_file(plan, quantized_code=b'\x7e')
  let prepared = prepare_fixture(plan, file)
  let loaded = transfer_prepared_numeric_file(
    prepared,
    (destination, destination_offset, position, byte_count) => {
      for index in 0..<byte_count {
        destination[destination_offset + index] = file[position.to_int() + index]
      }
    },
    () => 1,
    arena_bytes => {
      bytes: FixedArray::make(arena_bytes.to_int(), b'\xff'),
      closed: false,
    },
    (arena, source, source_offset, destination_offset, byte_count) => {
      for index in 0L..<byte_count {
        arena.bytes[(destination_offset + index).to_int()] = source[(source_offset.to_int64() +
          index).to_int()]
      }
    },
    _ => (),
  )
  guard loaded is GenericNumericReady(arena) else {
    fail("expected ready FP8 fixture arena")
  }
  let weight_region = prepared.regions[1]
  let scale_region = prepared.regions[2]
  assert_eq(
    arena.bytes[weight_region.destination_offset_bytes.to_int()],
    b'\x7e',
  )
  assert_eq(
    arena.bytes[scale_region.destination_offset_bytes.to_int() + 3],
    b'\x3f',
  )
}

///|
test "authenticated numeric loader rejects FP8 dtype aliases before payload" {
  let plan = fp8_numeric_fixture_plan()
  let file = numeric_fixture_file(plan, quantized_code=b'\x7f', fp8_alias=true)
  try prepare_fixture(plan, file) catch {
    InvalidTensorDType(ordinal=1) => ()
    _ => fail("unexpected FP8 dtype-alias failure")
  } noraise {
    _ => fail("FP8 dtype alias was admitted")
  }
}

///|
test "authenticated numeric loader rejects both finite E4M3 NaN codes" {
  let plan = fp8_numeric_fixture_plan()
  for code in [b'\x7f', b'\xff'] {
    let file = numeric_fixture_file(plan, quantized_code=code)
    try prepare_fixture(plan, file) catch {
      NonFiniteFp8Code(ordinal=1, element=0) => ()
      _ => fail("unexpected non-finite FP8 payload failure")
    } noraise {
      _ => fail("non-finite FP8 code was admitted")
    }
  }
}

///|
test "finite E4M3 scanner is chunk bounded and reports exact element" {
  let bytes : Bytes = [0, 1, 2, 3, 4, 5, 6, 0xff, 8, 9]
  let maximum = Ref(0)
  try
    validate_finite_fp8_region(
      {
        ordinal: 4,
        source_offset_bytes: 0L,
        destination_offset_bytes: 0L,
        byte_length: 10L,
      },
      FixedArray::make(7, b'\x00'),
      7,
      payload_read(bytes, maximum),
    )
  catch {
    NonFiniteFp8Code(ordinal=4, element=7) => ()
    _ => fail("unexpected chunked FP8 scanner failure")
  } noraise {
    _ => fail("non-finite FP8 code across a chunk boundary was admitted")
  }
  assert_true(maximum.val <= 7)
}

///|
test "finite E4M3 payload retains strict positive finite scalar scale" {
  let plan = fp8_numeric_fixture_plan()
  for bits in [0U, 0x80000000U, 0xbf800000U, 0x7f800000U, 0x7fc00000U] {
    let file = numeric_fixture_file(plan, scale_bits=bits)
    try prepare_fixture(plan, file) catch {
      InvalidScale(ordinal=2, row=0) => ()
      _ => fail("unexpected FP8 scale failure")
    } noraise {
      _ => fail("invalid FP8 scalar scale was admitted")
    }
  }
}
