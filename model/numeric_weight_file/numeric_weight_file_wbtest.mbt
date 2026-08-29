///|
fn numeric_fixture_plan() -> @plan.ModelPlan {
  let operations = [
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=8, width=1),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
    ),
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(1),
      kind=OutputProjection,
      inputs=[
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
      ],
      outputs=[@plan.ActivationRef::new(1)],
      shape=ProjectionShape(input_width=1, output_width=1),
      required_capability=@plan.KernelCapabilityId::dense_projection(),
      workspace_bytes=0L,
    ),
  ]
  let numeric_schema = @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
    operations,
    [try! @plan.OperationId::new(1)],
  )
  try! @plan.ModelPlan::new_stateless(
    content_digest=@spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=8,
      vocabulary_size=8,
      hidden_size=1,
      intermediate_size=2,
      attention_head_count=1,
      kv_head_count=1,
      head_dimension=1,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=1,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(1),
    numeric_schema~,
  )
}

///|
fn append_u64_le(output : Array[Byte], value : UInt64) -> Unit {
  for shift in 0..<8 {
    output.push(((value >> (shift * 8)) & 0xffUL).to_byte())
  }
}

///|
fn append_string_bytes(output : Array[Byte], value : String) -> Unit {
  for byte in @utf8.encode(value) {
    output.push(byte)
  }
}

///|
fn numeric_shape_json(shape : @numeric_contract.NumericShape) -> String {
  let encoded : Array[String] = []
  for dimension in shape.dimensions() {
    encoded.push(dimension.to_string())
  }
  "[\{encoded.join(",")}]"
}

///|
fn fixture_dtype(
  tensor : @numeric_contract.NumericTensorSchema,
  i8_alias : Bool,
  fp8_alias : Bool,
) -> String {
  let dtype = tensor.storage().dtype()
  if dtype == @numeric_contract.StorageDType::bfloat16() {
    "BF16"
  } else if dtype == @numeric_contract.StorageDType::f8_e4m3_finite() {
    if fp8_alias {
      "F8E4M3"
    } else {
      "F8_E4M3"
    }
  } else if dtype == @numeric_contract.StorageDType::i8() {
    if i8_alias {
      "INT8"
    } else {
      "I8"
    }
  } else {
    "F32"
  }
}

///|
fn numeric_fixture_file(
  plan : @plan.ModelPlan,
  quantized_code? : Byte = b'\x01',
  scale_bits? : UInt = 0x3f800000U,
  i8_alias? : Bool = false,
  fp8_alias? : Bool = false,
  shift_second_offset? : Bool = false,
  wrong_plan_metadata? : Bool = false,
  trailing_payload? : Bool = false,
) -> Bytes {
  let tensors = plan.numeric_binding().schema().tensors()
  let header = StringBuilder()
  let plan_digest = if wrong_plan_metadata {
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
  } else {
    plan.identity().plan().as_hex()
  }
  header.write_string(
    "{\"__metadata__\":{\"format\":\"lunaflux.numeric-safetensors.v1\",\"model_content_sha256\":\"\{plan.identity().content().as_hex()}\",\"model_plan_sha256\":\"\{plan_digest}\",\"numeric_schema_sha256\":\"\{plan.numeric_binding().digest().as_hex()}\"}",
  )
  let mut offset = 0L
  for ordinal, tensor in tensors {
    let length = try! @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    )
    let start = if shift_second_offset && ordinal == 1 {
      offset + 1L
    } else {
      offset
    }
    let end = start + length
    header.write_string(
      ",\"lunaflux.tensor.v1.\{ordinal}\":{\"dtype\":\"\{fixture_dtype(tensor, i8_alias, fp8_alias)}\",\"shape\":\{numeric_shape_json(tensor.shape())},\"data_offsets\":[\{start},\{end}]}",
    )
    offset += length
  }
  header.write_string("}")
  let header_text = header.to_string()
  let output : Array[Byte] = []
  append_u64_le(output, @utf8.encode(header_text).length().to_uint64())
  append_string_bytes(output, header_text)
  for _ in 0..<16 {
    output.push(b'\x00')
  }
  output.push(quantized_code)
  output.push((scale_bits & 0xffU).to_byte())
  output.push(((scale_bits >> 8) & 0xffU).to_byte())
  output.push(((scale_bits >> 16) & 0xffU).to_byte())
  output.push(((scale_bits >> 24) & 0xffU).to_byte())
  if trailing_payload {
    output.push(b'\x00')
  }
  Bytes::from_array(output)
}

///|
fn fixture_digest(file : Bytes) -> NumericWeightArtifactDigest {
  let sha = @crypto.SHA256::new()
  sha.update_from_iter(file.iter())
  try! NumericWeightArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(sha.finalize()),
  )
}

///|
fn fixture_limits(transfer_chunk_bytes? : Int = 7) -> NumericWeightFileLimits {
  try! NumericWeightFileLimits::new(
    max_file_bytes=65536L,
    max_header_bytes=32768,
    transfer_chunk_bytes~,
    max_tensor_count=16,
  )
}

///|
fn prepare_fixture(
  plan : @plan.ModelPlan,
  file : Bytes,
  final_stamp? : Int = 1,
  transfer_chunk_bytes? : Int = 7,
) -> PreparedNumericWeightFile[Int] raise NumericWeightFileError {
  prepare_numeric_weight_file(
    plan,
    fixture_digest(file),
    1,
    file.length().to_int64(),
    fixture_limits(transfer_chunk_bytes~),
    16L,
    4096L,
    (destination, destination_offset, position, byte_count) => {
      for index in 0..<byte_count {
        destination[destination_offset + index] = file[position.to_int() + index]
      }
    },
    () => final_stamp,
  )
}

///|
test "strict numeric safetensors admits canonical mixed BF16 I8 F32 bytes" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan)
  let prepared = prepare_fixture(plan, file)
  assert_eq(prepared.layout.identity(), plan.identity())
  assert_eq(
    prepared.layout.numeric_schema_digest(),
    Some(plan.numeric_binding().digest()),
  )
  assert_eq(prepared.regions.length(), 3)
}

///|
test "strict numeric safetensors rejects dtype aliases before payload values" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan, i8_alias=true, quantized_code=b'\x80')
  try prepare_fixture(plan, file) catch {
    InvalidTensorDType(ordinal~) => assert_eq(ordinal, 1)
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected dtype rejection")
  }
}

///|
test "strict numeric safetensors rejects gaps before digest publication" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan, shift_second_offset=true)
  try prepare_fixture(plan, file) catch {
    PayloadNotContiguous(ordinal~) => assert_eq(ordinal, 1)
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected contiguous rejection")
  }
}

///|
test "strict numeric safetensors binds exact plan metadata" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan, wrong_plan_metadata=true)
  try prepare_fixture(plan, file) catch {
    ModelPlanMismatch => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected plan mismatch")
  }
}

///|
test "strict numeric payload rejects reserved I8 and non-finite scales" {
  let plan = numeric_fixture_plan()
  let reserved = numeric_fixture_file(plan, quantized_code=b'\x80')
  try prepare_fixture(plan, reserved) catch {
    ReservedI8Code(ordinal~, row~) => {
      assert_eq(ordinal, 1)
      assert_eq(row, 0)
    }
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected reserved I8 rejection")
  }
  let nan_scale = numeric_fixture_file(plan, scale_bits=0x7fc00000U)
  try prepare_fixture(plan, nan_scale) catch {
    InvalidScale(ordinal~, row~) => {
      assert_eq(ordinal, 2)
      assert_eq(row, 0)
    }
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected scale rejection")
  }
}

///|
test "strict numeric payload requires exact one scale for an all-zero row" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(
    plan,
    quantized_code=b'\x00',
    scale_bits=0x40000000U,
  )
  try prepare_fixture(plan, file) catch {
    ZeroRowScaleMismatch(ordinal~, row~) => {
      assert_eq(ordinal, 1)
      assert_eq(row, 0)
    }
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected zero-row scale rejection")
  }
}

///|
test "strict numeric admission rejects trailing bytes and stamp changes" {
  let plan = numeric_fixture_plan()
  let trailing = numeric_fixture_file(plan, trailing_payload=true)
  try prepare_fixture(plan, trailing) catch {
    TrailingPayload => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected trailing payload rejection")
  }
  let file = numeric_fixture_file(plan)
  try prepare_fixture(plan, file, final_stamp=2) catch {
    FileChanged => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected source stamp rejection")
  }
}

///|
test "strict numeric admission rejects an unpinned artifact digest last" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan)
  let wrong = try! NumericWeightArtifactDigest::from_sha256(
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  )
  try
    prepare_numeric_weight_file(
      plan,
      wrong,
      1,
      file.length().to_int64(),
      fixture_limits(),
      16L,
      4096L,
      (destination, destination_offset, position, byte_count) => {
        for index in 0..<byte_count {
          destination[destination_offset + index] = file[position.to_int() +
            index]
        }
      },
      () => 1,
    )
  catch {
    ArtifactDigestMismatch => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected artifact digest rejection")
  }
}

///|
priv struct FixtureArena {
  bytes : FixedArray[Byte]
  mut closed : Bool
}

///|
test "authenticated second pass copies only into final numeric regions" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan)
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
    fail("expected ready fixture arena")
  }
  let i8_region = prepared.regions[1]
  let scale_region = prepared.regions[2]
  assert_eq(arena.bytes[i8_region.destination_offset_bytes.to_int()], b'\x01')
  assert_eq(
    arena.bytes[scale_region.destination_offset_bytes.to_int() + 3],
    b'\x3f',
  )
}

///|
test "second-pass source revalidation deterministically rolls back allocation" {
  let plan = numeric_fixture_plan()
  let file = numeric_fixture_file(plan)
  let prepared = prepare_fixture(plan, file)
  let owner : Ref[FixtureArena?] = Ref(None)
  try
    transfer_prepared_numeric_file(
      prepared,
      (destination, destination_offset, position, byte_count) => {
        for index in 0..<byte_count {
          destination[destination_offset + index] = file[position.to_int() +
            index]
        }
      },
      () => 2,
      arena_bytes => {
        let arena = {
          bytes: FixedArray::make(arena_bytes.to_int(), b'\xff'),
          closed: false,
        }
        owner.val = Some(arena)
        arena
      },
      (arena, source, source_offset, destination_offset, byte_count) => {
        for index in 0L..<byte_count {
          arena.bytes[(destination_offset + index).to_int()] = source[(source_offset.to_int64() +
            index).to_int()]
        }
      },
      arena => arena.closed = true,
    )
  catch {
    FileChanged => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("expected second-pass source rejection")
  }
  guard owner.val is Some(arena) else { fail("expected opened arena") }
  assert_true(arena.closed)
}
