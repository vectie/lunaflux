///|
fn gemv_tensor(
  index : Int32,
  dtype : @tile.LunaTileDType,
  shape : ArrayView[Int64],
  strides : ArrayView[Int64],
  offset~ : Int64 = 0L,
  alignment~ : Int64 = 4L,
  space~ : @tile.LunaTileAddressSpace = Global,
) -> @tile.LunaTileTensorView raise {
  @tile.LunaTileTensorView::new(
    id=id(index),
    dtype~,
    address_space=space,
    shape~,
    strides_bytes=strides,
    offset_bytes=offset,
    alignment_bytes=alignment,
  )
}

///|
fn gemv_operation(
  rows : Int32,
  columns : Int32,
  offset~ : Int64 = 256L,
) -> @compiler.OrderedBf16Gemv raise {
  @compiler.OrderedBf16Gemv::new(
    weights=gemv_tensor(
      1,
      BFloat16,
      [rows.to_int64(), columns.to_int64()],
      [columns.to_int64() * 2L, 2L],
      offset~,
      alignment=128L,
    ),
    activation=gemv_tensor(2, Float32, [columns.to_int64()], [4L]),
    output=gemv_tensor(3, Float32, [rows.to_int64()], [4L]),
  )
}

///|
fn gemv_bindings(
  weights~ : UInt64 = 0x50000000UL,
  input~ : UInt64 = 0x7ffffe00UL,
  output~ : UInt64 = 0x40000004UL,
) -> Array[@compiler.GlobalBinding] raise {
  [
    { tensor: id(1), address: weights, },
    { tensor: id(2), address: input, },
    { tensor: id(3), address: output, },
  ]
}

///|
fn descriptor_words(program : @device.PreparedProgram) -> Array[UInt32] raise {
  assert_eq(program.chunk_count(), 1)
  let bytes = program.chunk(0).unwrap()
  Array::makei(bytes.length() / 4, i => {
    bytes[i * 4].to_uint() |
    (bytes[i * 4 + 1].to_uint() << 8) |
    (bytes[i * 4 + 2].to_uint() << 16) |
    (bytes[i * 4 + 3].to_uint() << 24)
  })
}

///|
fn canonical_float(value : Float) -> UInt32 {
  let bits = value.reinterpret_as_uint()
  if (bits & 0x7fffffffU) > 0x7f800000U {
    0x7fc00000U
  } else {
    bits
  }
}

///|
// Independent host arithmetic: widen operands exactly, round each product to
// binary32, then separately round each ordered sum. Never use an FMA oracle.
fn gemv_oracle(
  weights : ArrayView[UInt32],
  input : ArrayView[Float],
  rows : Int32,
) -> Array[UInt32] {
  Array::makei(rows, row => {
    let mut sum = 0.0F
    for col, x in input {
      let w = Float::reinterpret_from_uint(
        weights[row * input.length() + col] << 16,
      )
      let product = Float::from_double(w.to_double() * x.to_double())
      sum = Float::from_double(sum.to_double() + product.to_double())
    }
    canonical_float(sum)
  })
}

///|
fn fixture_weight(row : Int32, column : Int32) -> UInt32 {
  match row {
    0 => if column < 3 { 0x3f80U } else { 0U }
    1 => if column == 0 { 0xb380U } else if column == 3 { 0x3f81U } else { 0U }
    2 => if column == 3 { 0x3f80U } else { 0U }
    3 => if column == 1 { 0x7fc1U } else { 0U }
    4 => if column == 1 { 0x7f80U } else { 0U }
    5 => if column == 1 { 0xff80U } else { 0U }
    6 => if column == 0 { 0x7f80U } else if column == 1 { 0xff80U } else { 0U }
    7 => if column == 1 { 0x0001U } else { 0U }
    8 => if column == 0 { 0x7f7fU } else { 0U }
    9 => 0x8000U
    _ =>
      ((row * 19 + column * 13) % 255).reinterpret_as_uint() |
      (if (row + column) % 3 == 0 { 0xbf00U } else { 0x3f00U })
  }
}

///|
/// Mixed-precision conformance payload: parallel 16x128 and scalar 3x64 GEMV,
/// upper DDR boundary, nonzero weight offset, unaligned-in-beat output, stalls,
/// ordered cancellation, non-fused rounding, BF16 subnormals and special values.
pub fn export_bf16_fixture(contract : String) -> String raise {
  let device = @device.parse_contract(contract)
  let input = Array::makei(128, i => {
    match i {
      0 => 16777216.0F
      1 => 1.0F
      2 => -16777216.0F
      3 => Float::reinterpret_from_uint(0x3f800001U)
      _ => Float::from_double((i % 11 - 5).to_double() / 16.0)
    }
  })
  let weights = Array::makei(16 * 128, i => fixture_weight(i / 128, i % 128))
  let scalar = Array::makei(3 * 64, i => fixture_weight(10 + i / 64, i % 64))
  let parallel_words = descriptor_words(
    gemv_operation(16, 128).lower(device, gemv_bindings()),
  )
  let scalar_words = descriptor_words(
    gemv_operation(3, 64, offset=0L).lower(
      device,
      gemv_bindings(weights=0x60000000UL, output=0x40000104UL),
    ),
  )
  let words = parallel_words[:parallel_words.length() - 16].to_owned()
  for word in scalar_words {
    words.push(word)
  }
  let expected = gemv_oracle(weights, input, 16)
  assert_eq(expected[0], 0U) // Ordered fold: (2^24 + 1) - 2^24 = 0.
  assert_eq(expected[1], 0x3c000080U) // Separate product rounding, not FMA.
  assert_eq(expected[2], 0x3f800001U) // FP32 activation must not become BF16.
  assert_eq(expected[7], 0x00010000U) // Smallest BF16 subnormal survives.
  let semantic : Json = {
    "operation": "ordered-bf16-row-major-fp32-gemv/v1",
    "shapes": [[16, 128], [3, 64]],
    "program_words": @json.to_json(words),
  }
  let identity = @crypto.bytes_to_hex_string(
    @crypto.sha256(@utf8.encode(semantic.stringify())),
  )
  let fixture : Json = {
    "schema": "lunachip.adapter-fixture/v1",
    "source_program": identity,
    "program_words": @json.to_json(words),
    "assets": [
      {
        "address": "2147483136",
        "words": @json.to_json(input.map(x => x.reinterpret_as_uint())),
      },
      {
        "address": "1342177536",
        "words": @json.to_json(
          Array::makei(weights.length() / 2, i => {
            weights[i * 2] | (weights[i * 2 + 1] << 16)
          }),
        ),
      },
      {
        "address": "1610612736",
        "words": @json.to_json(
          Array::makei(scalar.length() / 2, i => {
            scalar[i * 2] | (scalar[i * 2 + 1] << 16)
          }),
        ),
      },
    ],
    "outputs": [
      { "address": "1073741828", "words": @json.to_json(expected) },
      {
        "address": "1073742084",
        "words": @json.to_json(gemv_oracle(scalar, input[:64], 3)),
      },
    ],
  }
  fixture.stringify(indent=2) + "\n"
}
