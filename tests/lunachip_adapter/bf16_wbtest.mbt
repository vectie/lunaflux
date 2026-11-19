///|
test "BF16 GEMV descriptor bounds and nonoverlap are checked before encoding" {
  let device = @device.parse_contract(contract_text())
  let op = @device.Instruction::bf16_gemv(device, 4096, 0, 0x7fffff80UL, 1, 64)
  assert_eq(op.words()[0], 4U)
  assert_eq(op.words()[5], 64U)
  assert_eq(op.words()[8], 0x7fffff80U)
  assert_eq(op.words()[10], 0U)
  for
    (dst, input, address, rows, cols) in [
      (4096, 0, 0UL, 0, 64),
      (4096, 0, 0UL, 4097, 64),
      (4096, 0, 0UL, 1, 0),
      (4096, 0, 0UL, 1, 63),
      (4096, 0, 0UL, 1, 4097),
      (4096, 0, 4UL, 1, 64),
      (4096, 0, 0x7fffff80UL, 2, 64),
      (4096, 0, 0xffffffffffffff80UL, 1, 64),
      (0, 0, 0UL, 1, 64),
      (63, 0, 0UL, 2, 64),
      (32768, 0, 0UL, 1, 64),
      (4096, 32767, 0UL, 1, 64),
      (-1, 0, 0UL, 1, 64),
    ] {
    let mut rejected = false
    ignore(
      @device.Instruction::bf16_gemv(device, dst, input, address, rows, cols),
    ) catch {
      _ => rejected = true
    }
    assert_true(rejected)
  }
}

///|
test "explicit mixed BF16 operation emits load GEMV store and resolves view offset" {
  let device = @device.parse_contract(contract_text())
  let words = descriptor_words(
    gemv_operation(16, 128).lower(device, gemv_bindings()),
  )
  assert_eq(words.length(), 64)
  for i, op in [14U, 4U, 15U, 0U] {
    assert_eq(words[i * 16], op)
  }
  assert_eq(words[16 + 8], 0x50000100U)
  assert_eq(words[16 + 1], 4096U)
  assert_eq(words[16 + 4], 16U)
  assert_eq(words[16 + 5], 128U)
  // Maximum admitted vector/matrix extents are encoded without Int32 overflow.
  ignore(
    gemv_operation(4096, 4096, offset=0L).lower(
      device,
      gemv_bindings(input=0x70000000UL),
    ),
  )
  assert_eq(
    export_bf16_fixture(contract_text()),
    export_bf16_fixture(contract_text()),
  )
}

///|
test "raw LunaTile views require dtype rank stride alignment and unique ownership validation" {
  // TensorView::new itself is unchecked; this backend must validate it fully.
  let input = gemv_tensor(2, Float32, [128L], [4L])
  let output = gemv_tensor(3, Float32, [16L], [4L])
  for
    weight in [
      gemv_tensor(1, Float32, [16L, 128L], [256L, 2L], alignment=128L),
      gemv_tensor(1, BFloat16, [16L], [2L], alignment=128L),
      gemv_tensor(1, BFloat16, [16L, 128L], [2L, 32L], alignment=128L),
      gemv_tensor(1, BFloat16, [16L, 128L], [256L, 2L], alignment=64L),
      gemv_tensor(1, BFloat16, [16L, 128L], [256L, 2L], alignment=192L),
      gemv_tensor(
        1,
        BFloat16,
        [16L, 128L],
        [256L, 2L],
        alignment=128L,
        offset=-128L,
      ),
      gemv_tensor(
        1,
        BFloat16,
        [16L, 128L],
        [256L, 2L],
        alignment=128L,
        offset=2L,
      ),
      gemv_tensor(
        1,
        BFloat16,
        [16L, 128L],
        [256L, 2L],
        alignment=128L,
        space=Register,
      ),
      gemv_tensor(2, BFloat16, [16L, 128L], [256L, 2L], alignment=128L),
    ] {
    try
      @compiler.OrderedBf16Gemv::new(weights=weight, activation=input, output~)
    catch {
      @compiler.UnsupportedTensor => ()
      _ => fail("wrong tensor error")
    } noraise {
      _ => fail("invalid weight view accepted")
    }
  }
  for
    activation in [
      gemv_tensor(2, BFloat16, [128L], [2L]),
      gemv_tensor(2, Float32, [64L], [4L]),
      gemv_tensor(2, Float32, [128L], [8L]),
    ] {
    let mut rejected = false
    ignore(
      @compiler.OrderedBf16Gemv::new(
        weights=gemv_tensor(
          1,
          BFloat16,
          [16L, 128L],
          [256L, 2L],
          alignment=128L,
        ),
        activation~,
        output~,
      ),
    ) catch {
      _ => rejected = true
    }
    assert_true(rejected)
  }
}

///|
test "GEMV rejects missing duplicate overlapping misaligned and wrapping DDR bindings" {
  let device = @device.parse_contract(contract_text())
  let op = gemv_operation(16, 128)
  let original = gemv_bindings()
  for
    bindings in [
      [original[0], original[1]],
      [original[0], original[0], original[2]],
      gemv_bindings(weights=0x50000004UL),
      gemv_bindings(output=0x50000100UL),
      gemv_bindings(input=0x50000200UL),
      gemv_bindings(output=0x7ffffe00UL),
      gemv_bindings(input=0x7ffffe04UL),
      gemv_bindings(weights=0x7ffff000UL),
      gemv_bindings(weights=0xffffffffffffffffUL),
      gemv_bindings(output=0x40000002UL),
    ] {
    let mut rejected = false
    ignore(op.lower(device, bindings)) catch {
      _ => rejected = true
    }
    assert_true(rejected)
  }
}

///|
test "numeric controls distinguish ordered FP32 nonfused and BF16 activation rounding" {
  let x = [
    16777216.0F,
    1.0F,
    -16777216.0F,
    Float::reinterpret_from_uint(0x3f800001U),
  ]
  let weights = [0xb380U, 0U, 0U, 0x3f81U]
  let separate = gemv_oracle(weights, x, 1)[0]
  let fused = Float::from_double(-1.0 + 1.0078125 * x[3].to_double()).reinterpret_as_uint()
  assert_eq(separate, 0x3c000080U)
  assert_eq(fused, 0x3c000081U)
  assert_true(separate != fused)
  assert_eq(gemv_oracle([0x3f80U, 0x3f80U, 0x3f80U, 0U], x, 1)[0], 0U)
  assert_eq(gemv_oracle([0U, 0U, 0U, 0x3f80U], x, 1)[0], 0x3f800001U)
  assert_true(0x3f800001U != 0x3f800000U)
}

///|
test "Qwen row tiles reuse all matrix layouts including the tied vocabulary tail" {
  let device = @device.parse_contract(contract_text())
  let layout = @layout.prepare(device, metadata())
  let x = layout.workspace().address
  let y = x + 16384UL
  let mut matrices = 0
  for ordinal in 0..<layout.weight_count() {
    let dims = layout.dimensions(ordinal).unwrap()
    if dims.length() != 2 {
      continue
    }
    matrices += 1
    let count = dims[0].min(4096L).to_int()
    let first = dims[0].to_int() - count
    let words = descriptor_words(
      @transfer.prepare_projection(device, layout, ordinal, first, count, x, y),
    )
    let expected = layout.weight(ordinal).unwrap().address +
      first.to_uint64() * dims[1].reinterpret_as_uint64() * 2UL
    assert_eq(words[24], expected.to_uint())
    assert_eq(words[20], count.reinterpret_as_uint())
    assert_eq(words[21], dims[1].to_int().reinterpret_as_uint())
  }
  assert_eq(matrices, 197)
  let tail = descriptor_words(
    @transfer.prepare_projection(device, layout, 0, 151935, 1, x, y),
  )
  assert_eq(tail[24], (151935UL * 1024UL * 2UL).to_uint())
  for
    (ordinal, first, rows, input, output) in [
      (-1, 0, 1, x, y),
      (310, 0, 1, x, y),
      (1, 0, 1, x, y),
      (0, -1, 1, x, y),
      (0, 151935, 2, x, y),
      (0, 0, 4097, x, y),
      (0, 0, 1, 0UL, y),
      (0, 0, 1, x, layout.reserve().address),
      (0, 0, 1, x, x),
      (0, 0, 1, x, 0xffffffffffffffffUL),
    ] {
    let mut rejected = false
    ignore(
      @transfer.prepare_projection(
        device, layout, ordinal, first, rows, input, output,
      ),
    ) catch {
      _ => rejected = true
    }
    assert_true(rejected)
  }
}
