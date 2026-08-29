///|
fn matrix_encode_reference(values : FixedArray[Int]) -> Bytes {
  let output = FixedArray::make(values.length() * 2, b'\x00')
  for index, value in values {
    output[index * 2] = value.to_byte()
    output[index * 2 + 1] = (value >> 8).to_byte()
  }
  Bytes::from_array(output[:])
}

///|
test "shape matrix stays inside the exact admitted envelope" {
  let cases = shape_matrix_cases()
  assert_eq(cases.length(), 5)
  for case in cases {
    let prefill = case.counts[0]
    let decode = case.counts[1]
    let rows = case.counts[2]
    let tokens = case.counts[3]
    let pages = case.counts[4]
    assert_true(prefill >= 0 && decode >= 0)
    assert_eq(prefill + decode, rows)
    assert_true(rows >= 1 && rows <= 2)
    assert_true(tokens >= 1 && tokens <= 3)
    assert_true(pages >= 1 && pages <= 3)
    assert_eq(case.row_offsets[0], 0)
    assert_eq(case.row_offsets[rows], tokens)
    assert_eq(case.page_offsets[0], 0)
    assert_eq(case.page_offsets[rows], pages)
    for row in 0..<rows {
      assert_true(case.row_offsets[row] < case.row_offsets[row + 1])
      assert_true(case.page_offsets[row] < case.page_offsets[row + 1])
      assert_eq(
        case.page_offsets[row + 1] - case.page_offsets[row],
        (case.sequence_lengths[row] + 1) / 2,
      )
      for token in case.row_offsets[row]..<case.row_offsets[row + 1] {
        assert_true(case.positions[token] < case.sequence_lengths[row])
      }
    }
  }
}

///|
test "independent CPU referee covers every matrix case and boundary" {
  for case in shape_matrix_cases() {
    let reference = matrix_reference(case)
    let outputs = FixedArray::makei(reference.length(), index => {
      matrix_encode_reference(reference[index])
    })
    let (token, maximum_error) = try! matrix_validate_reference(case, outputs)
    assert_true(token >= 0 && token < 16)
    assert_eq(maximum_error, 0.0)
    let corrupted = reference[4].copy()
    corrupted[0] = matrix_bf16_bits(1000.0F)
    outputs[4] = matrix_encode_reference(corrupted)
    try matrix_validate_reference(case, outputs) catch {
      ShapeMatrixFailure(message) => {
        assert_true(message.contains(case.name))
        assert_true(message.contains("boundary 4"))
      }
    } noraise {
      _ => fail("corrupted attention boundary passed for \{case.name}")
    }
  }
}

///|
async test "checked graph contract has the runtime-pinned digest" {
  let bytes = @fs.read_file(
    "tests/paged_bf16_shape_matrix_probe/fixtures/graph_contract.v1",
  ).binary()
  assert_eq(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    SHAPE_MATRIX_CONTRACT_SHA256,
  )
}
