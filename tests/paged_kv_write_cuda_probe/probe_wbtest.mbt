///|
test "KV-write physical probe pins exact candidate and six scheduler-valid cases" {
  let candidate = kv_probe_candidate("9".repeat(64))
  assert_eq(candidate.target().compute_major(), 12)
  assert_eq(candidate.target().compute_minor(), 0)
  assert_eq(candidate.compiler_policy().identity().patch(), 115)
  assert_eq(candidate.launch_dimensions().grid_x(), 8)
  assert_eq(candidate.launch_dimensions().block_x(), 256)
  assert_eq(candidate.dispatch_canary_per_token(), 22321U)
  let cases = kv_probe_cases()
  assert_eq(cases.length(), 6)
  let mut tokens = 0
  let mut saw_prefill = false
  let mut saw_decode = false
  let mut saw_mixed = false
  for case_ in cases {
    assert_eq(case_.step_counts[0] + case_.step_counts[1], case_.step_counts[2])
    assert_eq(case_.positions.length(), case_.step_counts[3])
    assert_eq(case_.row_offsets[case_.step_counts[2]], case_.positions.length())
    assert_eq(case_.page_offsets[case_.step_counts[2]], case_.pages.length())
    for row in 0..<case_.step_counts[2] {
      assert_eq(
        case_.page_offsets[row + 1] - case_.page_offsets[row],
        (case_.sequence_lengths[row] + 15) / 16,
      )
    }
    for row in 0..<case_.step_counts[0] {
      let begin = case_.row_offsets[row]
      let end = case_.row_offsets[row + 1]
      let mut token = begin + 1
      while token < end {
        assert_eq(case_.positions[token], case_.positions[token - 1] + 1)
        token += 1
      }
      assert_eq(case_.sequence_lengths[row], case_.positions[end - 1] + 1)
    }
    for row in case_.step_counts[0]..<case_.step_counts[2] {
      assert_eq(case_.row_offsets[row + 1] - case_.row_offsets[row], 1)
      assert_eq(
        case_.sequence_lengths[row],
        case_.positions[case_.row_offsets[row]] + 1,
      )
    }
    saw_prefill = saw_prefill || case_.step_counts[0] > 0
    saw_decode = saw_decode || case_.step_counts[1] > 0
    saw_mixed = saw_mixed ||
      (case_.step_counts[0] > 0 && case_.step_counts[1] > 0)
    tokens += case_.positions.length()
  }
  assert_eq(tokens, 17)
  assert_true(saw_prefill && saw_decode && saw_mixed)
}

///|
test "serial oracle is distinct deterministic source with exact positioned-QKV layout" {
  let first = kv_serial_oracle_source()
  let second = kv_serial_oracle_source()
  assert_eq(first, second)
  let source = @utf8.decode(first)
  assert_true(source.contains("blockDim.x != 1"))
  assert_true(source.contains("qkv+128ULL+component"))
  assert_true(source.contains("qkv+192ULL+component"))
  assert_true(
    source.contains("(long long)rows!=(long long)prefill+(long long)decode"),
  )
  assert_true(source.contains("logical_page>=end-begin"))
  assert_false(source.contains("attention"))
}

///|
test "scalar cache oracle mutates exact K and V cells only" {
  let case_ = kv_probe_cases()[2]
  let qkv = kv_case_qkv_bits(case_)
  let expected = kv_scalar_expected(case_, qkv[:])
  let mut key = 0
  let mut value = 0
  for cell in expected.0 {
    if cell != 0x7fc1 {
      key += 1
    }
  }
  for cell in expected.1 {
    if cell != 0x7fc1 {
      value += 1
    }
  }
  assert_eq(key, 128)
  assert_eq(value, 128)
}
