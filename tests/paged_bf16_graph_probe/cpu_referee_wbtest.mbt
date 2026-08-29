///|
fn encode_reference(values : FixedArray[Int]) -> Bytes {
  let output = FixedArray::make(values.length() * 2, b'\x00')
  for index, value in values {
    output[index * 2] = value.to_byte()
    output[index * 2 + 1] = (value >> 8).to_byte()
  }
  Bytes::from_array(output[:])
}

///|
test "CPU referee covers every boundary and samples deterministically" {
  let reference = build_graph_reference()
  let outputs = FixedArray::makei(reference.length(), index => {
    encode_reference(reference[index])
  })
  let (token, maximum_error) = try! validate_graph_reference(outputs)
  assert_true(token >= 0 && token < 16)
  assert_eq(maximum_error, 0.0)
  let corrupted = FixedArray::make(reference[4].length() * 2, b'\x00')
  for index, value in reference[4] {
    corrupted[index * 2] = value.to_byte()
    corrupted[index * 2 + 1] = (value >> 8).to_byte()
  }
  corrupted[1] = (corrupted[1].to_int() ^ 1).to_byte()
  outputs[4] = Bytes::from_array(corrupted[:])
  try validate_graph_reference(outputs) catch {
    GraphProbeFailure(message) => assert_true(message.contains("boundary 4"))
  } noraise {
    _ => fail("corrupted attention boundary passed the CPU referee")
  }
}

///|
test "CPU fixture construction rejects every wrong fixed size" {
  for index in 0..<fixture_kinds.length() {
    let wrong = FixedArray::make(buffer_sizes[index] + 2, b'\x00')
    if fixture_kinds[index] == 21 {
      assert_true(fill_graph_fixture(21, wrong))
    } else {
      assert_false(fill_graph_fixture(fixture_kinds[index], wrong))
    }
  }
}
