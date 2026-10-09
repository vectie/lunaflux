///|
test "SSA liveness reuses dead storage without producer consumer overlap" {
  let values : Array[@media_encode_plan.Value] = [
    { elements: 8L, origin: Input, },
    { elements: 8L, origin: Temporary, },
    { elements: 8L, origin: Temporary, },
    { elements: 8L, origin: Temporary, },
  ]
  let nodes : Array[@media_encode_plan.Node] = [
    { operation: Add(elements=8L), inputs: [0, 0], output: 1, },
    { operation: Add(elements=8L), inputs: [1, 0], output: 2, },
    { operation: Add(elements=8L), inputs: [2, 0], output: 3, },
  ]
  let graph = @media_encode_plan.Graph::new(
    values,
    nodes,
    output=3,
    max_scratch_bytes=512L,
  )
  assert_eq(graph.scratch_bytes, 512L)
  assert_eq(graph.offsets[1], 0L)
  assert_eq(graph.offsets[2], 256L)
  assert_eq(graph.offsets[3], 0L)
  try
    @media_encode_plan.Graph::new(
      values,
      nodes,
      output=3,
      max_scratch_bytes=511L,
    )
  catch {
    BudgetExceeded => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("accepted undersized scratch")
  }
}

///|
test "graph rejects self read and mismatched operands before lowering" {
  let values : Array[@media_encode_plan.Value] = [
    { elements: 8L, origin: Input, },
    { elements: 8L, origin: Temporary, },
  ]
  for inputs in [[1, 0], [0], [0, 2]] {
    try
      @media_encode_plan.Graph::new(
        values,
        [
          {
            operation: Add(elements=8L),
            inputs: ReadOnlyArray::from_array(inputs),
            output: 1,
          },
        ],
        output=1,
        max_scratch_bytes=256L,
      )
    catch {
      InvalidGraph => ()
      _ => fail("wrong failure")
    } noraise {
      _ => fail("accepted malformed graph")
    }
  }
}

///|
test "dilated strided convolution validates derived extent" {
  let valid : @media_encode_plan.Operation = Conv(
    batch=2,
    input_channels=4,
    output_channels=8,
    input_time=20,
    output_time=4,
    kernel=10,
    dilation=1,
    stride=5,
    padding=3,
  )
  let (inputs, output) = valid.signature()
  assert_eq(inputs[0], 160L)
  assert_eq(inputs[1], 320L)
  assert_eq(output, 64L)
  let invalid : @media_encode_plan.Operation = Conv(
    batch=2,
    input_channels=4,
    output_channels=8,
    input_time=20,
    output_time=5,
    kernel=10,
    dilation=1,
    stride=5,
    padding=3,
  )
  try invalid.signature() catch {
    InvalidGeometry => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("accepted wrong output extent")
  }
}
