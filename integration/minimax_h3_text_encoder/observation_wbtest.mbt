///|
test "operator capture failure stops before subsequent submission" {
  for failure in ["run", "snapshot", "consume"] {
    let submitted : Array[Int] = []
    let captured : Array[String] = []
    try
      observe_operator_outputs(
        14,
        step => {
          submitted.push(step)
          if failure == "run" && step == 5 {
            fail("run")
          }
        },
        region => {
          if failure == "snapshot" && submitted.last() == Some(5) {
            fail("snapshot")
          }
          Bytes::make(1, region.to_byte())
        },
        (name, _) => {
          captured.push(name)
          if failure == "consume" && name == "q-norm" {
            fail("consume")
          }
        },
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("observation failure swallowed")
    }
    assert_eq(submitted, [0, 1, 2, 3, 4, 5])
    assert_true(!captured.contains("k-norm") && !captured.contains("attention"))
  }
}

///|
test "matched layers consume original predecessors, never the previous diagnostic output" {
  let inputs : Array[Int] = []
  let original : (Int) -> Bytes raise = i => {
    inputs.push(i)
    Bytes::make(2, i.to_byte())
  }
  assert_eq(matched_layer_input(0, 2L, Some(original)), None)
  for next in 1..<=50 {
    assert_eq(
      matched_layer_input(next, 2L, Some(original)),
      Some(Bytes::make(2, (next - 1).to_byte())),
    )
  }
  assert_eq(inputs, Array::makei(50, i => i))
  assert_eq(matched_layer_input(7, 2L, None), None)
  try matched_layer_input(7, 3L, Some(original)) catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue) => ()
    error => raise error
  } noraise {
    _ => fail("a different input shape was substituted")
  }
}

///|
test "first-layer operator captures follow the binding recipe and valid regions" {
  let boundaries = first_layer_operator_boundaries()
  let outputs = first_layer_operator_outputs()
  assert_eq(boundaries.length(), 14)
  assert_eq(outputs.length(), 16)
  let names : Array[String] = []
  for (end, region, name) in outputs {
    assert_true(end > 0 && end <= boundaries.length())
    assert_true(region >= 0 && region < 8)
    names.push(name)
  }
  assert_eq(names, [
    "embedding", "norm-input", "q", "k", "v", "q-norm", "k-norm", "q-rotary", "k-rotary",
    "attention", "output", "residual-attn", "norm-mlp", "activation", "down", "residual-mlp",
  ])
  assert_eq(boundaries[13], (13, 14))
  assert_true(PrefixComplete != Complete)
}

///|
test "observation partitions exactly the text and deepstack launch recipes" {
  let text = hidden_state_boundaries(false)
  let visual = hidden_state_boundaries(true)
  assert_eq(text.length(), 51)
  assert_eq(visual.length(), 51)
  assert_eq(text[0], (0, 1))
  assert_eq(visual[0], (0, 2))
  for layer in 1..<=50 {
    assert_eq(text[layer], (layer, 1 + layer * layer_function_order().length()))
    assert_eq(visual[layer], (layer, text[layer].1 + 1 + Int::min(layer, 3)))
    assert_true(text[layer].1 > text[layer - 1].1)
    assert_true(visual[layer].1 > visual[layer - 1].1)
  }
  assert_eq(text[50].1, 651)
  assert_eq(visual[50].1, 655)
}

///|
test "hidden observations are ordered and stop immediately on execution failure" {
  let calls : Array[Int] = []
  let outputs : Array[Int] = []
  try
    observe_hidden_states(
      51,
      index => {
        calls.push(index)
        if index == 3 {
          fail("execution failure")
        }
        b"\x01"
      },
      (index, bytes) => {
        assert_eq(bytes, b"\x01")
        outputs.push(index)
      },
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("observation swallowed execution failure")
  }
  assert_eq(calls, [0, 1, 2, 3])
  assert_eq(outputs, [0, 1, 2])
}

///|
test "observer failure cannot submit subsequent layers" {
  let calls : Array[Int] = []
  let outputs : Array[Int] = []
  try
    observe_hidden_states(
      51,
      index => {
        calls.push(index)
        b"\x01"
      },
      (index, _) => {
        outputs.push(index)
        if index == 2 {
          fail("observer failure")
        }
      },
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("observation swallowed callback failure")
  }
  assert_eq(calls, [0, 1, 2])
  assert_eq(outputs, [0, 1, 2])
}
