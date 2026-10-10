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
