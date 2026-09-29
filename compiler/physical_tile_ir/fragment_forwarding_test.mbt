///|
test "single-use right forwarding renames words without changing products" {
  let program = @physical_tile_ir.FragmentProgram::new(
    rows=1,
    columns=8,
    extent=128,
    quantum=16,
    slots=1,
  ).unwrap()
  let forwarding = program
    .forward_right(words=4, instruction_words=2, [0, 2, 1, 3])
    .unwrap()
  assert_eq(forwarding.program(), program)
  assert_eq(forwarding.geometry(), (4, 2, 2))
  assert_eq(forwarding.word(0, 0), Some(0))
  assert_eq(forwarding.word(0, 1), Some(2))
  assert_eq(forwarding.word(1, 0), Some(1))
  assert_eq(forwarding.word(1, 1), Some(3))
  assert_eq(forwarding.word(-1, 0), None)
  assert_eq(forwarding.word(2, 0), None)
  assert_eq(forwarding.word(0, 2), None)
  // The reused left SSA value remains shared; every right value has exactly
  // one consumer. Forwarding never reloads the reusable left operand.
  for output in 0..<program.product_count() {
    assert_eq(forwarding.program().product(output), Some((8, output, output)))
  }
  assert_true(forwarding.canonical().contains("order=0,2,1,3"))
}

///|
test "fragment forwarding rejects sharing lookahead and non-bijective packs" {
  for (rows, slots) in [(2, 1), (1, 2)] {
    let program = @physical_tile_ir.FragmentProgram::new(
      rows~,
      columns=4,
      extent=128,
      quantum=16,
      slots~,
    ).unwrap()
    assert_eq(
      program.forward_right(words=4, instruction_words=2, [0, 2, 1, 3]),
      None,
    )
  }
  let program = @physical_tile_ir.FragmentProgram::new(
    rows=1,
    columns=4,
    extent=128,
    quantum=16,
    slots=1,
  ).unwrap()
  for order in [[0, 1, 1, 3], [0, 1, 2, 4], [-1, 1, 2, 3], [0, 1, 2]] {
    assert_eq(program.forward_right(words=4, instruction_words=2, order), None)
  }
  for size in [0, 3, 8] {
    assert_eq(
      program.forward_right(words=4, instruction_words=size, [0, 1, 2, 3]),
      None,
    )
  }
}
