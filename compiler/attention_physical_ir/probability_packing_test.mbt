///|
test "streaming probability packing preserves ordered row reductions" {
  let fold = @attention_physical_ir.OnlineFold::new(
    queries=64,
    keys=64,
    dimensions=128,
  ).unwrap()
  let program = fold
    .fragment_products(
      fragment_rows=16,
      fragment_columns=16,
      reduction_quantum=16,
    )
    .unwrap()
  let owners = [0, 0, 1, 1, 0, 0, 1, 1]
  let packing = program.pack_probabilities(owners, elements_per_word=2).unwrap()
  assert_eq(packing.program(), program)
  assert_eq(packing.geometry(), (4, 2))
  assert_eq(packing.peak_scalar_inputs(), 2)
  let observed : Array[Int] = []
  for word in 0..<4 {
    for element in 0..<2 {
      observed.push(packing.owner(word, element).unwrap())
    }
  }
  assert_eq(observed, owners)
  assert_eq(packing.owner(-1, 0), None)
  assert_eq(packing.owner(4, 0), None)
  assert_eq(packing.owner(0, 2), None)
  assert_eq(program.pack_probabilities([], elements_per_word=2), None)
  assert_eq(program.pack_probabilities(owners, elements_per_word=0), None)
  assert_eq(program.pack_probabilities(owners, elements_per_word=3), None)
  assert_eq(program.pack_probabilities([0, 16], elements_per_word=2), None)
}
