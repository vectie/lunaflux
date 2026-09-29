///|
test "products share operand values without merging ordered accumulators" {
  let p = FragmentProgram::new(
    rows=2,
    columns=3,
    extent=64,
    quantum=16,
    slots=2,
  ).unwrap()
  assert_eq(p.load_count(), 5)
  assert_eq(p.product_count(), 6)
  assert_eq(p.load(0), Some((RightOperand, 0)))
  assert_eq(p.load(4), Some((LeftOperand, 1)))
  assert_eq(p.product(0), Some((3, 0, 0)))
  assert_eq(p.product(3), Some((4, 0, 3)))
  assert_eq(p.product(5), Some((4, 2, 5)))
  assert_eq(p.product(6), None)
  assert_eq(p.load(-1), None)
}

///|
test "lookahead never overwrites the live register slot or reads past the tail" {
  for extent in [16, 32, 64, 128] {
    for slots in [1, 2] {
      if slots > extent / 16 {
        continue
      }
      let p = FragmentProgram::new(
        rows=1,
        columns=2,
        extent~,
        quantum=16,
        slots~,
      ).unwrap()
      let resident = Array::make(slots, -1)
      if p.primes_operands() {
        resident[0] = 0
      }
      for iteration in 0..<(extent / 16) {
        for action in p.iteration() {
          match action {
            ReadCurrent => {
              let (offset, slot) = p.read_at(iteration, next=false).unwrap()
              resident[slot] = offset
            }
            ReadNextIfPresent =>
              if p.read_at(iteration, next=true) is Some((offset, slot)) {
                assert_true(slot != iteration % slots)
                resident[slot] = offset
              }
            AccumulateCurrent =>
              assert_eq(resident[iteration % slots], iteration * 16)
          }
        }
      }
      assert_eq(p.read_at(extent / 16 - 1, next=true), None)
      assert_eq(p.read_at(-1, next=false), None)
      assert_eq(p.read_at(extent / 16, next=false), None)
    }
  }
}

///|
test "physical descriptors reject invalid and overflowing geometry" {
  assert_eq(
    FragmentProgram::new(rows=0, columns=1, extent=16, quantum=16, slots=1),
    None,
  )
  assert_eq(
    FragmentProgram::new(rows=1, columns=1, extent=17, quantum=16, slots=1),
    None,
  )
  assert_eq(
    FragmentProgram::new(rows=1, columns=1, extent=16, quantum=16, slots=2),
    None,
  )
  assert_eq(
    FragmentProgram::new(
      rows=2147483647,
      columns=2,
      extent=16,
      quantum=16,
      slots=1,
    ),
    None,
  )
  let a = FragmentProgram::new(
    rows=1,
    columns=2,
    extent=64,
    quantum=16,
    slots=1,
  ).unwrap()
  let b = FragmentProgram::new(
    rows=1,
    columns=2,
    extent=64,
    quantum=16,
    slots=2,
  ).unwrap()
  assert_true(a.canonical() != b.canonical())
}
