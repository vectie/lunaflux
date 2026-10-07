///|
test "read ownership is bijective and preserves consumer values" {
  let program = @physical_tile_ir.FragmentProgram::new(
    rows=1,
    columns=4,
    extent=128,
    quantum=16,
    slots=1,
  ).unwrap()
  let forward = program
    .forward_right(words=4, instruction_words=2, [0, 2, 1, 3])
    .unwrap()
  for width in [1, 4, 8, 16] {
    let plan = forward.contiguous_read(provider_group_width=width).unwrap()
    let owners = []
    for owner in 0..<(width * 4) {
      let source = plan.provider(owner).unwrap()
      assert_false(owners.contains(source))
      owners.push(source)
      assert_eq(source % width, owner % width)
      assert_eq(source / width, plan.source_word(owner / width).unwrap())
    }
    assert_eq(plan.provider(-1), None)
    assert_eq(plan.provider(width * 4), None)
    assert_eq(plan.source_word(-1), None)
    assert_eq(plan.source_word(4), None)
  }
  assert_eq(forward.contiguous_read(provider_group_width=0), None)
  assert_eq(forward.contiguous_read(provider_group_width=2147483647), None)
}

///|
test "ordered read-ahead releases every slot before rebinding" {
  for items in 1..<=16 {
    for slots in 1..<=items {
      let plan = @physical_tile_ir.OrderedOperandReads::new(items~, slots~).unwrap()
      let live = Array::make(slots, -1)
      let reads = []
      let products = []
      for action in plan.actions() {
        match action {
          ReadOperand(item~, slot~) => {
            assert_eq(live[slot], -1)
            live[slot] = item
            reads.push(item)
          }
          ConsumeOperand(item~, slot~) => {
            assert_eq(live[slot], item)
            live[slot] = -1
            products.push(item)
          }
        }
      }
      assert_eq(reads, Array::makei(items, i => i))
      assert_eq(products, reads)
      assert_true(live.all(v => v == -1))
      assert_eq(plan.geometry(), (items, slots))
    }
  }
  for (items, slots) in [(0, 1), (1, 0), (2, 3), (1025, 1)] {
    assert_eq(@physical_tile_ir.OrderedOperandReads::new(items~, slots~), None)
  }
}
