///|
test "online fragments preserve query ownership and numerical fold" {
  for queries in [16, 32, 64, 128] {
    for keys in [16, 32, 64, 128] {
      for dimensions in [32, 64, 128] {
        let fold = @attention_physical_ir.OnlineFold::new(
          queries~,
          keys~,
          dimensions~,
        ).unwrap()
        let program = fold
          .fragment_products(
            fragment_rows=16,
            fragment_columns=16,
            reduction_quantum=16,
          )
          .unwrap()
        assert_eq(program.fold(), fold)
        assert_eq(program.geometry(), (16, 16, 16))
        assert_eq(program.score().geometry(), (1, keys / 16, dimensions, 16, 1))
        assert_eq(program.value().geometry(), (1, dimensions / 16, keys, 16, 1))
        assert_eq(program.fold().score_phase(), fold.score_phase())
        assert_eq(program.fold().value_phase(), fold.value_phase())
        assert_eq(program.fold().carried_state(), fold.carried_state())
      }
    }
  }
}

///|
test "online fragment refinement rejects incomplete instruction domains" {
  for (queries, keys, dimensions) in [(15, 32, 32), (16, 31, 32), (16, 32, 31)] {
    let fold = @attention_physical_ir.OnlineFold::new(
      queries~,
      keys~,
      dimensions~,
    ).unwrap()
    assert_eq(
      fold.fragment_products(
        fragment_rows=16,
        fragment_columns=16,
        reduction_quantum=16,
      ),
      None,
    )
  }
  let fold = @attention_physical_ir.OnlineFold::new(
    queries=16,
    keys=32,
    dimensions=32,
  ).unwrap()
  assert_eq(
    fold.fragment_products(
      fragment_rows=0,
      fragment_columns=16,
      reduction_quantum=16,
    ),
    None,
  )
  assert_eq(
    fold.fragment_products(
      fragment_rows=16,
      fragment_columns=0,
      reduction_quantum=16,
    ),
    None,
  )
  assert_eq(
    fold.fragment_products(
      fragment_rows=16,
      fragment_columns=16,
      reduction_quantum=0,
    ),
    None,
  )
  assert_eq(
    fold.fragment_products(
      fragment_rows=16,
      fragment_columns=16,
      reduction_quantum=64,
    ),
    None,
  )
}
