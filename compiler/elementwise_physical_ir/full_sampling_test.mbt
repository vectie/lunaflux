///|
test "full sampling network is a total-order permutation with a bounded envelope" {
  for size in 1..<=33 {
    let program = @elementwise_physical_ir.FullVocabularySamplingProgram::new(
      values=size,
      maximum_rows=3,
      maximum_workspace_bytes=8192L,
    ).unwrap()
    let (_, padded, rows) = program.geometry()
    assert_eq(rows, 3)
    let ids = FixedArray::makei(padded, i => {
      if i < size {
        i
      } else {
        2147483647
      }
    })
    let values = FixedArray::makei(padded, i => {
      if i < size {
        i * 13 % 7
      } else {
        -1
      }
    })
    for (width, distance) in program.stages() {
      for i in 0..<padded {
        let j = i ^ distance
        if j > i {
          let i_first = values[i] > values[j] ||
            (values[i] == values[j] && ids[i] < ids[j])
          if i_first != ((i & width) == 0) {
            let v = values[i]
            let id = ids[i]
            values[i] = values[j]
            ids[i] = ids[j]
            values[j] = v
            ids[j] = id
          }
        }
      }
    }
    let seen = FixedArray::make(size, false)
    for i in 0..<size {
      assert_true(ids[i] >= 0 && ids[i] < size)
      assert_false(seen[ids[i]])
      seen[ids[i]] = true
      if i > 0 {
        assert_true(
          values[i - 1] > values[i] ||
          (values[i - 1] == values[i] && ids[i - 1] < ids[i]),
        )
      }
    }
    assert_true(program.workspace_bytes() <= 8192L)
  }
}

///|
test "full sampling rejects overflow and insufficient workspace before lowering" {
  assert_true(
    @elementwise_physical_ir.FullVocabularySamplingProgram::new(
      values=0,
      maximum_rows=1,
      maximum_workspace_bytes=100L,
    )
    is None,
  )
  assert_true(
    @elementwise_physical_ir.FullVocabularySamplingProgram::new(
      values=67108865,
      maximum_rows=1,
      maximum_workspace_bytes=9223372036854775807L,
    )
    is None,
  )
  assert_true(
    @elementwise_physical_ir.FullVocabularySamplingProgram::new(
      values=8,
      maximum_rows=2,
      maximum_workspace_bytes=255L,
    )
    is None,
  )
  assert_true(
    @elementwise_physical_ir.FullVocabularySamplingProgram::new(
      values=8,
      maximum_rows=2,
      maximum_workspace_bytes=256L,
    )
    is Some(_),
  )
}
