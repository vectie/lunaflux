///|
test "explicit row mapping preserves interleaving BF16 bits and padding" {
  let conditioning = [0x7fc1U, 0x8000U, 0x3f80U, 0xff80U]
  let video = [11U, 12U, 21U, 22U]
  let audio = [31U, 32U]
  let plan = @packed_rows_plan.Plan::new(
    width=2,
    total_rows=7,
    conditioning=[0, 4],
    video=[5, 3],
    audio=[1],
    max_elements=14L,
  )
  let result : Array[UInt] = []
  for row in plan.rows() {
    for lane in 0..<2 {
      let value = match row.source {
        Padding => 0U
        Conditioning => conditioning[row.index * 2 + lane]
        Video => video[row.index * 2 + lane]
        Audio => audio[row.index * 2 + lane]
      }
      result.push(value)
    }
  }
  assert_eq(result, [
    0x7fc1U, 0x8000U, 31U, 32U, 0U, 0U, 21U, 22U, 0x3f80U, 0xff80U, 11U, 12U, 0U,
    0U,
  ])
  assert_eq(plan.element_count(), 14L)
  assert_eq(plan.input_rows(Video), 2)
  assert_eq(plan.input_rows(Audio), 1)
  assert_eq(plan.input_rows(Conditioning), 2)
  let tags = [1, 3, 0, 2, 1, 2, 0]
  let indices = [0, 0, 0, 1, 1, 0, 0]
  let bytes = plan.mapping()
  assert_eq(bytes.length(), 56)
  for row in 0..<7 {
    assert_eq(bytes[row * 8].to_int(), tags[row])
    assert_eq(bytes[row * 8 + 4].to_int(), indices[row])
    for byte in [1, 2, 3, 5, 6, 7] {
      assert_eq(bytes[row * 8 + byte], b'\x00')
    }
  }
}

///|
test "row map snapshots input and rejects overlapping or out of range rows" {
  let positions = [1, 2]
  let plan = @packed_rows_plan.Plan::new(
    width=4,
    total_rows=4,
    conditioning=[],
    video=positions,
    audio=[0],
    max_elements=16L,
  )
  positions[0] = 3
  assert_eq(plan.rows()[1].index, 0)
  assert_true(plan.rows()[3].source is Padding)
  for rows in [[0], [-1], [4], [1, 1]] {
    try
      @packed_rows_plan.Plan::new(
        width=4,
        total_rows=4,
        conditioning=[],
        video=rows,
        audio=[0],
        max_elements=16L,
      )
    catch {
      @packed_rows_plan.PlanError::InvalidRows => ()
      _ => fail("unexpected row error")
    } noraise {
      _ => fail("invalid row map accepted")
    }
  }
  try
    @packed_rows_plan.Plan::new(
      width=4,
      total_rows=4,
      conditioning=[],
      video=[],
      audio=[],
      max_elements=15L,
    )
  catch {
    @packed_rows_plan.PlanError::ElementLimitExceeded => ()
    _ => fail("unexpected budget error")
  } noraise {
    _ => fail("short budget accepted")
  }
}
