///|
test "packed shards slice payload and block scales without requantization" {
  let rep = @format.Representation::new(format=NvFp4, rows=8, columns=64)
  let row = @precision_ir.PackedMatrixSlice::new(
    rep,
    row_start=4,
    rows=4,
    column_start=0,
    columns=64,
  )
  assert_eq(row.payload().geometry(), (128L, 32L, 4, 32L))
  assert_eq(row.scales().unwrap().geometry(), (16L, 4L, 4, 4L))
  assert_eq(row.representation().shape(), (4, 64))
  assert_true(row.representation().has_global_scale())
  let column = @precision_ir.PackedMatrixSlice::new(
    rep,
    row_start=0,
    rows=8,
    column_start=32,
    columns=32,
  )
  assert_eq(column.payload().geometry(), (16L, 32L, 8, 16L))
  assert_eq(column.scales().unwrap().geometry(), (2L, 4L, 8, 2L))
  for (start, width) in [(1, 16), (8, 16), (16, 17), (32, 48), (-16, 16)] {
    try
      @precision_ir.PackedMatrixSlice::new(
        rep,
        row_start=0,
        rows=8,
        column_start=start,
        columns=width,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("unaligned/out-of-range shard")
    }
  }
}

///|
test "tensor parallel expert plan partitions inner axis not expert identities" {
  let original = @precision_ir.ExpertMlpPrecision::new(
    gate=expert_operand(64, 64),
    up=expert_operand(64, 64),
    down=expert_operand(64, 64),
    expert_count=2,
    local_experts=[0, 1],
    selected_experts=2,
    maximum_rows=3,
    dot_law=MatrixFloat32,
    memory_ceiling=1000000L,
  )
  for rank in 0..<2 {
    let p = @precision_ir.ExpertTensorPartition::new(
      original,
      rank~,
      world_size=2,
      memory_ceiling=1000000L,
    )
    assert_eq(p.rank(), rank)
    assert_eq(p.world_size(), 2)
    assert_eq(p.intermediate_start(), rank * 32)
    assert_eq(p.rank_plan().geometry(), (3, 64, 32, 2, 2))
    assert_eq(p.rank_plan().local_experts(), original.local_experts())
    assert_eq(p.rank_plan().down_publication(), ReduceFloat32BeforeRound)
    assert_eq(p.rank_plan().workspace_bytes(), (384L, 1536L))
    assert_eq(p.collective_width(), 128)
    assert_eq(p.slices()[0].representation().shape(), (32, 64))
    assert_eq(p.slices()[2].representation().shape(), (64, 32))
    assert_true(p.rank_plan().bank_bytes() < original.bank_bytes())
  }
  for (rank, world) in [(0, 1), (-1, 2), (2, 2), (0, 3)] {
    try
      @precision_ir.ExpertTensorPartition::new(
        original,
        rank~,
        world_size=world,
        memory_ceiling=1000000L,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid partition")
    }
  }
}

///|
test "two-dimensional scale groups require both shard boundaries aligned" {
  let rep = @format.Representation::new(
    format=MxFp8E4M3,
    rows=256,
    columns=256,
    grouping=Block(rows=128, columns=128),
  )
  let p = @precision_ir.PackedMatrixSlice::new(
    rep,
    row_start=128,
    rows=128,
    column_start=128,
    columns=128,
  )
  assert_eq(p.scales().unwrap().geometry(), (3L, 2L, 1, 1L))
  try
    @precision_ir.PackedMatrixSlice::new(
      rep,
      row_start=64,
      rows=128,
      column_start=0,
      columns=256,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("split scale block")
  }
}
