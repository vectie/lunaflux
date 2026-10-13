///|
test "expert tensor ranks preserve identity and halve intermediate packed weights" {
  let accuracy = @precision_ir.Accuracy::new(absolute=0.02, relative=0.02)
  let whole = @glm53_packed_expert_upload.expert_compute_plan(
    layer_index=3,
    expert_indices=Array::makei(288, i => i),
    maximum_rows=16,
    accuracy~,
    matrix_law=MatrixFloat32,
    memory_ceiling=8589934592L,
  )
  for rank in 0..<2 {
    let partition = @glm53_packed_expert_upload.tensor_expert_compute_plan(
      layer_index=3,
      maximum_rows=16,
      rank~,
      world_size=2,
      accuracy~,
      matrix_law=MatrixFloat32,
      memory_ceiling=4294967296L,
    )
    assert_eq(partition.rank_plan().geometry(), (16, 4096, 1024, 288, 8))
    assert_eq(partition.rank_plan().local_experts(), whole.local_experts())
    assert_eq(partition.collective_width(), 32768)
    assert_eq(partition.intermediate_start(), rank * 1024)
    assert_true(
      partition.rank_plan().bank_bytes() < whole.bank_bytes() * 51L / 100L,
    )
    assert_eq(
      partition.rank_plan().down_publication(),
      ReduceFloat32BeforeRound,
    )
  }
}
