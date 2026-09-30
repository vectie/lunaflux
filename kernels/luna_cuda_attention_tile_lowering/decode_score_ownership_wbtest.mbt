///|
test "dual score component phases account for row stride and subword banks" {
  for dimensions = 4; dimensions <= 256; dimensions = dimensions + 4 {
    let phase = cuda_dual_score_group_phase(dimensions)
    assert_true(cuda_dual_score_phase_legal(dimensions, phase))
  }
  assert_eq(cuda_dual_score_group_phase(128), 1)
  assert_eq(cuda_dual_score_group_phase(48), 0)
  assert_eq(cuda_dual_score_group_phase(60), 2)
}
