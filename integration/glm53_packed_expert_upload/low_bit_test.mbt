///|
test "native low-bit expert arithmetic propagates through both TP partitions" {
  let accuracy = @precision_ir.Accuracy::new(absolute=0.1, relative=0.1)
  for rank in 0..<2 {
    let partition = @glm53_packed_expert_upload.tensor_expert_compute_plan(
      layer_index=3,
      maximum_rows=16,
      rank~,
      world_size=2,
      accuracy~,
      matrix_law=MatrixFloat32,
      expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
      memory_ceiling=100000000000L,
    )
    let p = partition.rank_plan()
    assert_eq(p.scaled_dot_law(), BlockScaledMatrixFloat32(block_columns=16))
    assert_eq(p.down_publication(), ReduceFloat32BeforeRound)
    let (rows, hidden, intermediate, _, selected) = p.geometry()
    guard p.activation_quantization() is Some((input, product)) else {
      fail("missing quantization effects")
    }
    assert_eq(input.operand().representation().shape(), (rows, hidden))
    assert_eq(
      product.operand().representation().shape(),
      (rows * selected, intermediate),
    )
    assert_eq(product.passes().length(), 3)
  }
}
