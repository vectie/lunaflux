///|
test "index quantization is an explicit numerical and storage choice" {
  let ordered = @precision_ir.IndexScoringLaw::OrderedBFloat16
  let matrix = @precision_ir.IndexScoringLaw::RowScaledE4M3Matrix
  assert_true(ordered != matrix)
  assert_eq(ordered.pooled_row_bytes(128), 256L)
  assert_eq(matrix.pooled_row_bytes(128), 132L)
}
