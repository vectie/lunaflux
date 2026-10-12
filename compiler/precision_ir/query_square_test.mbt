///|
test "learned query square order is explicit without changing frame or weight geometry" {
  let make = law => {
    @precision_ir.QuantizedQueryKvPrecision::new(
      hidden_width=256,
      query_rank=128,
      head_count=2,
      head_width=128,
      maximum_rows=3,
      normalization_epsilon=1.0e-6,
      activation_amax_floor=0.0001,
      maximum_workspace_bytes=6144L,
      query_square_law=law,
    )
  }
  let old = make(VectorizedTreeFloat32(partitions=256, vector_width=1))
  let selected = make(VectorizedTreeFloat32(partitions=128, vector_width=4))
  assert_eq(
    selected.query_square_law(),
    VectorizedTreeFloat32(partitions=128, vector_width=4),
  )
  assert_eq(selected.frame_bytes(), old.frame_bytes())
  assert_eq(selected.weight_bytes(), old.weight_bytes())
  assert_eq(selected.geometry(), old.geometry())
  assert_eq(selected.head_normalization(), old.head_normalization())
  try make(VectorizedTreeFloat32(partitions=3, vector_width=4)) catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("invalid query fold accepted")
  }
}
