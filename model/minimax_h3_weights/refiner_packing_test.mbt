///|
test "refiner packing excludes AdaLN and stores final norm only after last layer" {
  let model = @minimax_h3.build(h3_weight_metadata())
  let first = refiner_weight_packing(model, layer=0, max_bytes=2000000000L)
  let last = refiner_weight_packing(model, layer=1, max_bytes=2000000000L)
  assert_eq(first.copies.length(), 11)
  assert_true(first.final_norm_offset is None)
  assert_eq(last.copies.length(), 12)
  assert_true(last.final_norm_offset is Some(_))
  assert_eq(
    first.copies[0].tensor_name,
    "token_refiner.refiner_blocks.0.attn.to_q.weight",
  )
  assert_eq(first.copies[6].source_offset, 14336L * 5376L * 2L)
  assert_eq(last.copies[11].tensor_name, "token_refiner.final_norm.weight")
  ignore(refiner_weight_packing(model, layer=1, max_bytes=last.byte_count))
}
