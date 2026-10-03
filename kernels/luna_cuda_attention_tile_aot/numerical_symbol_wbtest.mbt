///|
test "mixed frontier namespace follows actual elementary arithmetic" {
  assert_eq(
    cuda_attention_numerical_symbol("prefill_exp2_v1", AttentionMatrixQkPv),
    "prefill_v1",
  )
  assert_eq(
    cuda_attention_numerical_symbol(
      "prefill_exp2_v1",
      AttentionMatrixQkPvApproxExp,
    ),
    "prefill_exp2_v1",
  )
  assert_eq(
    cuda_attention_numerical_symbol("prefill_v1", AttentionMatrixQkPv),
    "prefill_v1",
  )
  assert_eq(
    cuda_attention_numerical_symbol("custom", AttentionMatrixQkPvApproxExp),
    "custom_exp2_v1",
  )
}
