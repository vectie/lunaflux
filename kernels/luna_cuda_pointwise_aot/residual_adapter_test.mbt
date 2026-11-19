///|
test "standalone residual reuses BF16 add with exact in-place indexing" {
  guard @physical_tile_ir.VectorOwnership::new(vectors=2048, owners=256)
    is Some(ownership) else {
    fail("ownership")
  }
  let source = @utf8.decode(
    @luna_cuda_pointwise_aot.bf16_residual_source(
      ownership,
      symbol="decoder_residual",
    )[:],
  )
  assert_true(source.contains("live_tokens * 2048ULL"))
  assert_true(source.contains("blockIdx.x * 256ULL"))
  assert_true(
    source.contains(
      "const float sum = __bfloat162float(left[i]) + __bfloat162float(right[i])",
    ),
  )
  assert_true(source.contains("output[i] = __float2bfloat16_rn(sum)"))
}
