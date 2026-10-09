///|
test "F32 RF update preserves staged rounding without scratch or division" {
  let source = @utf8.decode(
    @rectified_flow_update_cuda_source.render(elements=1025L, block_size=256),
  )
  assert_true(source.contains("i >= 1025ULL"))
  assert_true(source.contains("* 256ULL"))
  assert_true(source.contains("__fadd_rn(x, scaled)"))
  assert_true(source.contains("__fmul_rn(coefficients[2], denoised)"))
  assert_true(!source.contains("bfloat16"))
  assert_true(!source.contains("__fmaf"))
  assert_true(!source.contains("__syncthreads"))
}

///|
test "invalid element or block geometry rejects" {
  for
    (elements, block_size) in [
      (0L, 256),
      (2147483648L, 256),
      (1L, 0),
      (1L, 1025),
    ] {
    try
      @rectified_flow_update_cuda_source.render(elements~, block_size~)
    catch {
      @rectified_flow_update_cuda_source.SourceError::InvalidGeometry => ()
    } noraise {
      _ => fail("invalid geometry admitted")
    }
  }
}
