///|
test "stream capture lowers mean not weighted normalization or a vocabulary head" {
  let p = @precision.StreamMeanCapturePrecision::new(
    stream_count=4,
    hidden_width=257,
    maximum_rows=3,
    segment_count=3,
    maximum_output_bytes=4626L,
  )
  let s = @stream_mean_capture_cuda_source.stream_mean_capture_source(
    p,
    symbol="capture",
  )
  inspect(s.grid_x(), content="4")
  inspect(s.symbol(0), content="capture_segment_0")
  let code = @utf8.decode(s.source_bytes())
  assert_true(code.contains("__fdiv_rn(sum,4.0f)"))
  assert_true(code.contains("row*3ULL+2ULL"))
  assert_false(code.contains("rsqrt"))
  assert_false(code.contains("hc_head"))
  for name in ["", "bad-name", "1capture"] {
    try
      @stream_mean_capture_cuda_source.stream_mean_capture_source(
        p,
        symbol=name,
      )
    catch {
      @precision.PrecisionPlanError::IllegalPlacement => ()
      _ => fail("unexpected symbol error")
    } noraise {
      _ => fail("invalid source symbol")
    }
  }
}
