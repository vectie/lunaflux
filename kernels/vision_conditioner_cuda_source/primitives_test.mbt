///|
test "counted vision primitives preserve existing default source and add live-row ABIs" {
  let plain = @utf8.decode(
    @vision_conditioner_cuda_source.norm_source(rows=16, width=8, symbol="norm"),
  )
  let counted = @utf8.decode(
    @vision_conditioner_cuda_source.norm_source(
      rows=16,
      width=8,
      symbol="norm",
      counted=true,
    ),
  )
  assert_true(!plain.contains("counts"))
  assert_true(counted.contains("const int* counts,"))
  assert_true(counted.contains("row>=counts[3]"))
  assert_true(
    @utf8.decode(
      @vision_conditioner_cuda_source.linear_source(
        rows=16,
        input_width=8,
        output_width=4,
        activation=1,
        symbol="linear",
        counted=true,
      ),
    ).contains("i>=counts[3]*4"),
  )
  assert_true(
    @utf8.decode(
      @vision_conditioner_cuda_source.residual_source(
        rows=16,
        width=8,
        symbol="residual",
        counted=true,
      ),
    ).contains("i<counts[3]*8"),
  )
}
