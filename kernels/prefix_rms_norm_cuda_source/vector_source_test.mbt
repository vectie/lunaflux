///|
test "vector fold carries launch geometry handles tails and preserves default ABI" {
  let selected = @prefix_rms_norm_cuda_source.prefix_rms_norm_bf16_with_optional_suffix_copy_source(
    prefix_width=4097,
    suffix_width=3,
    maximum_rows=2,
    epsilon=0.000001,
    function_symbol="vector_norm",
    square_law=VectorizedTreeFloat32(partitions=512, vector_width=4),
  )
  assert_eq(selected.launch_threads(), 512)
  let text = @utf8.decode(selected.source_bytes())
  assert_true(text.contains("base + component < 4097"))
  assert_true(text.contains("blockDim.x != 512"))
  assert_true(text.contains("column += 512"))
  assert_eq(
    selected.numerical_contract(),
    VectorizedTreeFloat32PrefixReductionBf16RoundAndBitExactSuffixCopyV1(
      VectorizedTreeFloat32(partitions=512, vector_width=4),
    ),
  )
  for
    law in [
      @precision.SquareReductionLaw::OrderedFloat32,
      CompensatedFloat32,
      VectorizedTreeFloat32(partitions=513, vector_width=4),
    ] {
    try
      @prefix_rms_norm_cuda_source.prefix_rms_norm_bf16_with_optional_suffix_copy_source(
        prefix_width=4096,
        suffix_width=0,
        maximum_rows=1,
        epsilon=0.000001,
        function_symbol="unsupported",
        square_law=law,
      )
    catch {
      UnsupportedSquareReductionLaw => ()
      _ => fail("unexpected prefix norm error")
    } noraise {
      _ => fail("unsupported fold silently substituted")
    }
  }
}
