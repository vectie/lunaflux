///|
test "raw block projection owns fixed scratch and arithmetic-selected launch" {
  let selected = block_fp8_ue8m0_projection_cuda_source(
    input_width=65536,
    output_width=128,
    maximum_rows=3,
    function_symbol="matrix_projection",
    scaled_dot_law=UnscaledBlockMatrixFloat32(
      block_columns=128,
      publication=SeparateMultiplyAdd,
    ),
  )
  assert_eq(selected.launch(), (3, 4, 128))
  let text = @utf8.decode(selected.source_bytes())
  assert_true(text.contains("encoded[128]"))
  assert_true(text.contains("Close every reader's scratch lifetime"))
  assert_true(text.contains("_block_dot<128,65536>"))
  assert_true(text.contains("mma.sync.aligned.m16n8k32"))
  assert_true(text.contains("__fadd_rn(total[i],block[i])"))
  let baseline = block_fp8_ue8m0_projection_cuda_source(
    input_width=65536,
    output_width=128,
    maximum_rows=3,
    function_symbol="matrix_projection",
  )
  assert_eq(baseline.launch(), (3, 1, 256))
  assert_true(!@utf8.decode(baseline.source_bytes()).contains("mma.sync"))
  for
    law in [
      @precision.ScaledDotReductionLaw::UnscaledBlockMatrixFloat32(
        block_columns=64,
        publication=SeparateMultiplyAdd,
      ),
      UnscaledBlockMatrixFloat32(
        block_columns=128,
        publication=FusedMultiplyAdd,
      ),
    ] {
    try
      block_fp8_ue8m0_projection_cuda_source(
        input_width=256,
        output_width=128,
        maximum_rows=3,
        function_symbol="unsupported",
        scaled_dot_law=law,
      )
    catch {
      UnsupportedArithmetic => ()
      e => raise e
    } noraise {
      _ => fail("unsupported arithmetic silently substituted")
    }
  }
}
