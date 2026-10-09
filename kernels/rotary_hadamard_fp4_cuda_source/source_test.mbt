///|
test "Hadamard reads a complete prior stage and FP4 simulation uses RNE power-two scales" {
  let plan = @precision.RotaryHadamardFp4Precision::new(
    head_count=2,
    head_width=128,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=1024,
    frequency=Base(theta=10000.0),
    simulation_block_width=32,
    maximum_workspace_bytes=3072L,
  )
  let source = @rotary_hadamard_fp4_cuda_source.rotary_hadamard_fp4_source(
    plan,
    symbol_prefix="rotate_sim",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(
    text.contains("output_words[base + col] = input_words[base + col]"),
  )
  assert_true(text.contains("__shared__ float first[128], second[128]"))
  assert_true(text.contains("old_values[col & ~stride]"))
  assert_true(text.contains("__syncthreads(); float *swap"))
  assert_true(text.contains("const float normalization ="))
  assert_false(text.contains("rsqrtf("))
  assert_true(text.contains("0x1.8p-124f"))
  assert_true(text.contains("error == distance && !(code & 1)"))
  assert_false(text.contains("cuda_fp4"))
  assert_eq(source.launch(0), (3, 2, 256))
  assert_eq(source.launch(1), (3, 2, 256))
  assert_eq(source.launch(2), (3, 8, 32))
}
