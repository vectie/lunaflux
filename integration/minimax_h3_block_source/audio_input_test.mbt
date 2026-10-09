///|
test "audio latent source joins unpacking and affine with exact stereo geometry" {
  let source = @minimax_h3_block_source.build_audio_input(
    time=207,
    max_elements=13248L,
  )
  assert_eq(source.value_bytes, 52992L)
  assert_eq(source.symbols.length(), 2)
  assert_eq(source.symbols[0], "lunaflux_dense_permutation_u32_v1")
  let text = @utf8.decode(source.source)
  assert_true(text.contains("__fmul_rn"))
  assert_true(text.contains("__fadd_rn"))
  try
    @minimax_h3_block_source.build_audio_input(time=207, max_elements=13247L)
  catch {
    _ => ()
  } noraise {
    _ => fail("short unpack budget accepted")
  }
}
