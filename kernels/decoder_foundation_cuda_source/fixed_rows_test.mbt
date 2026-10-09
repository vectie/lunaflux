///|
test "fixed-row BF16 RMSNorm source is deterministic and exact" {
  let source = rms_norm_bf16_fixed_rows_source(
    rows=207L,
    hidden_size=5376,
    epsilon=0.00001,
    function_symbol="lunaflux_joint_diffusion_audio_final_rms_norm_bf16_v1",
  )
  assert_eq(source.operation(), RmsNormBf16)
  assert_eq(source.numerical_contract(), OrderedFloat32ReductionBf16RoundV1)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("if (row >= 207ULL) return;"))
  assert_true(text.contains("row * 5376ULL"))
  assert_true(text.contains("__shared__ float partial[256]"))
  assert_true(text.contains("__fmul_rn"))
  assert_true(text.contains("__fadd_rn"))
  assert_true(text.contains("__fdiv_rn"))
  assert_true(text.contains("__float2bfloat16_rn"))
  assert_false(text.contains("counts"))
  let repeated = rms_norm_bf16_fixed_rows_source(
    rows=207L,
    hidden_size=5376,
    epsilon=0.00001,
    function_symbol="lunaflux_joint_diffusion_audio_final_rms_norm_bf16_v1",
  )
  assert_eq(source.source_bytes(), repeated.source_bytes())
}

///|
test "fixed-row BF16 RMSNorm validation fails closed" {
  let cases = [
    (0L, 1, 0.00001, "valid_symbol"),
    (2147483648L, 1, 0.00001, "valid_symbol"),
    (1L, 0, 0.00001, "valid_symbol"),
    (1L, 1, 0.0, "valid_symbol"),
    (1L, 1, 0.00001, "1invalid_symbol"),
  ]
  for index, entry in cases {
    let (rows, hidden_size, epsilon, function_symbol) = entry
    try
      rms_norm_bf16_fixed_rows_source(
        rows~,
        hidden_size~,
        epsilon~,
        function_symbol~,
      )
    catch {
      InvalidRowCount if index == 0 || index == 1 => ()
      InvalidHiddenSize if index == 2 => ()
      InvalidEpsilon if index == 3 => ()
      InvalidFunctionSymbol if index == 4 => ()
      _ => fail("unexpected fixed-row source validation rejection")
    } noraise {
      _ => fail("invalid fixed-row RMSNorm source geometry was admitted")
    }
  }
}
