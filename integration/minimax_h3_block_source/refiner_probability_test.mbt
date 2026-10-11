///|
test "refiner explicit probability law preserves projection and ABI" {
  let config = @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
    variant=Fl2va,
    checkpoint_format=DiffusersV1,
  )
  let model = @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      config,
      @spec.ContentDigest::from_sha256(String::make(64, '7')),
    ),
  )
  let baseline = @minimax_h3_block_source.build_refiner(
    model,
    text_rows=3,
    block_norm_epsilon=config.norm_epsilon(),
    function_prefix="h3_refiner",
  )
  let candidate = @minimax_h3_block_source.build_refiner(
    model,
    text_rows=3,
    block_norm_epsilon=config.norm_epsilon(),
    function_prefix="h3_refiner",
    probability_law=GlobalUnnormalizedBf16,
  )
  assert_eq(baseline.symbols, candidate.symbols)
  let source = @utf8.decode(candidate.source)
  assert_eq(
    source.replace(
      old="__bfloat162float(__float2bfloat16_rn(probability))",
      new="probability",
    ),
    @utf8.decode(baseline.source),
  )
  if @env.get_env_var("LUNA_EXPORT_REFINER_BUNDLE") == Some("1") {
    println("LUNA_REFINER_BUNDLE_BEGIN")
    println(source)
    println("LUNA_REFINER_BUNDLE_END")
  }
}
