///|
test "scalar sampling exactly matches canonical parameter replay" {
  let limits = @sampling.SamplingLimits::new(1, 5)
  let inference = inference_limits()
  let parameters = @inference.SamplingParameters::sample(
    temperature=0.75,
    top_k=3,
    top_p=0.9,
    inference,
  )
  let seed = @inference.SamplingSeed::new(0x123456789abcdef0UL)
  let canonical = @sampling.SamplerScratch::new(5, limits)
  let scalar = @sampling.SamplerScratch::new(5, limits)
  for index in 0..<64 {
    let expected = @sampling.stochastic_sample_at(
      [3.0, 2.0, 1.0, 0.0, -1.0],
      parameters,
      limits,
      seed,
      index,
      canonical,
    )
    let actual = @sampling.stochastic_sample_at_scalars(
      [3.0, 2.0, 1.0, 0.0, -1.0],
      temperature=0.75,
      has_top_k=true,
      top_k=3,
      has_top_p=true,
      top_p=0.9,
      limits,
      seed,
      index,
      scalar,
    )
    assert_eq(actual, expected)
  }
}

///|
test "scalar sampling rejects malformed optional wire values" {
  let limits = @sampling.SamplingLimits::new(1, 2)
  let scratch = @sampling.SamplerScratch::new(2, limits)
  let seed = @inference.SamplingSeed::new(1UL)
  try
    @sampling.stochastic_sample_at_scalars(
      [1.0, 0.0],
      temperature=1.0,
      has_top_k=true,
      top_k=0,
      has_top_p=false,
      top_p=0.0,
      limits,
      seed,
      0,
      scratch,
    )
  catch {
    InvalidDistribution => ()
    _ => fail("unexpected scalar validation error")
  } noraise {
    _ => fail("invalid scalar top-k was admitted")
  }
}
