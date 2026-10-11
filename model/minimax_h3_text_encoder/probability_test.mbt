///|
test "encoder carries immutable attention precision without changing its default" {
  let default = TextEncoderPlan::new(rows=140, epsilon=0.000001)
  assert_eq(
    default.attention_probability_law(),
    @precision.AttentionProbabilityLaw::OrderedOnlineF32,
  )
  let block = TextEncoderPlan::new(
    rows=140,
    epsilon=0.000001,
    probability_law=OnlineBf16(block_tokens=64),
  )
  assert_eq(
    block.attention_probability_law(),
    @precision.AttentionProbabilityLaw::OnlineBf16(block_tokens=64),
  )
  assert_eq(default.workspace_regions(), block.workspace_regions())
  for
    law in [
      @precision.AttentionProbabilityLaw::GlobalF32,
      OnlineBf16(block_tokens=0),
    ] {
    try
      TextEncoderPlan::new(rows=140, epsilon=0.000001, probability_law=law)
    catch {
      InvalidProbabilityLaw => ()
      _ => fail("wrong error")
    } noraise {
      _ => fail("invalid encoder probability accepted")
    }
  }
}
