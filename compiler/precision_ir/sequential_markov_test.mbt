///|
test "dependent sampling plan retains BF16 storage and pre-normalization confidence" {
  let p = @precision_ir.SequentialMarkovPrecision::new(
    vocabulary_size=17,
    hidden_width=8,
    embedding_width=4,
    maximum_rows=5,
    temperature=0.0,
    maximum_workspace_bytes=60L,
  )
  assert_eq(p.geometry(), (17, 8, 4, 5))
  assert_eq(p.weight_bytes(), (136L, 136L, 24L))
  assert_eq(p.workspace_bytes(), 60L)
  assert_eq(p.hidden_bytes(), 80L)
  assert_eq(p.logits_bytes(), 340L)
  for (temperature, budget) in [(-1.0, 60L), (0.0, 59L)] {
    try
      @precision_ir.SequentialMarkovPrecision::new(
        vocabulary_size=17,
        hidden_width=8,
        embedding_width=4,
        maximum_rows=5,
        temperature~,
        maximum_workspace_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid chain accepted")
    }
  }
}
