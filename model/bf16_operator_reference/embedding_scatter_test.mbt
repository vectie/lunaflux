///|
test "image insertion uses absolute causal positions and copies raw BF16 words" {
  let plan = @effects.EmbeddingScatterProgram::new(
    width=4,
    maximum_rows=3,
    maximum_positions=16,
    maximum_features=2,
    owners=2,
  ).unwrap()
  let feature_map = Array::make(8, -1)
  feature_map[5] = 1
  feature_map[7] = 0
  let features = [0x7f81U, 0x8000U, 0xffffU, 1U, 0x1234U, 0U, 0x3f80U, 0xbf80U]
  let residual = FixedArray::make(12, 0xdeadU)
  @bf16_operator_reference.embedding_scatter_words_into(
    plan,
    [5, 0, 7],
    feature_map[:],
    features[:],
    residual,
    rows=3,
    feature_rows=2,
    prefill=true,
  )
  assert_eq(residual[:].to_owned(), [
    0x1234U, 0U, 0x3f80U, 0xbf80U, 0xdeadU, 0xdeadU, 0xdeadU, 0xdeadU, 0x7f81U, 0x8000U,
    0xffffU, 1U,
  ])
  residual.fill(0xbeefU)
  @bf16_operator_reference.embedding_scatter_words_into(
    plan,
    [5, 0, 7],
    feature_map[:],
    features[:],
    residual,
    rows=3,
    feature_rows=2,
    prefill=false,
  )
  assert_true(residual[:].all(word => word == 0xbeefU))
  feature_map[5] = 2
  assert_true(
    try
      @bf16_operator_reference.embedding_scatter_words_into(
        plan,
        [5, 0, 7],
        feature_map[:],
        features[:],
        residual,
        rows=3,
        feature_rows=2,
        prefill=true,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
