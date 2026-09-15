///|
test "cooperative grouped fold preserves source across effect-plan reuse" {
  let digests = []
  for rows in [1, 8, 32] {
    for context in [1, 65, 4096] {
      let lowering = source_test_decode_lowering(
        candidate_id=420,
        batch_rows=rows,
        context_tokens=context,
      )
      let emitted = emit_cuda_decode_attention_tile_source(
        lowering,
        source_test_abi(),
        "cooperative_effect_contract",
        1048576,
      )
      digests.push("\{rows}/\{context}:" + emitted.digest().as_hex())
    }
  }
  inspect(
    digests.join("\n"),
    content=(
      #|1/1:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
      #|1/65:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
      #|1/4096:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
      #|8/1:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
      #|8/65:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
      #|8/4096:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
      #|32/1:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
      #|32/65:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
      #|32/4096:8dc568e013b4a8d42a4df89b43c4e4fb94cf1b22fb12a56f5e195eaba76971de
    ),
  )
}
