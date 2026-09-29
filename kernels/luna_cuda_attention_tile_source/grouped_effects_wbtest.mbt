///|
test "grouped decode effect extraction preserves generated sources" {
  let digests : Array[String] = []
  for id in [430, 440, 441] {
    for rows in [1, 8, 32] {
      for context in [65, 4096] {
        let lowering = source_test_decode_lowering(
          candidate_id=id,
          batch_rows=rows,
          context_tokens=context,
        )
        let emitted = emit_cuda_grouped_split_decode_attention_tile_source(
          lowering,
          source_test_abi(),
          "grouped_effect_contract",
          1048576,
        )
        digests.push("\{id}/\{rows}/\{context}:" + emitted.digest().as_hex())
      }
    }
  }
  inspect(
    digests.join("\n"),
    content=(
      #|430/1/65:fe99a4466003218636156e381c30a39285344b813b16b36cc8c52ccabd131959
      #|430/1/4096:512dc092e7566c8e785f2096c2a36ba1820617209463bb107d7b3aea2c338a28
      #|430/8/65:fe99a4466003218636156e381c30a39285344b813b16b36cc8c52ccabd131959
      #|430/8/4096:fe99a4466003218636156e381c30a39285344b813b16b36cc8c52ccabd131959
      #|430/32/65:fe99a4466003218636156e381c30a39285344b813b16b36cc8c52ccabd131959
      #|430/32/4096:fe99a4466003218636156e381c30a39285344b813b16b36cc8c52ccabd131959
      #|440/1/65:d611c08757353a562bfcfc78edf6c02e6f8f346796679c7989a17830d0e72819
      #|440/1/4096:52fcb4fb9708ca756147c635e657248843a2b6a9335babc06611c1ff6c09303e
      #|440/8/65:d611c08757353a562bfcfc78edf6c02e6f8f346796679c7989a17830d0e72819
      #|440/8/4096:d611c08757353a562bfcfc78edf6c02e6f8f346796679c7989a17830d0e72819
      #|440/32/65:d611c08757353a562bfcfc78edf6c02e6f8f346796679c7989a17830d0e72819
      #|440/32/4096:d611c08757353a562bfcfc78edf6c02e6f8f346796679c7989a17830d0e72819
      #|441/1/65:cdb9b4b41368594ed2be07e61c9270d85de15dea86f29f9cd54107a35f0866b9
      #|441/1/4096:a00fb1832be5d94a1c57feb9a062b3d4851b3f1f135ba14ef2341b2fec7b3d32
      #|441/8/65:cdb9b4b41368594ed2be07e61c9270d85de15dea86f29f9cd54107a35f0866b9
      #|441/8/4096:cdb9b4b41368594ed2be07e61c9270d85de15dea86f29f9cd54107a35f0866b9
      #|441/32/65:cdb9b4b41368594ed2be07e61c9270d85de15dea86f29f9cd54107a35f0866b9
      #|441/32/4096:cdb9b4b41368594ed2be07e61c9270d85de15dea86f29f9cd54107a35f0866b9
    ),
  )
}
