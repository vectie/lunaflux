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
      #|440/1/65:018cb5b37756d14415c7f88f7601aa49053a95eb465642fdbabe1594518244ca
      #|440/1/4096:223c11c971b40cb57c160e304b5abe84a70e8ad22e2d58ed80387baf5570df0c
      #|440/8/65:018cb5b37756d14415c7f88f7601aa49053a95eb465642fdbabe1594518244ca
      #|440/8/4096:018cb5b37756d14415c7f88f7601aa49053a95eb465642fdbabe1594518244ca
      #|440/32/65:018cb5b37756d14415c7f88f7601aa49053a95eb465642fdbabe1594518244ca
      #|440/32/4096:018cb5b37756d14415c7f88f7601aa49053a95eb465642fdbabe1594518244ca
      #|441/1/65:108893a7a3bb6ec90a8f1ab6da1ca7847be42ee51a09bb66c2aa0bff13e21102
      #|441/1/4096:3c8897ec59a6718e00a649a4b79afd0a84795b0b079c6e6db257352843c67318
      #|441/8/65:108893a7a3bb6ec90a8f1ab6da1ca7847be42ee51a09bb66c2aa0bff13e21102
      #|441/8/4096:108893a7a3bb6ec90a8f1ab6da1ca7847be42ee51a09bb66c2aa0bff13e21102
      #|441/32/65:108893a7a3bb6ec90a8f1ab6da1ca7847be42ee51a09bb66c2aa0bff13e21102
      #|441/32/4096:108893a7a3bb6ec90a8f1ab6da1ca7847be42ee51a09bb66c2aa0bff13e21102
    ),
  )
}
