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
      #|430/1/65:285b018dfeef7d5075984da1f4e50889d6b9e8e6249bc1aeecfc97087189bfab
      #|430/1/4096:5e137f1935fca2674586c9adbc18ebfa3d2ce58f0e8cc0b17e390ee51ef2e7b4
      #|430/8/65:285b018dfeef7d5075984da1f4e50889d6b9e8e6249bc1aeecfc97087189bfab
      #|430/8/4096:285b018dfeef7d5075984da1f4e50889d6b9e8e6249bc1aeecfc97087189bfab
      #|430/32/65:285b018dfeef7d5075984da1f4e50889d6b9e8e6249bc1aeecfc97087189bfab
      #|430/32/4096:285b018dfeef7d5075984da1f4e50889d6b9e8e6249bc1aeecfc97087189bfab
      #|440/1/65:5ffa649a9450181d88e2e970273066a4daf46bbf02f27d2560fe95081c07ac20
      #|440/1/4096:e86e54c44d32b02d79c8875e5e40c9a40f401303ab0f31ec4672b82fd344965e
      #|440/8/65:5ffa649a9450181d88e2e970273066a4daf46bbf02f27d2560fe95081c07ac20
      #|440/8/4096:5ffa649a9450181d88e2e970273066a4daf46bbf02f27d2560fe95081c07ac20
      #|440/32/65:5ffa649a9450181d88e2e970273066a4daf46bbf02f27d2560fe95081c07ac20
      #|440/32/4096:5ffa649a9450181d88e2e970273066a4daf46bbf02f27d2560fe95081c07ac20
      #|441/1/65:56cc68d89ef47b7f24b9eb8bd805fb542862bacb035603c84bc8f7c368cc0f8f
      #|441/1/4096:0b200f00a240d6532154dce3f722eef5b0e3bc996c49ef441ccd26af66b61542
      #|441/8/65:56cc68d89ef47b7f24b9eb8bd805fb542862bacb035603c84bc8f7c368cc0f8f
      #|441/8/4096:56cc68d89ef47b7f24b9eb8bd805fb542862bacb035603c84bc8f7c368cc0f8f
      #|441/32/65:56cc68d89ef47b7f24b9eb8bd805fb542862bacb035603c84bc8f7c368cc0f8f
      #|441/32/4096:56cc68d89ef47b7f24b9eb8bd805fb542862bacb035603c84bc8f7c368cc0f8f
    ),
  )
}
