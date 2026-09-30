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
      #|440/1/65:3d49c550641c424c7992480d01988b8f4098b938e59740a3e853fafea5f64007
      #|440/1/4096:6bfc03ac95e9f5cd3c70013e94fd26219ce76ef635c6e81092b4b9ea6adcb08d
      #|440/8/65:3d49c550641c424c7992480d01988b8f4098b938e59740a3e853fafea5f64007
      #|440/8/4096:3d49c550641c424c7992480d01988b8f4098b938e59740a3e853fafea5f64007
      #|440/32/65:3d49c550641c424c7992480d01988b8f4098b938e59740a3e853fafea5f64007
      #|440/32/4096:3d49c550641c424c7992480d01988b8f4098b938e59740a3e853fafea5f64007
      #|441/1/65:6a54f43cd0943d0557e6ac2d42df98940f2df374b614adbc69895a3c50cb0e60
      #|441/1/4096:d2a39d850818ffa5ff0bad114ee725b22d9bc6b152d3b62a0222f65efa6d1132
      #|441/8/65:6a54f43cd0943d0557e6ac2d42df98940f2df374b614adbc69895a3c50cb0e60
      #|441/8/4096:6a54f43cd0943d0557e6ac2d42df98940f2df374b614adbc69895a3c50cb0e60
      #|441/32/65:6a54f43cd0943d0557e6ac2d42df98940f2df374b614adbc69895a3c50cb0e60
      #|441/32/4096:6a54f43cd0943d0557e6ac2d42df98940f2df374b614adbc69895a3c50cb0e60
    ),
  )
}
