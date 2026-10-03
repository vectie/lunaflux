///|
test "packed physical refinement emits deterministic full ingress source" {
  let digests : Array[String] = []
  for head_dimension in [32, 64, 128] {
    for maximum_tokens in [8, 31, 32] {
      let candidate = qwen_fused_test_candidate(
        head_dimension~,
        maximum_tokens~,
      )
      digests.push(candidate.source_digest().as_hex())
      let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
        candidate,
        fused_test_limits(),
      )
      digests.push(production.source_digest().as_hex())
    }
  }
  debug_inspect(
    digests,
    content=(
      #|[
      #|  "f8ad274c7cee09f31ea34c9e0f1f8b4a00344feeee1466e7d9bd0e5760457d7b",
      #|  "b81ae5806db9ba37ccfcc5003e1f48b344ca14d4c2f6e80c5cb894fe5332a898",
      #|  "76ff849ad8aef7bfefaf878705fb5354c98d069b0b2a85fd0e6c2b97a6fa245b",
      #|  "6e61af5eab7e6a9eb14aa94d1842180454fcf97f84a7cb41c71d085d6594b311",
      #|  "8a2f1774851001491495af83ab403e14de774a7e2990a1ba18df0ea777f0a1dd",
      #|  "244f481b3140117880e49dc5d1a8df65553ee37918d6a15e246b28f7fa62de92",
      #|  "069199bc04cce41a300957cc7bfe78482972ab9815805397ebe84d58978d63a1",
      #|  "58b14988ac97fbcfd1d380524c828b88c85197237db3721fcf6b86874420dbce",
      #|  "7a0218ea7822a1e32a01b4d75f272e4cdb9f6b514318f84da091c3e960f26b70",
      #|  "f059b68fac27475a28a254511e8317246bd59b7d6732652e7940c3301b481f4e",
      #|  "029f9d978342f20a0bc3f088df01cbe4f872af47e950161d6b4cb550be66d1ac",
      #|  "37d777dc34065cfac5fb5082ad7c5aa651433774153fb9846278a11e7b391a9f",
      #|  "65ea175b784af0c601ff5c6d1c3e02ce5617783084b1b16c9b2cbe61d495781a",
      #|  "14adf62ba822aaafc42f47621c6a1a32cfed00589a7fc1a6796c7135682a31dd",
      #|  "328dad69aee0a1f90948a00f26ef326d3901dd71c6c1633169afb4dc883adcb4",
      #|  "b96b07c322968b41a7081a239a73b13f4efb8706ecf90021f6f2790fc1222cad",
      #|  "63113c89974fa86c9a6164e46c096285d3be23d7007276b6f9d37da634d0663a",
      #|  "1ed1fa72d711836fc2e6979f62a0e44b7d54c68a6a9d6f7e25097cdef5f12f07",
      #|]

    ),
  )
}
