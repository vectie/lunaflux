///|
test "physical refinement preserves full ingress source" {
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
      #|  "a35913ae26a33bd274ea9da84882ee4fa44c650c3ace880686ddc032194c46f8",
      #|  "cf69c4bb304d7db69707fc5158100fea14e454e4898ecfb329de58cbd3407c70",
      #|  "710da2a580932a2523e4f247f61b08dcd689668371c12d30b81cd6ceaba00a75",
      #|  "45277a59cf35c55887098daacaf46e2cdfb167b60e16348656838ed03a1186b0",
      #|  "3ebaf2f9eefba45863a488a87e1028c5924332b51b3e867d226eb6fa05153482",
      #|  "83cc108d53929c7ecb46577ff070b4a95e8539321059f3dfc7f706a98e7427fb",
      #|  "09af999d0bced414a62fa5ec1079e9efc5610257d3e9e3c1e245ec89346f2eea",
      #|  "5c204fd672bcb1156c9df534ba66241b543afff067fb103ed4f6c33f9fcea174",
      #|  "aa0b0bdc54257825146c87ed61de74057173597a89c4e1bd4ccc1eb86a7493a4",
      #|  "ec0ab3757c7c2b87cc447d164edd96b1088fcc562984a40ace75ace7cb2b05ea",
      #|  "2faa62db3baf850d1d3c670382c7f1f0d71cc145bb2a741f79af3cb793efd77a",
      #|  "c9662de88393c7f944c775d71381b12d1813805bbd7def6d52ffb11d52bd4f11",
      #|  "8c1a6f542a6cdeebe88d8079113261f4885a6c0bc802cca1b824ae8ed471e6f4",
      #|  "510f8eda35ba73d26f359edc461a2c9740ddd20bcf7651dbb7daaa64dae3ac36",
      #|  "5f980b55d3482cd4e1add70f6e63986116bfc17b21698b188482b2dd66c0718c",
      #|  "e66fd17cd570d73f3449f641fd81eb8e7f7ec45e9cc00e437ff778470f659e65",
      #|  "4e0a56d521a746badbc6af0f09adadfd77a533505a0d13dcbb6286d16661d719",
      #|  "aed028ca6ab97c96d4419305a4c5c234f5ca4c2d080b8bfe190778fbad78803e",
      #|]
    ),
  )
}
