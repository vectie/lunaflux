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
      #|  "50955a382fc0dfec1602f2c46884c809b3b046f5d83db1774993fca46f5a0200",
      #|  "fb8eb34493b0994e6929ccd1f49555b03172a0c4e2e7f9bbec829de9ec12f02a",
      #|  "6a129d18915eaf806b3c6fd16acb476f5b8872a025977c1e828fc106b3b0a4a9",
      #|  "9e3e7315433504c37811818f957e1b181443b4a4f4f13d3e051ddbd3dc9aa120",
      #|  "a63abd4035ed4f9cbc6e75c1a5ebf13ed3a22439f8d79264bef36669b0642e34",
      #|  "3dee6c86d29a2f2e534ac3f26c9260fc5a838722e33bdc52a638160b939abf15",
      #|  "131d443af625130c479d31228a77fd39b8745a9cb93268f10b6ad7cb4c270bc5",
      #|  "bebc1d013c2a6bb476467aebf28b9eddf5097855992b82468454c0bcfa83cb99",
      #|  "f192286e70ae3e733cd7e7ba5d3592b96df5752b5683a706895780c8eb7aa7fc",
      #|  "320a132125a5263769c0647fee5cfe000259b7826db40cacc9a17e7374bc9502",
      #|  "9554a5bb104754d01e65043682058015d4ac967b2239b1cc63d70e82686ca5be",
      #|  "9a6b034a8224788a3a5ad36f2865ed03caf5791272995df4853ac389d64fda08",
      #|  "aa991c86209a52b958fba4e43efdb658ffcbb94d477a285c3bfd73f351da3cb5",
      #|  "ed466d8660f6c8e7a51a70b943a1fa39759e46149c69386cae975c6e24711fbb",
      #|  "8390e6f0a86676d2eec90e0de234b275f82f3c47766e60faa9768e3cc8cfd752",
      #|  "23784450e753afa101cb95e74ee75f234663796ff16961d8a95fc1aeae808ad1",
      #|  "853d51a32eb86675f9c5742692f35121b52ca942056a1bc228361faff962e069",
      #|  "b26b246dda3942521bf415bd013f74d6f7b3783b53c32e18cdeb4cb042336ca8",
      #|]
    ),
  )
}
