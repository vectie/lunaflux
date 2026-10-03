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
      #|  "296c297e06fa8c5b195ad1e2571b5200d72eaf49f046356709711cc3d631ea12",
      #|  "ca7fe8eb5f271db3a1b4fc63674f0d8f413382dd86838b5de862bbb2be3de830",
      #|  "acf397082a2a63b6948bea71cecf737d20b18ee6d884044f7312e0839e6bfbc5",
      #|  "c3de52db732bf272fb64e6458ac8a28724472db47dffaac572680aa17f5cd09f",
      #|  "55b6b661e97f1a4ccd5e2d408ea4e8eae527f11fcb695d7a2a850bb03375f4d7",
      #|  "ee5a2db010431f4cf3c7c74e26f48516c3b40d597cd7732660974e129481cda9",
      #|  "046e78a381f11c30295930376de378ad0f5edb3490733f4a83d276937b960e33",
      #|  "1a454c8d828cc93b56532f9728bb9c9bc6360822b67c685dca40d9fee7fa4431",
      #|  "4eb371ee64ed3baac747a944d0ae69d7793cecd879babcd356f71d7a2e48e035",
      #|  "49d6770047ced1463644c32f740010b03e723cbe9a408bfff925b1e9e1e284df",
      #|  "fe66d1190fe4a1285cbb68103aec954269f1bef6a0591d255e14b6a434de714b",
      #|  "5b76c128445bd90c1a077632266a85d512c383239427258dcd29a3e5902b4d92",
      #|  "22b0c3af9ceeb7e3720e8439ce88f23672f9e76dcc740f443082316762a816a1",
      #|  "1f6b87a3c9cba3a5df6fc4a535c9805ef9461cbe5c3641cdc2cbf5c428d3234d",
      #|  "9e8281c02f14a1341d0d2a4b23d36d452b71d54e458feafd77f8e29f668ddb35",
      #|  "4a6aadd1ec773a8e01dec305e8d6b9ef8af087aa02082bed535966a17f2ac115",
      #|  "59899d78041773b2605403b3f760a1f4052fc83201835f53ed06547d5b7d40bb",
      #|  "6385b93144fc03ed3a8be123383a528f0cce1b4d9e1881ec4cbfc1cc557e51ae",
      #|]
    ),
  )
}
