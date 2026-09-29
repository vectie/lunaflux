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
      #|  "e14d8a11d6cf5ba6d04309eebb554f3aa90446b0a86cdc1e0a85b8c446c80a0e",
      #|  "20b02fb94791a7f641bfc2d2ed5b9042b2785c9b95af4e6bd9d4d7df73a84027",
      #|  "4be6d6d663a4958a0467bdbd2f22f3beed15428b101e9b88fe7f8a356803d489",
      #|  "36835dca319b13402479c9d42f08dcfde917efe8167d725e8782f3519fae154e",
      #|  "4448abf80dbcb9142b092d3cb34f806966ef27d8648547bb5b0ef6c5337d57bc",
      #|  "ddcb43e81afc3a8d3585e7baa457e425561edcb1252d6273db81187002fb24ba",
      #|  "fd4e5bb9de0c4e965353e661ab85e12291bce29472f8940d18bc31fda40bcb91",
      #|  "e6d9b26c50576882152ca82d454fc8fe8e6e04ab4d41db7d1b4920aeb1d55269",
      #|  "1bafea65b9b0e6095af5b6bedfa118d7e2e9e228ff4c7a581086dd9735b40b44",
      #|  "b900840c7115f0ef8a40fdefa2b7e8bcfebbaa66e40eca06ff7dd32014ca6cea",
      #|  "b8ba8f415be4e3501eefd1b61d99748b8361cfdea9c1ba0076638663e6cf44d0",
      #|  "7ee712de83285183b53f2180b8618e545f65208242a9a23c2d05be34fc6bd686",
      #|  "ac92848a843a000390ca11e27c9b3a043771d40873e5fc910b5132a7fdacd2a7",
      #|  "b8ef6f0eeada9f6724369e873b05b9cb35bbc03d2c85c8c74a2a9b9741dad4a9",
      #|  "6a25de9c4dfb3cfd58c8b93cefc32f38db7c6d25c6de75b1df4889e420267369",
      #|  "5b4d71ba3a6c7e5b23b3d22e07a2ee1d9a9065644a4d2d3eae4d8a060c5b4c87",
      #|  "76eab473e0076e40093668fe62002f3a4ae8a790dff197f96225c3a999c5999a",
      #|  "65cb1f25667a0f7cf6f386bce2649ddd7a7fbb86494b1bdc89ce693d18cf484a",
      #|]
    ),
  )
}
