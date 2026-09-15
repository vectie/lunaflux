///|
test "query-owned effect extraction preserves all existing generated sources" {
  let digests : Array[String] = []
  for id in [318, 319, 320, 321, 322, 323, 324, 1003, 1004, 1005, 1006, 1008] {
    for dense in [false, true] {
      for partitioned in [false, true] {
        let lowering = source_test_lowering(
          true,
          100000L,
          2,
          candidate_id=Some(id),
          dense_current=dense,
          partitioned~,
        )
        let emitted = emit_cuda_query_owned_attention_source(
          lowering,
          source_test_abi(),
          "query_effect_contract",
          1048576,
        )
        digests.push(
          "\{id}/\{dense}/\{partitioned}:" + emitted.digest().as_hex(),
        )
      }
    }
  }
  inspect(
    digests.join("\n"),
    content=(
      #|318/false/false:db42ae94116fecd923d758d486706f3f8f9dcbb137150b3fa1772be85e41e4dc
      #|318/false/true:1540ac2d4e595b897b938bcfc3dc6059aaaee928a6bf114bab99d8061407e7cb
      #|318/true/false:a60aff4e8f2d16ae1c21215b95919a42e1386351d402c9728a25f203cd48d571
      #|318/true/true:f5d2cff5d93102dc7ed5a3f5ea11ea299582ea8280dcc5a8073f7dbe3328f012
      #|319/false/false:7cc2a14fcf55e90fe260b060701a25f8f900b4990a04f99d4441af7743bd5fcd
      #|319/false/true:fc724d3294d8ec5a9a8509d69fc441bc82d0a0ddded5df73baba9184ba751a94
      #|319/true/false:c069b412a9922cd8b0f1d715443bf5de16d8da2186403a48d47bbf5b2ea0314c
      #|319/true/true:8bbc11e7ddde09272cbc14b08ccb8889725cc83dc904044b7ea0b71670cc677d
      #|320/false/false:92fa017fe6f41c3b54d4a7a6ea78b4d07e2171c53ee86048cdb2c57f39b9ac76
      #|320/false/true:f1e240d9133c006c1f5b427a6488c39e0f9d24990644a024b31795e40437955b
      #|320/true/false:8ef5e6fb1622d8cc49d88b0be1c609553242e32d621e0e04a0a6f438a047be4b
      #|320/true/true:ce3ca6c35198c0c232a07c6191accb426a0263ce360fdd9a581240818bfaaa15
      #|321/false/false:cfc785117a8d93b9291feda34a63c22376acd6637c6bb658966270236c853b70
      #|321/false/true:1ab0c4018bb7bea65f83e84efe7ed1a938d5cfe2cd0e4d3c789f9a6bf8405598
      #|321/true/false:74de0df2062a6845988858f36272e8b5e26aa1e59509a908ad7ff796d07bd051
      #|321/true/true:f92de84599363db9e2be7315a6c8644331a53b747555449c2dcc3a714f09b39d
      #|322/false/false:cc05128dbd6b9d3e35eb774ea2d0291b7ac9b6c5d47d754ba4deb4da60e60821
      #|322/false/true:f4c9f7b86914905bdf2960b105d0a8065408fd7aab48f0318e809a2c6dfba6bb
      #|322/true/false:739ee7bf8df6a9e6fa9101cc7cec3e4c55ed767298da70d2381df4f6016de23e
      #|322/true/true:39f927afeeb58df768b9d80e5518fe01fae0f7eeb3a93db1732904cc25f50c5c
      #|323/false/false:b391735a10b411f4c78f1acb08860f5788ffbc02189f029f1cef99fdc9b82dce
      #|323/false/true:403339c1599115ae7d8754442f39c313d189b2b83ec95c7f866adf4494478fb6
      #|323/true/false:4e2f8a67534564f789d012f298a1ed5a5bd288c3d9a897511902c58015338834
      #|323/true/true:f1ec24fb25651418f339775aabe4ed7400593f627c1bd58a6c663c9cd02e10d8
      #|324/false/false:55215c17b5adedb40496b5386db9d33068f7c7d5c7ab7f2d1711564a327cf722
      #|324/false/true:64488dc9ac097956acb4a6862b1322dba79ff7a37a35a276266a5d43f07fe955
      #|324/true/false:a0722563b136fa1a942f85d9eacdebe4e6604364b1b69f7eaf5b10ac24d1f9d5
      #|324/true/true:5c1725de26a7b48de966a5074c6e9b0a4554595a51b86c885f2ffc4729d5e6d3
      #|1003/false/false:a60d6c1c0bc23439da53406647903f1d690d892eb8bcf8c90a5d897600ba5e63
      #|1003/false/true:8718c65c5fa48fdb1c6303d46112bd8ebd6387094da80da0a7e6b7b42a79c780
      #|1003/true/false:969767c8633cee77d4faee9510e71d680ba8cbdd7271435a1333aea44a83e7eb
      #|1003/true/true:4d591b7e8c509fa02929b94e2b43bbe545d7c5a59bfa6f68ba85278e99e5fd3f
      #|1004/false/false:437853e0443691fe4ebe1dd7f2199f587016df52ba633a657c87563b59512d24
      #|1004/false/true:e475ee007ee5c17bd17d2ebf87982ba52703d40d6d6bfac024465d56f7637413
      #|1004/true/false:f6c5296106fdf58a1bb99e32cf8611b2a80a9f786df8343399c101c120549e8e
      #|1004/true/true:64736c847355e22bcf0dcc783788806b598f0ad2f533f543b9829bb5bc8aa435
      #|1005/false/false:eeb384f1ee8830bd365042cfe2214e9e75afa7380b35c6eea9a313bbcab8e3e9
      #|1005/false/true:3648873c660b1b2f233bd5030df992ec92249306a8ffa8cbe1c7fc12e95c3ad7
      #|1005/true/false:7f69e9919b7916219519c6b5e5e752c922582f721be61404c93e9878f9a38610
      #|1005/true/true:84ebc7d3b0248c75a1dd2fa92989707ca88d73832cf8d02cfda6b11c65fa484b
      #|1006/false/false:c0a1905352872ef0c7d548aa086a92f0a1dd09c8e2c2dd56b8fa3f924c2f34c7
      #|1006/false/true:5a659c99f2106c3ecb48e3a521459ad05555ebc7fa7d2e951bba73d004248f74
      #|1006/true/false:071a805fe821a9363d73c44931bfe84904bf90d839be973c7c4fdba549af4c80
      #|1006/true/true:551a9ead768c0b08b83c0d5cc24222b3a1263949015824d2447895ac198fbb29
      #|1008/false/false:2598cddce850a5330717b3e8a8e9163fc5f726026428039716a07ff44902c60b
      #|1008/false/true:fd4535372ca9683bd1c3c0269c378293ae573890e821a80856522e5c4c5f40b0
      #|1008/true/false:60e68cbcca9dacebfa39702bee49e015ebd800f1ec5d44158e1a27f883463731
      #|1008/true/true:3112c04e4178b5fd014ddb7af795ed1db88583f8d047465ee1f6d4436d9d2b42
    ),
  )
}
