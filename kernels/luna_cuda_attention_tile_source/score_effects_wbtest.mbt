///|
test "shared score transfer extraction preserves generated source" {
  let digests : Array[String] = []
  for id in [300, 310, 311, 312, 313, 314, 315, 316, 317] {
    for dense in [false, true] {
      for partitioned in [false, true] {
        let lowering = source_test_lowering(
          true,
          262144L,
          2,
          candidate_id=Some(id),
          dense_current=dense,
          partitioned~,
        ) catch {
          err => fail("\{id}/\{dense}/\{partitioned}: \{err}")
        }
        assert_false(lowering.query_owned_fragments())
        let emitted = emit_cuda_prefill_attention_tile_source(
          lowering,
          source_test_abi(),
          "score_effect_contract",
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
      #|300/false/false:f4080c66b8941d998cf47e95a977ad14e03331ec5b1b89ffd50d1a90b97ef220
      #|300/false/true:d7718b33d5371f1ea58135f1ed94aa9c7e60525f918ca80a1365a3d2d6035461
      #|300/true/false:7b20d5f213b0c9aefa037e6238c6f2e7858fe6d6446427481da170f11457aa9a
      #|300/true/true:0bcd22bfb4006beb2a1bc5712de56fb1b737b09b3d5ce302b7e74e65854b7ca9
      #|310/false/false:7a978127e9a00b26f0d39930426557342a1da3fcdc624c6327ac26a46f7e771d
      #|310/false/true:258ec82ed8ec0faca43e2b97752a311742bf1b049e5f3723829c3009608a8ed1
      #|310/true/false:b15d7c0b586cb973e48cdfd80c53ca837c48cc39aa4e8910ff8a2158f28cf8d7
      #|310/true/true:576bd8a8bc970d941b39add8645fbdb292834f16d7b6113ca2a3d0b64ef08145
      #|311/false/false:eb549826e7b66781a0629a23a397fde0e8746722907e594da9a3d36122d0e044
      #|311/false/true:b967ddea8ed0707e0dd08ff93de87d4b81ee1c49a88fe79ad165cf3808d5c0f6
      #|311/true/false:1e08c4fdc0597e6d24c2763bd381cb3b0ea105ae3b33d524a34c58ffbbc9ab01
      #|311/true/true:110a2a96c244f1af9306dfa6b47b802e262cc9ab41f05773769fc6dd2b9876c5
      #|312/false/false:e466fd26fdea489766527070fd6f1445e06ba1c212b9fb588876d89207bdeda9
      #|312/false/true:74344729b6a2e2c6191cb42d96b39a7eeb7e4946d4c383ebf61508329df6b9d2
      #|312/true/false:b6f7f2657067bd76bbcf4b3e310c23189d95cab13616dcf1094e49f7f2e54b0f
      #|312/true/true:87e62b3ddaed8bd1b048b17c76aab73711535e4c976a22afca2ecff3e247d4c7
      #|313/false/false:00d5adb8bedc72506dccfcf1a361be64b6e6716b9f4d8137e931161c93d360a0
      #|313/false/true:97d3598715e9797f27287721683be3a625af52332cb4551fbfe3e0c18cb6c130
      #|313/true/false:1e3506a9f3dd9803c428f7f55161950beb287c8a3dc0babbb9a267880b367693
      #|313/true/true:0243386a46b9bb82f95518e2c7e37b95d151bf8265c0a727024979a5f85bd8a0
      #|314/false/false:7e45179a15c95a4017e09fcaf890eee1539e152d99b3cb89d60294b5d9c9adc5
      #|314/false/true:4b0ed1f36e618b13b7e600c577ffe2026eccc64126fd25bd4b13df5c97c0b5b2
      #|314/true/false:169b75e2b38dceb5c41148abd3e724897e3c5f3062315e355e91aca525f6933a
      #|314/true/true:29ea29ce84018043eda08b58a52e2ccd887fd762b03520e262d303a07f156d00
      #|315/false/false:680bc3ee87b321be95e198abcb06573ab16d9072f3b71faca32adf774ac97e9b
      #|315/false/true:2b7a713eea44bd4ad1dd97c5cfb35016e869593001661d520f5319084df78169
      #|315/true/false:7d5da40b12033de6c61d8ac738d27f925a5fcce191fa0c488cbc834b4b3cc077
      #|315/true/true:2830ca98a88bdc4fde5c37493cc4bb54c864d617a3e50abf83bf06dbe0cea92e
      #|316/false/false:426d41ad35bb904677b5e1a65a1c420b2e834ba7447ed121ae3f931017ac26ac
      #|316/false/true:014272200a945e0c80e320699fb9e98d0c5d7ef5e604a2b1f9a8a59d0b31493f
      #|316/true/false:75eee6b547f7ddf0d2d1747cdc00dadd9ec978fb644ef76e450e5b348d797cc7
      #|316/true/true:4febc3420c58ee2a32fd110955f43f87a1c9d549b966faaf0cda68db71f6e0f6
      #|317/false/false:64906a9f7e0869d4bb95d3d9f0b86142e92f5b62799a0bbfa87e7d920c479c91
      #|317/false/true:38f19a26aa1fc16423670c2bf94951e7987f9dd8fb7fb86eab39978457c82f94
      #|317/true/false:1357428585b305a2d1496d61d1e80d1bca5a517b72cfcf86582ea479ad90ed83
      #|317/true/true:45cb91c47558d2defebc01593913301a58539a6c4b9e157a906dd9f59dda2a2a
    ),
  )
}
