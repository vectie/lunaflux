///|
test "query-owned forwarding deterministically composes with every effect schedule" {
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
      #|318/false/false:fb4f610488d4b812521e32376224837c7c74457d3e619b43d2af5425b383a77b
      #|318/false/true:fbdb0440aef5e8d3f6927327161ddaf5ba3beae6d769466a17c99417a0f3a325
      #|318/true/false:2c71f62aab51fd0fc400f46a42af40225ae2bdfc0a80194b4a3ad920430f2058
      #|318/true/true:ce05ad29a3a73829ce64d06007aaab7fb73862563e9b94393505f83e8956f788
      #|319/false/false:b1592e3cfaa631b51493381b7dcfe0a94e21731d00e8dcf08dbc21f4d28e4185
      #|319/false/true:0ac77371531af6cef940da7403bb45745e88612488b5aaff4e933f93669bf72e
      #|319/true/false:fa0a1b972b811e00fc2939f6c88880d23685ff8a8557b2d33ca737eeb02d31f1
      #|319/true/true:06a2b0b121dd5861e39abdddfff61c97fa9525090c958c50704826ea0e75fc69
      #|320/false/false:db9de9f48c3268e9f256699e656afaefe005f2167680cce68ec77d874a3f9680
      #|320/false/true:2ba34cb291b2d1dcbf18bed15204eb5d65c9ef1fa4a2b14ecfdb0acf62ac4348
      #|320/true/false:e49468e2db2900d46691daeeaef939ceede74a1945fb191646877a066a5c63b8
      #|320/true/true:aef54a8249d25aed0855a0b6aa48aa164f9de87025c3767c49e09587ccdc9b22
      #|321/false/false:89cb6ebf5f5ecf65d1f1d98ab85fda112e6f24523b0329cf9876f4b53b2f22e9
      #|321/false/true:c22b19d11306e539895c81cb9a3bed560dcd87ef8614de9a703a8b6595ceef82
      #|321/true/false:aefd4b4b886539e6fb418832e1084d138dfc7123d15eedfb683b3379285bac85
      #|321/true/true:fd5b68810bb8882257a4401505114584e5d3340d9aebab45c74a13e5f6316582
      #|322/false/false:9dc27f9421d9da2efb9e6bf30db3fa5dc3fbf7c565f26e0f50c15bb0ff858421
      #|322/false/true:0fc3074cbbc175268e5f9e9f910bdc778177c94ca88367d71dceba87a35b81c6
      #|322/true/false:19a77236fa10252549b633af09ca817cb111ff0aee8c88226447648b808f776c
      #|322/true/true:d3cadf36b28cf693a26c6bf324efc2f0ff858267be767e716b8b12d53e1ed281
      #|323/false/false:777998da08c6459f1160db6ecbfb0669f71d8b98a7bbc85dff8ecdcc6347225d
      #|323/false/true:2a3311398d5352ff1a579808de56e6852dd61dfabe6981be7f4f6ba0f4056cb5
      #|323/true/false:c7b546d1ba9b1c4332c94d511caa8b4687b06b2f1fd3119d76cadc70a9cb4fcf
      #|323/true/true:182ea366500de72b99c2b69c29553e0f7b0229c69db163a367d58ce069bcadc9
      #|324/false/false:104389d705ae16e09ec360f0e82d8dc3ed07adfcab863ce5c79755e329ebb39a
      #|324/false/true:b0d43a64e9ea0190efa056a3dfa09163abd3e6606f62777a84a77cce83149d2c
      #|324/true/false:1941853f38fb761c4d06b4bf19ec20d0359755da0b5391e7298dd96a65c6f7f5
      #|324/true/true:8f946b48bb7a84323c8ab83bbc7abd97fbf8a195bd557c1bf93db6c25a18803a
      #|1003/false/false:e93a1ea1deb505003564c924a88f4f9000d219ccf00fedd336790fcbc50ca63f
      #|1003/false/true:f96bff993e621650a49e48bd9965d2e73631a9dd0ce5c0710bf07c90196c1305
      #|1003/true/false:fe6ad163dcadb85a22dcb0130e3f6d1a6571857c729e67d5912a493fdc3f291b
      #|1003/true/true:e424e38282e2d2b7fc6778009d1ed1708cdbc6f22686ff8a234d0f22d3581644
      #|1004/false/false:052fe31b11fda9b4aa6f98d10456add5e0e60f75156e3494fb96b46965bbd595
      #|1004/false/true:883220c10333115272d12684d3bb6ad0d0e76a0d0a41ce78db596ae58bb1423c
      #|1004/true/false:8c125a83b2d30e0d2090853309ffd503f9cb95236f714992d499ce0f1850f537
      #|1004/true/true:5059ed25a2a162b03ccec08b89bf7ec2717087e879c2ec23a018b05648333301
      #|1005/false/false:e346bec2ca488f4c9d8dceeb3eb005733abc2633a782ea5d39c6327e2b3f1131
      #|1005/false/true:edec5e69f4ae6567b9e79a720dd48c29395d511f856ce62bb62c343067debf52
      #|1005/true/false:4210b70ab20f9ae89c987aa6b496455c21f5c5bd7ed083e2600b0d379403a583
      #|1005/true/true:b767f4e573610ee48a1ff1792ad38331864f3eb81288f566856d76a5f7c1f822
      #|1006/false/false:451000aa43142a5ae65122450cc3c04a2bc35a143d6bf0d3d1cc8eed5a0a709d
      #|1006/false/true:aa3081eef3cef6e80f7c17cbc5f96d4fae32269a0d355b9a81cfe44736aa8a24
      #|1006/true/false:80228b9abfe3a9c5ae57b9f9730eac4bdad6aa165901dc314551e2edf0475965
      #|1006/true/true:893a2e1b54632048a6d9162974aba9f77787df14b7c16ee64dc1227b3679e2b1
      #|1008/false/false:279ed9c5edec0164069c92810735937deec6174618217b6d5f1e2ff7abdde46a
      #|1008/false/true:64fa0d25edb3cdbbffa028448febbb070cf4d4720f890eb86175f1f3986701b9
      #|1008/true/false:59b30d37bbb06917c51e9252e98fcc3d617f5e927fce3e01e6b580fed71e0ae3
      #|1008/true/true:dc1a82cae83f03280465a61ead4ccda0be237738601448061728b6b97f662050
    ),
  )
}
