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
      #|318/false/false:0a7020e81ebd27f99fb47e1590eb88e94068900371fb5004a57b64133ec74357
      #|318/false/true:94346dd688a08cbc153cb9d5e2407b8407995a82e615706c191c1c16c998cd15
      #|318/true/false:65e115ff6aa0eb0825146e825dc0c608791ae5d89d85e5b00d21cb64e3de1f58
      #|318/true/true:4abc95c4c19200885bc04a84bfe372b95718e793a58d35776e01c74afcff21ed
      #|319/false/false:22bd0d6ae8ef1847890b3de0a077ab9afc650113e99de2c640ae0a04ee39ca7f
      #|319/false/true:03b287d12d54c0f2b3f3c6006a5ca9eac3abb5989cba39e3cc2a56f3127aac41
      #|319/true/false:d9932443a192a2c1b0431fdfdb1ca2376a8cd25f4ce0fa37e739e9b2c04c36b7
      #|319/true/true:a6bebd193b1f4825bf5407203546ea3472829fcb6360f07d0bdd299fb76852f2
      #|320/false/false:ee4d4249ef45e6a3529c5a6201ca763f19bcc7ce225841fed1efde6ae384928f
      #|320/false/true:1d13a0bb8a5c8711768344f43833e469ca6fefec956463b54513ee3f0b3140bd
      #|320/true/false:77d7643c5b7069732b8be5686f4f9a746bab826f9ac182abdfd0de36ccce7937
      #|320/true/true:6adcc83bd7a714f6a7da6dcf76daabc093986745d333d84b00ecb1cde2d4c03b
      #|321/false/false:dea04660e042ad1faf3e02aca926d587f89f04c6bdb9fbacd035816d3f3f70d3
      #|321/false/true:6d053f96185a806ab2c3715d48ac050249969c5befacd75ec3f08a003787e045
      #|321/true/false:04f8522420b7f7d26a21251ab9c798a27616be4ab5461b2d48343210b22e6548
      #|321/true/true:ab7c429430f86afa2c4b063a1155c20147dab868506f10ea4881a6b6df0eda20
      #|322/false/false:b1f99b072b7e4da7a54d09c3be0129ed39084f77d9296561bd727ca016bab7fb
      #|322/false/true:ac338a19cc5c907b4613459229b7c7084401977b27414f88f0b69dece33a6e1b
      #|322/true/false:39f101a2b1562947df122d373c93a1ef0fb8ee85a3e93d7995a5c8292da4cc5b
      #|322/true/true:78394ca26734ad56a500e0896b1a175476259d38df6810a93b8b044f80ad46c1
      #|323/false/false:ef37a1d055339f07dbac84cc2d132893e4ff4b7dff45a6a6adf05d0b2b1ca687
      #|323/false/true:4e99baceaf3e8dfd9061cc42b5cbb034a9dba6d1f15920c75de8c0d3ab016dea
      #|323/true/false:2c0ecd9b5e75ecdf58c7c61bdd9dd56d153be56e64711dd03f6613dd3fec7bd2
      #|323/true/true:26f53cb1da95eccaeed98396c237cd60ad7fba2759519e4857e51164e8922a84
      #|324/false/false:83a0891ee5b7ee8cb0aedb46d328e9c169b0b414effaeb9f2a05db69578c302c
      #|324/false/true:89e9e0cb3c747fc9e8a85fca1bc530c8c428f0705d3513b480e0d862493408eb
      #|324/true/false:10657cba6a9ce454aebfda094fe0ec41eb3d78157a5e2dce6e1be02cce8482bb
      #|324/true/true:6b114878ae1bfce435d7676a5d46b9f33faa139968c5b8a80e1763c07eb79c62
      #|1003/false/false:d6325777d6c2efb187fdd77dbc08b8014ff2c6ef3c8ad98544c50e873b8939a8
      #|1003/false/true:1c203ab7f44c80a61a90a26b5f5ee935f71498ce034f82463f39f3b1cd7a21f8
      #|1003/true/false:0eb788a64ca58e8300d4cd3c1cdc6f83e000275ebd310142907220262c5ff3c5
      #|1003/true/true:626d3bf97e78b356822fd0c683a972f8b92bb28dec6daacbfb77fdd11f456ea4
      #|1004/false/false:0c7f89888ccd383f49ff16172fe785e2f4fb41ed68fe306f7e665fc8804ce91e
      #|1004/false/true:7067b3ebff196b5148d0abe34709843095fef68ffba74ac9f19300bc0779fc2d
      #|1004/true/false:1c268470490c66e7e369e5066c7f2e48c0af3b9af537df640fdb75d3d03062ea
      #|1004/true/true:e78c63e07e2976d92624e7cd34a6db20fb8a1f611fce4a793f7c88a36c077cc8
      #|1005/false/false:4dc12ce28ac846cf83aa412850d4611c85ee3e6f5b39d52c4e25d123b78a217b
      #|1005/false/true:40313f452f0f834753363f082215dbdbba968c6c430dc1f25ee5fad08d34dce4
      #|1005/true/false:3bbcc2457a1f9f8bd5fe890a0e61dea6ce373fb25cfa3e363a76c0f08e310902
      #|1005/true/true:7d6e748cdb36876afdf41e78d7f44f127ed081692b202ee42abee8c11111db3b
      #|1006/false/false:c9d2e9777b4a7aa0216e91ad792d5d448c019d8e8b8018f1c907de0b30102632
      #|1006/false/true:c74b3971c0650e87bdd3d3452d587e7734a7fd0d0c923ab0d99bf20db9f73d9d
      #|1006/true/false:162a6e83061ef8e4f4d9293bc9b30153c1f1d1014048d6370072b8e224b3add6
      #|1006/true/true:e6bd8cb893785db5aae690badfb503ad1885fa7d9d522754700b3b3a5b41ff97
      #|1008/false/false:c2cd82f27c0764b23573461008391a9ff2e30c37c2b317379f716a18d2ea93fa
      #|1008/false/true:8a6d31813bb1266dce80f0d5a39dfc66b996ac0ee679f74d13b792053da8e0bf
      #|1008/true/false:38ce794d2229a04cbe36ddb0dfd9576fe1d3795ba44779761353f77f6d87a96e
      #|1008/true/true:cbffad4bb680ee1a048a1936e91aa587144f0d153bb6525ad51f06edf9da2a85
    ),
  )
}
