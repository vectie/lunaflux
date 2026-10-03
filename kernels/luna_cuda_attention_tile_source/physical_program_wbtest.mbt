///|
test "every supported attention family enters the typed physical dispatch" {
  assert_true(
    source_test_lowering(true, 100000L, 2, candidate_id=Some(100)).physical_program()
    is DirectKeyProgram(_),
  )
  assert_true(
    source_test_lowering(true, 100000L, 2, candidate_id=Some(310)).physical_program()
    is SharedScoreProgram(_),
  )
  assert_true(
    source_test_lowering(true, 100000L, 2, candidate_id=Some(318)).physical_program()
    is QueryFragmentProgram(_, _),
  )
  assert_true(
    source_test_decode_lowering(candidate_id=400).physical_program()
    is DirectKeyProgram(_),
  )
  assert_true(
    source_test_decode_lowering(candidate_id=420).physical_program()
    is GroupedKeyProgram(_, _),
  )
  for id in [430, 440, 441] {
    assert_true(
      source_test_decode_lowering(candidate_id=id).physical_program()
      is SplitKeyProgram(_, _),
    )
  }
}

///|
test "physical architecture source identities cover every terminal attention family" {
  let digests : Array[String] = []
  for id in [100, 290, 300, 310, 312, 313, 314, 316, 318, 322] {
    let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(id))
    digests.push(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "physical_merge",
        1048576,
      )
      .digest()
      .as_hex()
      .to_string(),
    )
  }
  for id in [400, 420, 430, 440, 441] {
    let lowering = source_test_decode_lowering(candidate_id=id)
    digests.push(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "physical_merge",
        1048576,
      )
      .digest()
      .as_hex()
      .to_string(),
    )
  }
  for id in [400, 430, 440, 441] {
    let lowering = source_test_decode_lowering(candidate_id=id, batch_rows=1)
    digests.push(
      emit_cuda_partitioned_attention_tile_source(
        lowering,
        source_test_abi(),
        "partial",
        "merge",
        67108864L,
        1048576,
      )
      .digest()
      .as_hex()
      .to_string(),
    )
  }
  debug_inspect(
    digests,
    content=(
      #|[
      #|  "c351bdafbafefbb7d85561d045766e854be0fac991e8b71969f913a735fd4d7d",
      #|  "528df762d0651a5cacb8a10c40da77d3e726be7f460d759f7ff04f96ee165941",
      #|  "2a422760163e9ea6c05bcf1ecce8152f5f8fcce965d95fbb13fabcad80ebd4d0",
      #|  "854d90fd40e70e2855f78c8d912b62ca683646a134dda20bc9912c3f084031ce",
      #|  "43fccb54e2febe4239ed675926d1a9651d6014d4c73482eaa949bf3c181b8e62",
      #|  "c688e556140075d8841ed58cb9eb419e6ccc722e91831803eec09c1574a2c818",
      #|  "abab0e1aabd8000a27d6db7bf64303c29744d124dfe4e53ac8139befb0902660",
      #|  "d54c66f2f1ace31e64091c75b4c090515c3b31da662bbea35fcb382c87c57018",
      #|  "b20484b8dc9aaa5ee6b4bc34aa878de1a94b1a32f300b92bb898d66d179a295c",
      #|  "f41c6487ec38a6be21055c9f9c2eb6b9d5c99b4ccc12389c7c0037305a307457",
      #|  "da09e6fd281d703b70cf3075ba81ef55046152d50b0d5d317b3d6191bff105f4",
      #|  "a8f2cc9b6f423958241517f62b73ea978a754aa2886636c014889850ff0b8fc5",
      #|  "db259a82363d53d4066f5f2757acfed24d82f5013e4d88bbec9d29c205ed44e5",
      #|  "687197295b3dac37d5fe85d69509bd21844bd33133e832abfba5514385caaa2d",
      #|  "db6fd557ed49ec7c8b783839d6364087bceda1d15b9df1590f0c3e07adfc3efb",
      #|  "560e7d3438c1e24a7504e2316987ef5c04256085a6c6fc1e9d7ff65587059ed9",
      #|  "4647e9b29d2d991ba7eb75791d6961762cebe8766e4fdcb803ee90366e99467b",
      #|  "977e499f0b6e191eb8330c52492b37806b480b11c833afba017b5ab36bf39d69",
      #|  "72fe2effd4cd73e97ffe9bea4d2b4320e6f235ad14198e92289730a848d68fea",
      #|]
    ),
  )
}
