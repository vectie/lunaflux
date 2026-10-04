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
      #|  "fc6c65f79200e47cd2b03c977d879796e7339899bc4ea75ded5c7e7ac573ff76",
      #|  "03fabc079d24f69eea85701f93c8914ea2bbb5270f05841a55d212e972fadd64",
      #|  "8cd235f80081301cbf5f9a29d15645d8092bb0df826065ef3189e73457f4c1ed",
      #|  "113b6c546e57b39e4bb97b7406cb51f55a7e17341891d6e99ca7e51f697ebe55",
      #|  "bfacbbf497d203dba86e106b01581cffe86b797cc11c4356d477ce15d23e3353",
      #|  "e61dbb3f76cfc90d8556b2b4b8edd7e82814ef092230f633e17d9afb7e0e054a",
      #|  "1f657456c3938d6889703163302a822437c50e1caa3787f49c4cf94205c279d2",
      #|  "12a4f48efd456dd7c474dfa7ca760e0903ced0beb226d8352dbf097c31d06740",
      #|  "f5c537a351933a36054014374edcbd59d66ea83998eb0e69a7dfe794ef7bee25",
      #|  "36b7c9d426ad7ef133529fd1288b4fee3ef2ce228b96dadb34a69454b9bf861f",
      #|  "9168e09ef620701ecfed050ce41ccc3993090a524f64d1fd089ab73ff7816fce",
      #|  "71c39f43ae644b1c8b11b830be4c2aed4e13442b65318605afffd4d14ad7707d",
      #|  "543444eebfc275a56aee76d0695ea7409b647832452852fa549746fffa3255d8",
      #|  "ed7de1b397b364fd20352ee606aca684b266100f342aa35cd96f1445295b5c79",
      #|  "60db91adbc3c8d4433f1a9dabe7d9c73d54496eb84b52b66dc4ac31678550d76",
      #|  "c47fd74dbc6dfa0f78ed5e6d17fe0ef6753558fd53be1e067c1d15fbc7dc6a7e",
      #|  "6891027dbbfacf28b8872bdee45f1126aae93583fcbdd1bcfa49f3d76280ce6c",
      #|  "9d5cce96f8e54e550d48650e19636d77c560ca12a02b51e2b9eece4f76763435",
      #|]
    ),
  )
}
