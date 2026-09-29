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
      #|  "d37e5fac7cc9e7cad2760ac1dae234cb341f6e5b739bbe50768554c078aecfe2",
      #|  "7375d993fe15c65227f4248f2908f0d17821ba5c52d1019e1d6aa773f918a630",
      #|  "1bf825644314d632af6b024a5d2ffec2303f8f9a1745863da9c959a7aadc356d",
      #|  "78fd8aa902e55ca0cb056d98978c5a192be0daa9e05131157c713563074ff5f1",
      #|  "a5059a69ee78809c154423725eefc95793b174b25cd47962efeab58d3391218e",
      #|  "51fabdb5a56f39c84223fafbe80609a82caa0a9a6180881fde5bfa3060678f98",
      #|  "728b7fad96a567c0b348886ce40b4c1a6d70272273745e0670d4a34abec6ada8",
      #|  "b20738780ff737b7b2f78057960c74deee0304d2e22b38877c2a69e5bf7ccf29",
      #|  "454162dee03c7ee807a8c5547d00cf624f8ef25291a65a38a0486420c355a0d9",
      #|  "a23f212de37a8116ace66a78f5e48af1e40572bb8e4bdc89698ae3960b136a68",
      #|  "548907935e574c9f65ac53cf71628ca5675e3f4c8d1926e97f2f8c4363ad00c1",
      #|  "daf3660c33c033ef68a868de783232e72b2a26ea0a411bb9930aff64b820126b",
      #|  "0a600671bcf0ccde8627b17fb74c86ddfd451d36c46288f8dd98bb3dc1f2ded0",
      #|  "d3096dbbd8d86e01b3e4caac5073feb973c8653007d3c102a90a853966f4d560",
      #|  "2f0e0f716df520c204074cad7728b3a80af815a269ec32b40024958e85e8cd43",
      #|  "3b9743ee7331c260ae365fae410179b8a52b3e66f8d4d91701c883d44c82243d",
      #|  "72566793663b2e84ec1ea983a399a65c171c88dae85fb9ce5cc6de5741c74c0b",
      #|  "89a92819d0947b19e8cc360aaacabc06712207f5b9eff1c507161d3c58ffcafd",
      #|]

    ),
  )
}
