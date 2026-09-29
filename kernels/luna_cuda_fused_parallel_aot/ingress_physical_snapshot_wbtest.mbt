///|
test "physical refinement preserves full ingress source" {
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
      #|  "78eb7b6ba774def2b9ec2d41bf927e2fe6a14ddb73fe5ccbe4e0c57bd1bea471",
      #|  "c5c8a9550cb6926319a114e98b16ec6155c50ec2ee5dcd49db7013e42d43aaf9",
      #|  "a6c9831881a2ffc61a151271134e341abb32a5fc4ff11ffe0166fcbedca13b33",
      #|  "74f2faf0906c0f68c88e7546fd547777156a8b06d7eda78cba4481e3315ee02f",
      #|  "09484f317de030e9f29df2aca641dd5eabd0668447636a5510ab4a29d407e941",
      #|  "31a03cf66f197dbac1121796f08521472c2f71d81b8f73ded795d6eeefe3c767",
      #|  "98f9631419598092679b30f17bed533545a7bb7fba3da185bc0246bbb010a1b1",
      #|  "6c9d0caf46c5a78de8a5eaddc4e6cb206416c7d39c939863a2a794e7a422758f",
      #|  "111a47d65287866d40bb7e59af200cf0a4fc64d4c31d04dd972c8b8c6a23327d",
      #|  "814f3ed04b1bbdcb3210407947e6ae995445c910ecd40e69f0377ef1ad57fd7a",
      #|  "bb5de0b664bf9164c79cc5781b220adc10709dbbbfed276cabcdc1d0c9ab4144",
      #|  "ea63122afe051ac66d103d7fb7d91142b4b207f9f4b8c242bd4bcf16516b1272",
      #|  "db96aca7bf4149c6b9d57078e61d8f16a4cd533a07f36a6beab37ac0045f2b0b",
      #|  "439f5f1046ba9e6c8e97a57c4434335a608217f32c9a3ac00e8523edd550e3d4",
      #|  "407606bd1323bd695ba552f79a31dc3d4f38857ea44101a87f3cf8b7209e07f7",
      #|  "78f5c3d00e1b0bcc1163e098c98cf28c715e03329c18407bdb58a4827ee467a0",
      #|  "f81930b8807c6f84dad32942df79c70ef191eaf744c1e0acd1eb822a059f8cf3",
      #|  "94c5edfb414d3bc0f3e9dfd2f26a730fb213f907c9fc7aa2f7fcb5f530ca3592",
      #|]
    ),
  )
}
