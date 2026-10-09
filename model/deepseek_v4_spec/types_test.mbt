///|
test "DeepSeek V4 admits exact official profile identifiers" {
  let flash = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let pro = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Pro-Base",
  )
  let current = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  assert_eq(flash.profile(), Flash)
  assert_eq(flash.hidden_size(), 4096)
  assert_eq(flash.expert_storage(), Fp4E2M1)
  assert_eq(pro.profile(), ProBase)
  assert_eq(pro.hidden_size(), 7168)
  assert_eq(pro.routed_expert_count(), 384)
  assert_eq(pro.expert_storage(), Fp8E4M3)
  assert_true(current.has_dspark())
  assert_eq(current.dspark_block_size(), 5)
}

///|
test "DeepSeek V4 rejects aliases and unknown future profiles" {
  for
    hostile in [
      "DeepSeek-V4-Flash", "deepseek-ai/deepseek-v4-flash", "deepseek-ai/DeepSeek-V4-Flash-0731@main",
      "deepseek-ai/DeepSeek-V4-Pro-Next",
    ] {
    try
      @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(hostile)
    catch {
      UnsupportedProfile => ()
    } noraise {
      _ => fail("unsupported DeepSeek profile was admitted")
    }
  }
}

///|
test "DeepSeek V4 compression schedules are versioned and bounded" {
  let flash = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  assert_eq(flash.attention_profile_count(), 44)
  assert_eq(flash.compression_ratio(0), Some(0))
  assert_eq(flash.compression_ratio(2), Some(4))
  assert_eq(flash.compression_ratio(3), Some(128))
  assert_eq(flash.compression_ratio(42), Some(4))
  assert_eq(flash.compression_ratio(43), Some(0))
  assert_eq(flash.compression_ratio(-1), None)
  assert_eq(flash.compression_ratio(44), None)
  let pro = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Pro",
  )
  assert_eq(pro.attention_profile_count(), 62)
  assert_eq(pro.compression_ratio(0), Some(128))
  assert_eq(pro.compression_ratio(60), Some(4))
  assert_eq(pro.compression_ratio(61), Some(0))
}
